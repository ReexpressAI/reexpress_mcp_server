# Copyright Reexpress AI, Inc. All rights reserved.

import torch

from pathlib import Path
import time
import os

import utils_train_main
import uncertainty_statistics
import uuid
import constants
import utils_model
import utils_preprocess

import data_validator


def train_iterative_main(options, rng, main_device=None):
    start_time = time.time()

    if options.batch_size == options.eval_batch_size:
        print(f"Note: The eval batch size is the same as the training batch size. "
              f"Consider increasing the eval batch size for improved efficiency during training.")

    global_uncertainty_statistics = \
        uncertainty_statistics.UncertaintyStatistics(
            globalUncertaintyModelUUID=str(uuid.uuid4()),
            numberOfClasses=options.class_size,
            min_rescaled_similarity_across_iterations=None,
            max_hr_region_alpha_across_iterations=None
        )

    if not options.eval_only:
        best_shuffle_index = 0
        max_calibration_balanced_accuracy = 0
        max_calibration_balanced_accuracy_shuffle_iteration = -1

        max_calibration_balanced_mean_q = 0
        max_calibration_balanced_mean_q_shuffle_iteration = -1

        min_calibration_balanced_sdm_loss = torch.inf
        min_calibration_balanced_sdm_loss_shuffle_iteration = -1

        assert options.number_of_random_shuffles >= 0
        for shuffle_index in range(max(options.number_of_random_shuffles, 1)):
            if options.continue_training:
                model = utils_model.load_model_torch(options.model_dir, torch.device("cpu"))
                print(f"Continuing training from the model stored in {options.model_dir}")
            else:
                model = None
            path = Path(options.model_dir, f"{shuffle_index}")
            path.mkdir(parents=False, exist_ok=True)
            shuffle_index_model_dir = str(path.as_posix())

            inputs_are_converted_dataset_directories = \
                utils_preprocess.is_reexpress_dataset_directory(options.input_training_set_file) and \
                utils_preprocess.is_reexpress_dataset_directory(options.input_calibration_set_file)
            if not options.do_not_shuffle_data:
                best_iteration_data_path = Path(options.model_dir, "best_iteration_data")
                best_iteration_data_path.mkdir(parents=False, exist_ok=True)
                best_iteration_data_dir = str(best_iteration_data_path.as_posix())

                print(f"Current D_tr, D_ca shuffle index {shuffle_index}")
                if inputs_are_converted_dataset_directories:
                    # The shuffle is an index permutation over the memory-mapped converted sources: the data
                    # is neither re-parsed nor copied as raw documents, and only the gathered split tensors
                    # are materialized. The (tiny) index tensors fully determine the split and are saved for
                    # the best iteration (below), in place of duplicated data copies.
                    dataset_sources = [
                        utils_preprocess.load_reexpress_dataset_source(options.input_training_set_file),
                        utils_preprocess.load_reexpress_dataset_source(options.input_calibration_set_file)]
                    total_n = sum(dataset_source["n"] for dataset_source in dataset_sources)
                    shuffled_indices = torch.from_numpy(rng.permutation(total_n))
                    train_split_indices = shuffled_indices[0:total_n // 2]
                    calibration_split_indices = shuffled_indices[total_n // 2:]
                    train_meta_data, training_embedding_summary_stats = \
                        utils_preprocess.get_meta_data_dict_and_summary_stats_for_training_split(
                            options,
                            utils_preprocess.gather_split_from_dataset_sources(
                                dataset_sources, train_split_indices, include_documents=False))
                    calibration_split = utils_preprocess.gather_split_from_dataset_sources(
                        dataset_sources, calibration_split_indices, include_documents=False)
                    calibration_meta_data = utils_preprocess.get_meta_data_dict(
                        calibration_split["embeddings"], calibration_split["labels"].tolist(),
                        calibration_split["uuids"], [""] * len(calibration_split["uuids"]))
                else:
                    # Generally speaking, the training file should have balanced labels, but we do not currently enforce this when
                    # randomly shuffling. If your dataset is unbalanced, currently you will need to manually shuffle.
                    all_data = utils_preprocess.get_data(options.input_training_set_file)
                    all_data.extend(utils_preprocess.get_data(options.input_calibration_set_file))
                    rng.shuffle(all_data)
                    # this gets resaved if best epoch
                    train_data_json_list = all_data[0:len(all_data)//2]
                    calibration_data_json_list = all_data[len(all_data)//2:]
                    train_meta_data, training_embedding_summary_stats = utils_preprocess.get_metadata_lines_from_json_list(options, train_data_json_list,
                                                                        reduce=False,
                                                                        use_embeddings=options.use_embeddings,
                                                                        concat_embeddings_to_attributes=options.concat_embeddings_to_attributes,
                                                                        calculate_summary_stats=True, is_training=True)
                    calibration_meta_data, _ = utils_preprocess.get_metadata_lines_from_json_list(options, calibration_data_json_list,
                                                                              use_embeddings=options.use_embeddings,
                                                                              concat_embeddings_to_attributes=options.concat_embeddings_to_attributes,
                                                                              calculate_summary_stats=False, is_training=False)

            else:
                train_meta_data = None
                train_file = options.input_training_set_file
                calibration_file = options.input_calibration_set_file
                if options.load_train_and_calibration_from_best_iteration_data_dir:
                    best_iteration_data_path = Path(options.model_dir, "best_iteration_data")
                    best_iteration_data_dir = str(best_iteration_data_path.as_posix())
                    split_indices_file = os.path.join(best_iteration_data_dir,
                                                      constants.FILENAME_BEST_ITERATION_SPLIT_INDICES)
                    if os.path.exists(split_indices_file):
                        # Training runs with converted dataset directories save the split indices, rather
                        # than duplicated data copies; reconstruct the splits from the indices:
                        train_split, calibration_split = \
                            utils_preprocess.load_best_iteration_splits(options.model_dir)
                        train_meta_data, training_embedding_summary_stats = \
                            utils_preprocess.get_meta_data_dict_and_summary_stats_for_training_split(
                                options, train_split)
                        calibration_meta_data = utils_preprocess.get_meta_data_dict(
                            calibration_split["embeddings"], calibration_split["labels"].tolist(),
                            calibration_split["uuids"], [""] * len(calibration_split["uuids"]))
                    else:
                        train_file = os.path.join(best_iteration_data_dir, "train.jsonl")
                        calibration_file = os.path.join(best_iteration_data_dir, "calibration.jsonl")

                if train_meta_data is None:
                    train_meta_data, training_embedding_summary_stats = utils_preprocess.get_metadata_lines(options, train_file,
                                                         reduce=False,
                                                         use_embeddings=options.use_embeddings,
                                                         concat_embeddings_to_attributes=options.concat_embeddings_to_attributes,
                                                         calculate_summary_stats=True, is_training=True)
                    calibration_meta_data, _ = utils_preprocess.get_metadata_lines(options, calibration_file,
                                                               use_embeddings=options.use_embeddings,
                                                               concat_embeddings_to_attributes=options.concat_embeddings_to_attributes,
                                                               calculate_summary_stats=False, is_training=False)

            train_embeddings = train_meta_data["embeddings"]
            calibration_embeddings = calibration_meta_data["embeddings"]
            train_labels = torch.tensor(train_meta_data["labels"])
            calibration_labels = torch.tensor(calibration_meta_data["labels"])

            assert train_embeddings.shape[0] == train_labels.shape[0], f"{train_embeddings.shape[0]}, {train_labels.shape[0]}"
            assert calibration_embeddings.shape[0] == calibration_labels.shape[0], f"{calibration_embeddings.shape[0]}, {calibration_labels.shape[0]}"
            assert train_embeddings.shape[1] == calibration_embeddings.shape[1], f"{train_embeddings.shape[1]}, {calibration_embeddings.shape[1]}"

            print(f"train_embeddings.shape: {train_embeddings.shape}")
            print(f"calibration_embeddings.shape: {calibration_embeddings.shape}")

            for class_label in range(options.class_size):
                print(f"Training class {class_label}: {len([x for x in train_meta_data['labels'] if x == class_label])} documents")

            maxQAvailableFromIndexer = options.maxQAvailableFromIndexer
            if options.use_training_set_max_label_size_as_max_q:
                max_training_set_label_cardinality = 0
                label_set_cardinality = {}
                for label in range(options.class_size):
                    label_set_cardinality[label] = 0
                for label in train_labels:
                    if data_validator.isKnownValidLabel(label=label, numberOfClasses=options.class_size):
                        label = label.item()
                        label_set_cardinality[label] += 1
                for label in range(options.class_size):
                    print(f"Training label {label} support cardinality: {label_set_cardinality[label]}")
                    if label_set_cardinality[label] > max_training_set_label_cardinality:
                        max_training_set_label_cardinality = label_set_cardinality[label]
                maxQAvailableFromIndexer = max_training_set_label_cardinality
            print(f"Considering a max q value up to {maxQAvailableFromIndexer}")
            model_params = {"version": constants.ProgramIdentifiers_version,
                            "uncertaintyModelUUID": str(uuid.uuid4()),
                            "numberOfClasses": options.class_size,
                            "embedding_size": train_meta_data["embedding_size"] if "embedding_size" in train_meta_data else train_embeddings.shape[1],
                            "train_labels": train_labels.cpu(),
                            "train_predicted_labels": None,
                            "train_uuids": train_meta_data["uuids"],
                            "exemplar_vector_dimension": options.exemplar_vector_dimension,
                            "trueClass_To_dCDF": None,
                            "trueClass_To_qCumulativeSampleSizeArray": None,
                            "maxQAvailableFromIndexer": maxQAvailableFromIndexer,
                            "calibration_training_stage": 0,
                            "training_embedding_summary_stats": training_embedding_summary_stats,
                            "is_sdm_network_verification_layer": options.is_sdm_network_verification_layer,
                            "alpha_resolution": options.alpha_resolution,
                            "hr_regions": None,
                            # the following can all be None at test-time to save memory, if desired:
                            "calibration_labels": calibration_labels,  # torch tensor
                            "calibration_predicted_labels": None,
                            "calibration_uuids": calibration_meta_data["uuids"],
                            "calibration_sdm_outputs": None,
                            "calibration_rescaled_similarity_values": None,
                            "calibration_is_ood_indicators": None,
                            "train_trueClass_To_dCDF": None
                            }
            one_shuffle_index__max_dev_balanced_acc, one_shuffle_index_max_dev_balanced_mean_q, \
                one_shuffle_index__min_dev_balanced_sdm_loss, \
                one_shuffle_index__min_rescaled_similarity_to_determine_high_reliability_region,\
                one_shuffle_index__max_hr_region_alpha = \
                utils_train_main.train(options, train_embeddings=train_embeddings,
                                       calibration_embeddings=calibration_embeddings,
                                       train_labels=train_labels,
                                       calibration_labels=calibration_labels,
                                       model_params=model_params,
                                       main_device=main_device,
                                       model_dir=shuffle_index_model_dir, model=model,
                                       shuffle_index=shuffle_index)

            global_uncertainty_statistics.update_high_reliability_region_stats(
                min_rescaled_similarity_to_determine_high_reliability_region=
                one_shuffle_index__min_rescaled_similarity_to_determine_high_reliability_region,
                max_hr_region_alpha=one_shuffle_index__max_hr_region_alpha)
            if one_shuffle_index__max_dev_balanced_acc >= max_calibration_balanced_accuracy:
                max_calibration_balanced_accuracy = one_shuffle_index__max_dev_balanced_acc
                max_calibration_balanced_accuracy_shuffle_iteration = shuffle_index

            print(f"///////////////Training shuffle iteration {shuffle_index} summary///////////////")
            print(f"Max calibration balanced accuracy (used to determine shuffle index: "
                  f"{False}) of {max_calibration_balanced_accuracy} at "
                  f"shuffle index {max_calibration_balanced_accuracy_shuffle_iteration}")

            if one_shuffle_index_max_dev_balanced_mean_q >= max_calibration_balanced_mean_q:
                max_calibration_balanced_mean_q = one_shuffle_index_max_dev_balanced_mean_q
                max_calibration_balanced_mean_q_shuffle_iteration = shuffle_index

            print(f"Max calibration balanced mean q (used to determine shuffle index: "
                  f"{False}) of {max_calibration_balanced_mean_q} at "
                  f"shuffle index {max_calibration_balanced_mean_q_shuffle_iteration}")

            if one_shuffle_index__min_dev_balanced_sdm_loss <= min_calibration_balanced_sdm_loss:
                min_calibration_balanced_sdm_loss = one_shuffle_index__min_dev_balanced_sdm_loss
                min_calibration_balanced_sdm_loss_shuffle_iteration = shuffle_index

            print(f"Min calibration balanced SDM loss (used to determine shuffle index: "
                  f"{True}) of "
                  f"{min_calibration_balanced_sdm_loss} at "
                  f"shuffle index {min_calibration_balanced_sdm_loss_shuffle_iteration}")

            save_this_shuffle_index = min_calibration_balanced_sdm_loss_shuffle_iteration == shuffle_index

            if save_this_shuffle_index:
                # load best epoch (still same shuffle index) in order to re-save to the best iteration directory,
                # which is currently the parent directory:
                best_shuffle_index = shuffle_index
                model = utils_model.load_model_torch(shuffle_index_model_dir, torch.device("cpu"))
                utils_model.save_model(model, options.model_dir)
                print(f"Saved current index ({shuffle_index}) as the best shuffle iteration in the parent directory: "
                      f"{options.model_dir}")

                if not options.do_not_shuffle_data and not options.do_not_resave_shuffled_data:
                    if inputs_are_converted_dataset_directories:
                        # The (tiny) split indices, with the source directories, fully determine the best
                        # iteration's splits; no data is duplicated. See
                        # utils_preprocess.load_best_iteration_splits() and
                        # utils_preprocess.load_best_iteration_calibration_split() for the reconstruction.
                        torch.save({constants.STORAGE_KEY_version: constants.ProgramIdentifiers_version,
                                    constants.STORAGE_KEY_SPLIT_train_indices: train_split_indices,
                                    constants.STORAGE_KEY_SPLIT_calibration_indices: calibration_split_indices,
                                    constants.STORAGE_KEY_SPLIT_source_directories:
                                        [options.input_training_set_file, options.input_calibration_set_file]},
                                   os.path.join(best_iteration_data_dir,
                                                constants.FILENAME_BEST_ITERATION_SPLIT_INDICES))
                    else:
                        utils_model.save_json_lines(os.path.join(best_iteration_data_dir, "train.jsonl"),
                                                    train_data_json_list)
                        utils_model.save_json_lines(os.path.join(best_iteration_data_dir, "calibration.jsonl"),
                                                    calibration_data_json_list)
            # the running global uncertainty statistics are saved in the main directory after every iteration:
            utils_model.save_global_uncertainty_statistics(global_uncertainty_statistics, options.model_dir)

            cumulative_time = time.time() - start_time
            print(f"Cumulative running time: {cumulative_time}")
            print(f"Average running time per shuffle iteration: {cumulative_time/(shuffle_index+1)} out of "
                  f"{shuffle_index+1} iterations.")
        print(f"Best overall shuffle index: {best_shuffle_index}.")
