# Copyright Reexpress AI, Inc. All rights reserved.

import torch

from pathlib import Path
import time
import os

import baseline_utils_train_main
import uuid
import constants
import utils_preprocess

import data_validator


def train_iterative_main(options, rng, main_device=None):
    import baseline_utils_model
    start_time = time.time()

    if not options.eval_only:
        best_shuffle_index = 0

        min_calibration_balanced_sdm_loss = torch.inf
        min_calibration_balanced_sdm_loss_shuffle_iteration = -1

        assert options.number_of_random_shuffles >= 0
        for shuffle_index in range(max(options.number_of_random_shuffles, 1)):
            if options.continue_training:
                model = baseline_utils_model.load_baseline_model_torch(options.model_dir, torch.device("cpu"))
                print(f"Continuing training from the model stored in {options.model_dir}")
            else:
                model = None
            path = Path(options.model_dir, f"{shuffle_index}")
            path.mkdir(parents=False, exist_ok=True)
            shuffle_index_model_dir = str(path.as_posix())

            if not options.do_not_shuffle_data:
                best_iteration_data_path = Path(options.model_dir, "best_iteration_data")
                best_iteration_data_path.mkdir(parents=False, exist_ok=True)
                best_iteration_data_dir = str(best_iteration_data_path.as_posix())

                print(f"Current D_tr, D_ca shuffle index {shuffle_index}")
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
                train_file = options.input_training_set_file
                calibration_file = options.input_calibration_set_file
                if options.load_train_and_calibration_from_best_iteration_data_dir:
                    best_iteration_data_path = Path(options.model_dir, "best_iteration_data")
                    best_iteration_data_dir = str(best_iteration_data_path.as_posix())

                    train_file = os.path.join(best_iteration_data_dir, "train.jsonl")
                    calibration_file = os.path.join(best_iteration_data_dir, "calibration.jsonl")

                train_meta_data, training_embedding_summary_stats = utils_preprocess.get_metadata_lines(options, train_file,
                                                     reduce=False,
                                                     use_embeddings=options.use_embeddings,
                                                     concat_embeddings_to_attributes=options.concat_embeddings_to_attributes,
                                                     calculate_summary_stats=True, is_training=True)
                calibration_meta_data, _ = utils_preprocess.get_metadata_lines(options, calibration_file,
                                                           use_embeddings=options.use_embeddings,
                                                           concat_embeddings_to_attributes=options.concat_embeddings_to_attributes,
                                                           calculate_summary_stats=False, is_training=False)

            train_embeddings = train_meta_data["embeddings"].to(main_device)
            calibration_embeddings = calibration_meta_data["embeddings"].to(main_device)
            train_labels = torch.tensor(train_meta_data["labels"]).to(main_device)
            calibration_labels = torch.tensor(calibration_meta_data["labels"]).to(main_device)

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
                            "cdfThresholdTolerance": constants.defaultCdfThresholdTolerance,
                            "exemplar_vector_dimension": options.exemplar_vector_dimension,
                            "trueClass_To_dCDF": None,
                            "trueClass_To_qCumulativeSampleSizeArray": None,
                            "hr_output_thresholds": None,
                            "hr_class_conditional_accuracy": 0.0,
                            "alpha": options.alpha,
                            "maxQAvailableFromIndexer": maxQAvailableFromIndexer,
                            "calibration_training_stage": 0,
                            "min_rescaled_similarity_to_determine_high_reliability_region": torch.inf,
                            "training_embedding_summary_stats": training_embedding_summary_stats,
                            # the following can all be None at test-time to save memory, if desired:
                            "calibration_labels": calibration_labels,  # torch tensor
                            "calibration_predicted_labels": None,
                            "calibration_uuids": calibration_meta_data["uuids"],
                            "calibration_sdm_outputs": None,
                            "calibration_rescaled_similarity_values": None,
                            "calibration_is_ood_indicators": None,
                            "is_sdm_network_verification_layer": options.is_sdm_network_verification_layer,
                            "train_trueClass_To_dCDF": None
                            }

            min_held_out_balanced_cross_entropy_loss, min_held_out_balanced_cross_entropy_loss_epoch = \
                baseline_utils_train_main.train(options, train_embeddings=train_embeddings,
                                                calibration_embeddings=calibration_embeddings,
                                                train_labels=train_labels,
                                                calibration_labels=calibration_labels,
                                                model_params=model_params,
                                                main_device=main_device,
                                                model_dir=shuffle_index_model_dir, model=model)
            one_shuffle_index__min_dev_balanced_sdm_loss = min_held_out_balanced_cross_entropy_loss

            if one_shuffle_index__min_dev_balanced_sdm_loss <= min_calibration_balanced_sdm_loss:
                min_calibration_balanced_sdm_loss = one_shuffle_index__min_dev_balanced_sdm_loss
                min_calibration_balanced_sdm_loss_shuffle_iteration = shuffle_index

            # Note that for simplicity we are overloading the variable naming here. This is SDM with d=1, q=e-2.
            print(f"Min calibration balanced CROSS-ENTROPY loss (used to determine shuffle index: "
                  f"True) of "
                  f"{min_calibration_balanced_sdm_loss} at "
                  f"shuffle index {min_calibration_balanced_sdm_loss_shuffle_iteration}")

            save_this_shuffle_index = min_calibration_balanced_sdm_loss_shuffle_iteration == shuffle_index

            if save_this_shuffle_index:
                # load best epoch (still same shuffle index) in order to re-save to the best iteration directory,
                # which is currently the parent directory:
                best_shuffle_index = shuffle_index
                model = baseline_utils_model.load_baseline_model_torch(shuffle_index_model_dir, torch.device("cpu"))
                baseline_utils_model.save_baseline_model(model, options.model_dir)
                print(f"Saved current index ({shuffle_index}) as the best shuffle iteration in the parent directory: "
                      f"{options.model_dir}")

                if not options.do_not_shuffle_data and not options.do_not_resave_shuffled_data:
                    baseline_utils_model.save_json_lines(os.path.join(best_iteration_data_dir, "train.jsonl"),
                                                         train_data_json_list)
                    baseline_utils_model.save_json_lines(os.path.join(best_iteration_data_dir, "calibration.jsonl"),
                                                         calibration_data_json_list)

            cumulative_time = time.time() - start_time
            print(f"Cumulative running time: {cumulative_time}")
            print(f"Average running time per shuffle iteration: {cumulative_time/(shuffle_index+1)} out of "
                  f"{shuffle_index+1} iterations.")
        print(f"Best overall shuffle index: {best_shuffle_index}.")