# Copyright Reexpress AI, Inc. All rights reserved.

import torch

import numpy as np
import argparse

import constants
import utils_train_iterative_main
import utils_test_batch
import utils_update
import utils_calibrate

def main():
    parser = argparse.ArgumentParser(allow_abbrev=False, description="-----[Train and eval sdm estimators]-----")

    parser.add_argument("--input_training_set_file", default="",
                        help=".jsonl format, or a converted dataset directory "
                             "(see convert_to_reexpress_dataset.py)")
    parser.add_argument("--input_calibration_set_file", default="",
                        help=".jsonl format, or a converted dataset directory "
                             "(see convert_to_reexpress_dataset.py)")
    parser.add_argument("--input_eval_set_file", default="",
                        help=".jsonl format, or a converted dataset directory "
                             "(see convert_to_reexpress_dataset.py)")
    parser.add_argument("--eval_on_best_iteration_calibration_split", default=False, action='store_true',
                        help="Evaluate on the calibration split of the best training iteration, reconstructed "
                             "from the split indices saved in model_dir/best_iteration_data. (Available when "
                             "training used converted dataset directories as input.) --input_eval_set_file "
                             "is ignored in this case.")

    parser.add_argument("--class_size", default=2, type=int, help="class_size")
    parser.add_argument("--seed_value", default=0, type=int, help="seed_value")
    parser.add_argument("--use_json_input_instead_of_torch_file", default=False, action='store_true',
                        help="use_json_input_instead_of_torch_file")
    parser.add_argument("--epoch", default=20, type=int, help="number of max epoch")
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size during training")
    parser.add_argument("--eval_batch_size", default=50, type=int,
                        help="Batch size during evaluation. "
                             "This can (and should) typically be larger than the training batch size for efficiency.")
    parser.add_argument("--learning_rate", default=0.00001, type=float, help="learning rate")

    parser.add_argument("--maxQAvailableFromIndexer", default=constants.maxQAvailableFromIndexer, type=int,
                        help="max q considered")
    parser.add_argument("--use_training_set_max_label_size_as_max_q", default=False, action='store_true',
                        help="use_training_set_max_label_size_as_max_q")

    parser.add_argument("--eval_only", default=False, action='store_true', help="eval_only")

    parser.add_argument("--model_dir", default="",
                        help="model_dir")

    parser.add_argument("--use_embeddings", default=False, action='store_true', help="")
    parser.add_argument("--concat_embeddings_to_attributes", default=False, action='store_true', help="")

    parser.add_argument("--number_of_random_shuffles", default=20, type=int,
                        help="number of random shuffles of D_tr and D_ca, each of which is associated with a new"
                             " f(x) := o of g of h(x), where h(x) is held frozen")
    parser.add_argument("--do_not_shuffle_data", default=False, action='store_true',
                        help="In this case, the data is not shuffled. If --number_of_random_shuffles > 1, "
                             "iterations can still occur (to assess variation in learning, but the data stays fixed. "
                             "Generally speaking, it's recommended to shuffle the data.")
    parser.add_argument("--is_training_support", default=False, action='store_true',
                        help="Include this flag if the eval set is the training set. "
                             "This ignores the first match when calculating uncertainty, under the assumption that "
                             "the first match is identity.")
    parser.add_argument("--recalibrate_with_updated_alpha_resolution", default=False, action='store_true',
                        help="This will update the model in the main directory, updating "
                             "based on --alpha_resolution. However, note that the corresponding values for each "
                             "iteration (and the global statistics) do not get updated, since we do not currently "
                             "save the calibration data for every iteration.")
    parser.add_argument("--load_train_and_calibration_from_best_iteration_data_dir",
                        default=False, action='store_true', help="")
    parser.add_argument("--do_not_normalize_input_embeddings",
                        default=False, action='store_true',
                        help="Typically only use this if you have already standardized/normalized the embeddings. "
                             "Our default approach is to mean center based on the training set embeddings. This is "
                             "a global normalization that is applied in the forward of sdm_model.")
    parser.add_argument("--do_not_resave_shuffled_data",
                        default=False, action='store_true', help="")
    parser.add_argument("--exemplar_vector_dimension", default=constants.keyModelDimension, type=int, help="")

    parser.add_argument("--is_sdm_network_verification_layer",
                        default=False, action='store_true',
                        help="We have moved the full-parameter fine-tuning to another repo. "
                             "This flag, if used, will simply have the effect of saving the distance CDF "
                             "structures for the training set, which are not needed at test-time for predicting "
                             "over held-out sets, but may be of interest for analysis purposes.")

    parser.add_argument("--label_error_file", default="",
                        help="If provided, possible label annotation errors "
                             "(in the most conservative HR region but y != prediction) are saved, "
                             "sorted by the SDM(z')_prediction probability.")
    parser.add_argument("--predictions_in_high_reliability_region_file", default="",
                        help="If provided, instances with predictions in the most conservative High Reliability region "
                             "are saved, sorted by the SDM(z')_prediction probability.")
    parser.add_argument("--label_error_hr_lower_file", default="",
                        help="If provided, possible label annotation errors "
                             "(in the most conservative HR_lower region but y != prediction) "
                             "are saved, sorted by the SDM_lower(z')_prediction probability.")
    parser.add_argument("--predictions_in_high_reliability_region_lower_file", default="",
                        help="If provided, instances with predictions in the most conservative High Reliability "
                             "LOWER region are saved, sorted by the SDM_lower(z')_prediction probability.")
    parser.add_argument("--prediction_output_file", default="",
                        help="If provided, output predictions are saved to this file "
                             "in the order of the input file.")
    parser.add_argument("--update_support_set_with_eval_data", default=False, action='store_true',
                        help="update_support_set_with_eval_data")
    parser.add_argument("--skip_updates_already_in_support", default=False, action='store_true',
                        help="If --update_support_set_with_eval_data is provided, this will exclude any document "
                             "with the same id already in the support set or the calibration set. If you are sure "
                             "the documents are not already present, this can be excluded.")
    parser.add_argument("--main_device", default="cpu",
                        help="")
    parser.add_argument("--aux_device", default="cpu",
                        help="")
    parser.add_argument("--pretraining_initialization_epochs", default=0, type=int,
                        help="")
    parser.add_argument("--pretraining_learning_rate", default=0.00001, type=float, help="")
    parser.add_argument("--pretraining_initialization_tensors_file", default="",
                        help="")
    parser.add_argument("--ood_support_file", default="",
                        help="")
    parser.add_argument("--construct_results_latex_table_rows",
                        default=False, action='store_true',
                        help="")
    parser.add_argument("--additional_latex_meta_data", default="", help="dataset,model_name")
    parser.add_argument("--print_timing",
                        default=False, action='store_true',
                        help="Used for profiling training.")
    parser.add_argument("--alpha_resolution", default=constants.defaultAlphaResolution, type=float,
                        help="Resolution of the nested high-reliability regions: Alg. 1 in 'SDM Activations' is run at "
                             "alpha = 1 - k*alpha_resolution for k = 1, 2, ..., while alpha > 0.5, successively  "
                             "excluding the points in every higher region with a finite q'_min. "
                             "A resolution of 0.05, for example, results in the ladder "
                             "[0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55]; however, for a given model and "
                             "dataset, not all of the associated regions will necessarily be obtainable via the "
                             "data/model (i.e., have a finite q'_min). In effect, this is an adaptive, "
                             "data-driven approach for partitioning the calibration set into regions with "
                             "class- and prediction-conditional accuracy >= the given value.")

    # ensemble parameters:
    parser.add_argument("--eval_ensemble", default=False, action='store_true', help="")
    parser.add_argument("--eval_ensemble_start_iteration", default=-1, type=int, help="")
    parser.add_argument("--eval_ensemble_end_iteration", default=-1, type=int, help="Inclusive indexing")
    parser.add_argument("--eval_ensemble_label_error_file", default="",
                        help="If provided, possible label annotation errors "
                             "(in the most conservative HR region but y != prediction) are saved, "
                             "sorted by the SDM(z')_prediction probability.")
    parser.add_argument("--eval_ensemble_predictions_in_high_reliability_region_file", default="",
                        help="If provided, instances with predictions in the most conservative "
                             "High Reliability region are saved, "
                             "sorted by the SDM(z')_prediction probability.")
    parser.add_argument("--eval_ensemble_label_error_hr_lower_file", default="",
                        help="If provided, possible label annotation errors "
                             "(in the most conservative HR_lower region but y != prediction) "
                             "are saved, sorted by the SDM_lower(z')_prediction probability.")
    parser.add_argument("--eval_ensemble_predictions_in_high_reliability_region_lower_file", default="",
                        help="If provided, instances with predictions in the most conservative High Reliability "
                             "LOWER region are saved, "
                             "sorted by the SDM_lower(z')_prediction probability.")
    parser.add_argument("--eval_ensemble_prediction_output_file", default="",
                        help="If provided, output predictions are saved to this file "
                             "in the order of the input file.")

    # Options not yet implemented in this version:
    parser.add_argument("--continue_training",
                        default=False, action='store_true', help="")

    options = parser.parse_args()

    # Setting seed
    torch.manual_seed(options.seed_value)
    np.random.seed(options.seed_value)
    # random.seed(options.seed_value)
    rng = np.random.default_rng(seed=options.seed_value)

    assert not options.continue_training, "Not implemented"

    main_device = torch.device(options.main_device)
    print(f"The model will use {main_device} as the main device.")

    if not options.eval_only:
        utils_train_iterative_main.train_iterative_main(options, rng, main_device=main_device)

    if options.recalibrate_with_updated_alpha_resolution:
        print(f"Reloading best model to calibrate based on the provided alpha value.")
        utils_calibrate.calibrate_to_determine_high_reliability_region(options, model_dir=options.model_dir)

    utils_test_batch.test(options, main_device)

    if options.eval_ensemble:
        import os
        import utils_test_batch_ensemble
        import utils_model
        id2ensemble_stats = {}
        total_models_in_ensemble = \
            len(list(range(options.eval_ensemble_start_iteration, options.eval_ensemble_end_iteration + 1)))
        print(f"------------------------------------------------------------------------------------------")
        print(f"---------------Beginning Ensemble Evaluation of {total_models_in_ensemble} models---------------")
        for iteration in range(options.eval_ensemble_start_iteration, options.eval_ensemble_end_iteration + 1):
            iteration_model_dir = os.path.join(options.model_dir, str(iteration))
            print(f"------------------------------------------------------------------------------------------")
            print(f"---------------Processing Ensemble Shuffle Index {iteration_model_dir}---------------")
            id2ensemble_stats = \
                utils_test_batch.test(options, main_device,
                                      iteration_model_dir=iteration_model_dir, id2ensemble_stats=id2ensemble_stats)
        # First load the main model to get the most conservative HR region to consider in the ensemble:
        model = \
            utils_model.load_model_torch(options.model_dir, main_device, load_for_inference=True)
        hr_region_stats = model.get_most_conservative_high_reliability_region_stats()
        most_conservative_hr_alpha_to_consider_in_ensemble = hr_region_stats["most_conservative_hr_alpha"]
        print(f'\tUsing the most conservative alpha of the model in the main directory to determine the '
              f'most conservative high reliability region criteria that requires predictions and HR region matches '
              f'across ALL ensembled models: {most_conservative_hr_alpha_to_consider_in_ensemble}')
        maxQAvailableFromIndexer = model.maxQAvailableFromIndexer
        numberOfClasses = model.numberOfClasses
        del model
        utils_test_batch_ensemble.test(options, id2ensemble_stats=id2ensemble_stats,
                                       numberOfClasses=numberOfClasses,
                                       maxQAvailableFromIndexer=maxQAvailableFromIndexer,
                                       total_models_in_ensemble=total_models_in_ensemble,
                                       most_conservative_hr_alpha_to_consider_in_ensemble=
                                       most_conservative_hr_alpha_to_consider_in_ensemble)

    if options.update_support_set_with_eval_data:
        utils_update.batch_support_update(options, main_device)


if __name__ == "__main__":
    main()

