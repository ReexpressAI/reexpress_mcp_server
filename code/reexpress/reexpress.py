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
    parser = argparse.ArgumentParser(description="-----[Train and eval sdm estimators]-----")
    # Note that not all options are currently implemented and/or used in this research codebase. See the
    # Tutorials for replicating the paper's experiments, rather than the argument help messages and in-line comments
    # in the code, which may not reflect the currently released research codebase version.
    parser.add_argument("--input_training_set_file", default="",
                        help=".jsonl format")
    parser.add_argument("--input_calibration_set_file", default="",
                        help=".jsonl format")
    parser.add_argument("--input_eval_set_file", default="",
                        help=".jsonl format")

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

    parser.add_argument("--alpha", default=constants.defaultCdfAlpha, type=float, help="alpha in (0,1), "
                                                                                       "typically 0.9 or 0.95")
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
    parser.add_argument("--recalibrate_with_updated_alpha", default=False, action='store_true',
                        help="This will update the model in the main directory, updating "
                             "q'_min based on --alpha. However, note that the corresponding values for each "
                             "iteration (and the global statistics) do not get updated, since we do not currently "
                             "save the calibration data for every iteration.")
    parser.add_argument("--load_train_and_calibration_from_best_iteration_data_dir",
                        default=False, action='store_true', help="")
    parser.add_argument("--do_not_normalize_input_embeddings",
                        default=False, action='store_true',
                        help="Typically only use this if you have already standardized/normalized the embeddings. "
                             "Our default approach is to mean center based on the training set embeddings. This is "
                             "a global normalization that is applied in the forward of sdm_model.")
    parser.add_argument("--continue_training",
                        default=False, action='store_true', help="")
    parser.add_argument("--do_not_resave_shuffled_data",
                        default=False, action='store_true', help="")
    parser.add_argument("--exemplar_vector_dimension", default=constants.keyModelDimension, type=int, help="")

    parser.add_argument("--is_sdm_network_verification_layer",
                        default=False, action='store_true', help="")

    parser.add_argument("--label_error_file", default="",
                        help="If provided, possible label annotation errors are saved, sorted by the LOWER predictive "
                             "probability, where the subset is those that are valid index-conditional predictions.")
    parser.add_argument("--predictions_in_high_reliability_region_file", default="",
                        help="If provided, instances with predictions in the High Reliability region are saved, "
                             "sorted by the output from sdm().")
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
    parser.add_argument("--is_baseline_adaptor",
                        default=False, action='store_true',
                        help="Use this option to train and test a baseline adaptor using cross-entropy and softmax.")
    parser.add_argument("--construct_results_latex_table_rows",
                        default=False, action='store_true',
                        help="")
    parser.add_argument("--additional_latex_meta_data", default="", help="dataset,model_name")
    parser.add_argument("--print_timing",
                        default=False, action='store_true',
                        help="Used for profiling training.")

    options = parser.parse_args()

    # Setting seed
    torch.manual_seed(options.seed_value)
    np.random.seed(options.seed_value)
    # random.seed(options.seed_value)
    rng = np.random.default_rng(seed=options.seed_value)

    if options.is_training_support:
        assert options.batch_eval

    assert not options.continue_training, "Not implemented"

    main_device = torch.device(options.main_device)
    print(f"The model will use {main_device} as the main device.")

    if not options.eval_only:
        if options.is_baseline_adaptor:
            import baseline_utils_train_iterative_main
            baseline_utils_train_iterative_main.train_iterative_main(options, rng, main_device=main_device)
        else:
            utils_train_iterative_main.train_iterative_main(options, rng, main_device=main_device)

    if options.recalibrate_with_updated_alpha:
        print(f"Reloading best model to calibrate based on the provided alpha value.")
        utils_calibrate.calibrate_to_determine_high_reliability_region(options, model_dir=options.model_dir)

    if options.is_baseline_adaptor:
        import baseline_utils_test
        baseline_utils_test.test(options, main_device)
    else:
        utils_test_batch.test(options, main_device)
    if options.update_support_set_with_eval_data:
        assert not options.is_baseline_adaptor
        utils_update.batch_support_update(options, main_device)

    if options.construct_results_latex_table_rows:
        if options.is_baseline_adaptor:
            print(f"To print the LaTeX tables for the baseline adaptor, use the baseline evaluation script.")


if __name__ == "__main__":
    main()

