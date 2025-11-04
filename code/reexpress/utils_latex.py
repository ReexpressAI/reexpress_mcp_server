# Copyright Reexpress AI, Inc. All rights reserved.

import numpy as np

MARGINAL_ACC_KEY = "m"
MARGINAL_ADMITTED_KEY = "m_prop"
CLASS_CONDITIONAL_ACC_KEY = "c"
CLASS_CONDITIONAL_ADMITTED_KEY = "c_prop"
PREDICTION_CONDITIONAL_ACC_KEY = "p"
PREDICTION_CONDITIONAL_ADMITTED_KEY = "p_prop"


def get_acc_prop_tuple(list_to_process, total=None):
    if total is not None and total > 0:
        prop = len(list_to_process)/total
    else:
        prop = 0.0
    acc = np.mean(list_to_process) if len(list_to_process) > 0 else 0
    return acc, prop


def get_decorated_string(float_proportion, display_string, decorate=False, decorate_alpha=0.95,
                         is_fully_rejected=False):
    if not decorate:
        return display_string
    if is_fully_rejected:
        return rf"\colorbox{{correctPredictionColor}}{{{display_string}}}"
    if float_proportion >= decorate_alpha:
        return rf"\colorbox{{correctPredictionColor}}{{{display_string}}}"
    else:
        return rf"\colorbox{{wrongPredictionColor}}{{{display_string}}}"


def get_float_as_display_significant_digits_string(float_proportion, decorate=False, decorate_alpha=0.95,
                                                   is_fully_rejected=False) -> str:
    if is_fully_rejected:
        return get_decorated_string(float_proportion, r"\allRejected", decorate=decorate,
                                    decorate_alpha=decorate_alpha, is_fully_rejected=is_fully_rejected)
    if float_proportion == 0.0:
        return get_decorated_string(float_proportion, "0.", decorate=decorate, decorate_alpha=decorate_alpha)
    elif float_proportion == 1.0:
        return get_decorated_string(float_proportion, "1.", decorate=decorate, decorate_alpha=decorate_alpha)
    if float_proportion < 0.005 and float_proportion != 0.0:
        return get_decorated_string(float_proportion, "<0.01", decorate=decorate, decorate_alpha=decorate_alpha)
    else:
        return get_decorated_string(float_proportion, f"{float_proportion:.2f}",
                                    decorate=decorate, decorate_alpha=decorate_alpha)


def get_latex_row(dataset_name, model_name, alpha, latex_rows_dict, estimator_label, numberOfClasses):
    running_latex_rows = []
    for class_label in range(numberOfClasses):
        conditional_acc = \
            get_float_as_display_significant_digits_string(
                latex_rows_dict[f"{CLASS_CONDITIONAL_ACC_KEY}{class_label}"],
                decorate=True, decorate_alpha=alpha,
                is_fully_rejected=latex_rows_dict[f"{CLASS_CONDITIONAL_ADMITTED_KEY}{class_label}"] == 0.0)
        running_latex_rows.append(conditional_acc)
        admitted_proportion = \
            get_float_as_display_significant_digits_string(
                latex_rows_dict[f"{CLASS_CONDITIONAL_ADMITTED_KEY}{class_label}"],
                decorate=False, decorate_alpha=alpha, is_fully_rejected=False)
        running_latex_rows.append(admitted_proportion)
    for class_label in range(numberOfClasses):
        conditional_acc = \
            get_float_as_display_significant_digits_string(
                latex_rows_dict[f"{PREDICTION_CONDITIONAL_ACC_KEY}{class_label}"],
                decorate=True, decorate_alpha=alpha,
                is_fully_rejected=latex_rows_dict[f"{PREDICTION_CONDITIONAL_ADMITTED_KEY}{class_label}"] == 0.0)
        running_latex_rows.append(conditional_acc)
        admitted_proportion = \
            get_float_as_display_significant_digits_string(
                latex_rows_dict[f"{PREDICTION_CONDITIONAL_ADMITTED_KEY}{class_label}"],
                decorate=False, decorate_alpha=alpha, is_fully_rejected=False)
        running_latex_rows.append(admitted_proportion)
    marginal_acc = get_float_as_display_significant_digits_string(
        latex_rows_dict[MARGINAL_ACC_KEY], decorate=True, decorate_alpha=alpha,
        is_fully_rejected=latex_rows_dict[MARGINAL_ADMITTED_KEY] == 0.0)
    running_latex_rows.append(marginal_acc)
    marginal_admitted_proportion = get_float_as_display_significant_digits_string(
        latex_rows_dict[MARGINAL_ADMITTED_KEY], decorate=False,
        decorate_alpha=alpha, is_fully_rejected=False)
    running_latex_rows.append(marginal_admitted_proportion)
    return " & ".join([dataset_name, model_name, estimator_label]) + " & " + " & ".join(running_latex_rows) + r"\\"


def init_latex_rows_dict(numberOfClasses):
    latex_rows_dict = {}
    latex_rows_dict[MARGINAL_ACC_KEY] = 0.0  # marginal accuracy
    latex_rows_dict[MARGINAL_ADMITTED_KEY] = 0.0  # marginal |Admitted| / |N|
    for class_label in range(numberOfClasses):
        latex_rows_dict[f"{CLASS_CONDITIONAL_ACC_KEY}{class_label}"] = 0.0  # class-conditional accuracy
        latex_rows_dict[f"{CLASS_CONDITIONAL_ADMITTED_KEY}{class_label}"] = 0.0  # class-conditional |Admitted| / |N|
        latex_rows_dict[f"{PREDICTION_CONDITIONAL_ACC_KEY}{class_label}"] = 0.0  # prediction-conditional accuracy
        latex_rows_dict[f"{PREDICTION_CONDITIONAL_ADMITTED_KEY}{class_label}"] = 0.0  # prediction-conditional |Admitted| / |N|
    return latex_rows_dict


def print_latex_row(options, model, alpha_prime, latex_rows_dict_no_reject,
                    latex_rows_dict_softmax_f, latex_rows_dict_softmax_df,
                    latex_rows_dict_sdm, latex_rows_dict_sdm_hr):
    if options.construct_results_latex_table_rows:
        latex_meta_data_list = options.additional_latex_meta_data.strip().split(",")
        if len(latex_meta_data_list) == 2:
            dataset_name = rf'$\{latex_meta_data_list[0]}$'
            model_name = rf'$\{latex_meta_data_list[1]}$'
        else:
            dataset_name = rf'DATASET-NAME-HERE'
            model_name = rf'MODEL-NAME-HERE'
        print(f"Latex-formatted results table rows (alpha={alpha_prime})")
        print(
            get_latex_row(dataset_name, model_name, alpha_prime, latex_rows_dict_no_reject,
                          r'$\estimatorNoReject$', model.numberOfClasses)
        )
        print(
            get_latex_row(dataset_name, model_name, alpha_prime, latex_rows_dict_softmax_f,
                          r'$\estimatorSoftmax$', model.numberOfClasses)
        )
        print(
            get_latex_row(dataset_name, model_name, alpha_prime, latex_rows_dict_softmax_df,
                          r'$\estimatorSoftmaxOverDistanceMagnitude$',
                          model.numberOfClasses)
        )
        print(
            get_latex_row(dataset_name, model_name, alpha_prime, latex_rows_dict_sdm,
                          r'$\sdm$', model.numberOfClasses)
        )
        print(
            get_latex_row(dataset_name, model_name, alpha_prime, latex_rows_dict_sdm_hr,
                          r'$\sdmHR$', model.numberOfClasses)
        )
