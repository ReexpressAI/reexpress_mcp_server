# Copyright Reexpress AI, Inc. All rights reserved.

import constants

import torch
import numpy as np
from typing import Optional

from collections import namedtuple


# def print_summary_stats(metric01_list, metric_label="", indent=0):
#     indent = "\t" * indent
#     print(f"{indent}{metric_label}: mean: {np.mean(metric01_list) if len(metric01_list) > 0 else 'NA'}; "
#           f"size: {len(metric01_list)}")


class UncertaintyStatistics:
    """
    Global statistics across iterations of training and data splitting
    """
    def __init__(self, globalUncertaintyModelUUID: str,
                 numberOfClasses: int,
                 min_valid_qbin_across_iterations: Optional[list[float]] = None,
                 predicted_class_to_bin_to_median_output_magnitude_of_iteration: Optional[dict[int, dict[int, float]]] = None,
                 cauchy_quantile: float = 0.95):

        self.globalUncertaintyModelUUID = globalUncertaintyModelUUID
        self.numberOfClasses = numberOfClasses
        if min_valid_qbin_across_iterations is None:
            self.min_valid_qbin_across_iterations = []
        else:
            self.min_valid_qbin_across_iterations = min_valid_qbin_across_iterations

        if predicted_class_to_bin_to_median_output_magnitude_of_iteration is None:
            self.predicted_class_to_bin_to_median_output_magnitude_of_iteration = {}
            for label in range(self.numberOfClasses):
                self.predicted_class_to_bin_to_median_output_magnitude_of_iteration[label] = {}
                for hard_bin in range(constants.default_max_hard_bin):
                    self.predicted_class_to_bin_to_median_output_magnitude_of_iteration[label][hard_bin] = []
        else:
            self.predicted_class_to_bin_to_median_output_magnitude_of_iteration = \
                predicted_class_to_bin_to_median_output_magnitude_of_iteration

        self.cauchy_quantile = cauchy_quantile

    def update_min_valid_qbin(self, min_valid_qbin: float):
        self.min_valid_qbin_across_iterations.append(min_valid_qbin)

    @staticmethod
    def depth_2_keys_present_in_dictionary(at_least_depth_2_dictionary, key1, key2) -> bool:
        return key1 in at_least_depth_2_dictionary and key2 in at_least_depth_2_dictionary[key1]

    @staticmethod
    def cauchy_inverse_cdf(median=0, scale=1.0, quantile=0.95) -> float:
        return median + scale*np.tan(np.pi*(quantile-0.5))

    def update_output_magnitudes_for_bin(self, one_iteration_predicted_class_to_bin_to_median_output_magnitude):
        # print(one_iteration_predicted_class_to_bin_to_median_output_magnitude)
        # output_magnitude_median is assumed to be the median across the iteration
        for label in self.predicted_class_to_bin_to_median_output_magnitude_of_iteration:
            for hard_qbin in self.predicted_class_to_bin_to_median_output_magnitude_of_iteration[label]:
                if UncertaintyStatistics.depth_2_keys_present_in_dictionary(
                        one_iteration_predicted_class_to_bin_to_median_output_magnitude, label, hard_qbin) and \
                        one_iteration_predicted_class_to_bin_to_median_output_magnitude[label][hard_qbin] is not None:
                    self.predicted_class_to_bin_to_median_output_magnitude_of_iteration[label][hard_qbin].append(
                        one_iteration_predicted_class_to_bin_to_median_output_magnitude[label][hard_qbin])

    @staticmethod
    def get_median_absolute_deviation_around_the_median(list_of_floats: list[float]) -> float:
        """
        Median absolute deviation (around the median)
        Parameters
        ----------
        list_of_floats

        Returns
        -------

        """
        median_val = np.median(list_of_floats)
        return np.median(np.abs(np.array(list_of_floats) - median_val))

    def _get_min_valid_qbin_mad(self) -> float:

        if len(self.min_valid_qbin_across_iterations) > 0:
            min_q_bin = UncertaintyStatistics.get_median_absolute_deviation_around_the_median(self.min_valid_qbin_across_iterations)
            if np.isfinite(min_q_bin):
                return min_q_bin
        return np.inf

    def get_min_valid_qbin_with_bounded_error(self, min_valid_qbin_from_best_iteration) -> float:

        min_valid_qbin_mad = self._get_min_valid_qbin_mad()
        min_valid_qbin_with_bounded_error = UncertaintyStatistics.cauchy_inverse_cdf(
            median=min_valid_qbin_from_best_iteration,
            scale=min_valid_qbin_mad,
            quantile=self.cauchy_quantile)  # higher is more conservative, so take the upper part of the distribution
        if np.isfinite(min_valid_qbin_with_bounded_error):
            return min_valid_qbin_with_bounded_error
        else:
            print("WARNING: Min valid qbin MAD is not finite. "
                  "The model may have been trained with a single iteration, "
                  "and/or may be too weak to achieve the desired accuracy rate. "
                  "Returning min_valid_qbin_from_best_iteration.")
            return min_valid_qbin_from_best_iteration


    def _get_output_magnitude_mad_for_bin(self, label: int, hard_qbin: int) -> Optional[float]:
        if UncertaintyStatistics.depth_2_keys_present_in_dictionary(
                        self.predicted_class_to_bin_to_median_output_magnitude_of_iteration, label, hard_qbin):
            if len(self.predicted_class_to_bin_to_median_output_magnitude_of_iteration[label][hard_qbin]) > 0:
                return UncertaintyStatistics.get_median_absolute_deviation_around_the_median(
                    self.predicted_class_to_bin_to_median_output_magnitude_of_iteration[label][hard_qbin])
        # Typically, we'll assign 0 for empty bins. This is acceptable, since small sample sizes in
        # these cases will be reflected in the DKW bound. However, the caller is responsible for deciding.
        return None

    def _get_summarized_output_magnitude_structure(self) -> dict[int, dict[int, float]]:  # label->hardbin->MAD
        predicted_class_to_bin_to_output_magnitude_mad = {}
        for label in range(self.numberOfClasses):
            predicted_class_to_bin_to_output_magnitude_mad[label] = {}
            for hard_bin in range(constants.default_max_hard_bin):
                output_magnitude_mad = self._get_output_magnitude_mad_for_bin(label=label, hard_qbin=hard_bin)
                if output_magnitude_mad is not None:
                    predicted_class_to_bin_to_output_magnitude_mad[label][hard_bin] = output_magnitude_mad
        return predicted_class_to_bin_to_output_magnitude_mad

    def get_summarized_output_magnitude_structure_with_bounded_error_lower_offset_by_bin(self) -> dict[int, dict[int, float]]:  # label->hardbin->Cauchy_inverse(quantile)
        predicted_class_to_bin_to_output_magnitude_with_bounded_error_lower_offset_by_bin = {}
        for label in range(self.numberOfClasses):
            predicted_class_to_bin_to_output_magnitude_with_bounded_error_lower_offset_by_bin[label] = {}
            for hard_bin in range(constants.default_max_hard_bin):
                output_magnitude_mad = self._get_output_magnitude_mad_for_bin(label=label, hard_qbin=hard_bin)
                if output_magnitude_mad is not None and np.isfinite(output_magnitude_mad):
                    # the location parameter is 0, because we add the offset to each output --- that is, the
                    # assumption is that each distribution is centered on the given point
                    output_magnitude_bounded_error = UncertaintyStatistics.cauchy_inverse_cdf(
                        median=0,
                        scale=output_magnitude_mad,
                        quantile=self.cauchy_quantile)  # symmetric at 0, so take the upper part of the distribution
                    if np.isfinite(output_magnitude_bounded_error):
                        predicted_class_to_bin_to_output_magnitude_with_bounded_error_lower_offset_by_bin[label][hard_bin] = \
                            output_magnitude_bounded_error
        return predicted_class_to_bin_to_output_magnitude_with_bounded_error_lower_offset_by_bin

    def export_properties_to_dict(self):

        json_dict = {constants.STORAGE_KEY_version: constants.ProgramIdentifiers_version,
                     constants.STORAGE_KEY_globalUncertaintyModelUUID: self.globalUncertaintyModelUUID,
                     constants.STORAGE_KEY_numberOfClasses: self.numberOfClasses,
                     constants.STORAGE_KEY_min_valid_qbin_across_iterations: self.min_valid_qbin_across_iterations,
                     constants.STORAGE_KEY_predicted_class_to_bin_to_median_output_magnitude_of_iteration:
                         self.predicted_class_to_bin_to_median_output_magnitude_of_iteration,
                     constants.STORAGE_KEY_cauchy_quantile: self.cauchy_quantile
                     }
        return json_dict

    def import_properties_from_dict(self, json_dict):
        # When loading from disk, this must be called after class init before calibrating new data points.
        # Note that in JSON, int dictionary keys become strings
        predicted_class_to_bin_to_median_output_magnitude_of_iteration_json_flat = \
            json_dict[constants.STORAGE_KEY_predicted_class_to_bin_to_median_output_magnitude_of_iteration]
        self.predicted_class_to_bin_to_median_output_magnitude_of_iteration = {}
        for label_str in predicted_class_to_bin_to_median_output_magnitude_of_iteration_json_flat.keys():
            label = int(label_str)
            self.predicted_class_to_bin_to_median_output_magnitude_of_iteration[label] = {}
            for hard_bin_str in predicted_class_to_bin_to_median_output_magnitude_of_iteration_json_flat[label_str].keys():
                hard_bin = int(hard_bin_str)
                self.predicted_class_to_bin_to_median_output_magnitude_of_iteration[label][hard_bin] = \
                    [float(x) for x in predicted_class_to_bin_to_median_output_magnitude_of_iteration_json_flat[label_str][hard_bin_str]]