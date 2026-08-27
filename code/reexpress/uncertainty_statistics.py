# Copyright Reexpress AI, Inc. All rights reserved.

import constants

import numpy as np
from typing import Optional


class UncertaintyStatistics:
    """
    Global statistics across iterations of training and data splitting

   This collects the min valid similarity
    values across training iterations, which can be useful for analysis purposes. For example, it is a useful indicator
    to know if one of those values is inf, which suggests the alpha value may be too high to achieve with the given
    model and/or data. Note that the current version uses the most conservative region that is found, which may
    not be the same for each training iteration, so the alpha value of the most conservative region is also recorded.

    Note: This data structure does not currently get updated when the model is recalibrated without retraining (e.g.,
    when changing the resolution via --recalibrate_with_updated_alpha_resolution).
    """
    def __init__(self, globalUncertaintyModelUUID: str,
                 numberOfClasses: int,
                 min_rescaled_similarity_across_iterations: Optional[list[float]] = None,
                 max_hr_region_alpha_across_iterations: Optional[list[float]] = None):

        self.globalUncertaintyModelUUID = globalUncertaintyModelUUID
        self.numberOfClasses = numberOfClasses
        if min_rescaled_similarity_across_iterations is None:
            self.min_rescaled_similarity_across_iterations = []
        else:
            self.min_rescaled_similarity_across_iterations = min_rescaled_similarity_across_iterations
        if max_hr_region_alpha_across_iterations is None:
            self.max_hr_region_alpha_across_iterations = []
        else:
            self.max_hr_region_alpha_across_iterations = max_hr_region_alpha_across_iterations

    def update_high_reliability_region_stats(
            self, min_rescaled_similarity_to_determine_high_reliability_region: float,
            max_hr_region_alpha: float
    ):
        self.min_rescaled_similarity_across_iterations.append(
            min_rescaled_similarity_to_determine_high_reliability_region)
        self.max_hr_region_alpha_across_iterations.append(
            max_hr_region_alpha
        )

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

    def _get_min_valid_rescaled_similarity_mad(self) -> float:

        if len(self.min_rescaled_similarity_across_iterations) > 0:
            min_q_bin = UncertaintyStatistics.get_median_absolute_deviation_around_the_median(
                self.min_rescaled_similarity_across_iterations)
            if np.isfinite(min_q_bin):
                return min_q_bin
        return np.inf

    def validate_min_rescaled_similarities(self):
        count_non_finite = 0
        for rescaled_similarity in self.min_rescaled_similarity_across_iterations:
            if not np.isfinite(rescaled_similarity):
                count_non_finite += 1
        print(f"Summary stats over J iterations:")
        if count_non_finite > 0:
            print(f"\tWARNING: In {count_non_finite} training iterations out of "
                  f"{len(self.min_rescaled_similarity_across_iterations)}, a suitable threshold was not found at the "
                  f"given alpha value. The model and/or data may be too weak to reliably determine the High "
                  f"Reliability region.")
        else:
            print(f"\tThresholds were found at the given alpha value for all "
                  f"{len(self.min_rescaled_similarity_across_iterations)} training iterations.")
            print(f"Across iterations, the median absolute deviation around the median for the "
                  f"rescaled similarity (q') to determine the high reliability region is: "
                  f"{self._get_min_valid_rescaled_similarity_mad()}")

        is_consistent = True
        running_alpha = None
        i = 0
        for (max_alpha, q_prime_min) in zip(self.max_hr_region_alpha_across_iterations,
                                            self.min_rescaled_similarity_across_iterations):
            print(f"\tIteration {i}: most conservative alpha: {max_alpha}, corresponding q'_min: {q_prime_min}")
            if running_alpha is None:
                running_alpha = max_alpha
            else:
                if running_alpha != max_alpha:
                    is_consistent = False
            i += 1
        if not is_consistent:
            print(f"\tWARNING: The same most conservative alpha was not obtained over all iterations.")

    def export_properties_to_dict(self):

        json_dict = {constants.STORAGE_KEY_version: constants.ProgramIdentifiers_version,
                     constants.STORAGE_KEY_globalUncertaintyModelUUID: self.globalUncertaintyModelUUID,
                     constants.STORAGE_KEY_numberOfClasses: self.numberOfClasses,
                     constants.STORAGE_KEY_min_rescaled_similarity_across_iterations:
                         self.min_rescaled_similarity_across_iterations,
                     constants.STORAGE_KEY_max_hr_region_alpha_across_iterations:
                         self.max_hr_region_alpha_across_iterations
                     }
        return json_dict
