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
    model and/or data.
    """
    def __init__(self, globalUncertaintyModelUUID: str,
                 numberOfClasses: int,
                 min_rescaled_similarity_across_iterations: Optional[list[float]] = None):

        self.globalUncertaintyModelUUID = globalUncertaintyModelUUID
        self.numberOfClasses = numberOfClasses
        if min_rescaled_similarity_across_iterations is None:
            self.min_rescaled_similarity_across_iterations = []
        else:
            self.min_rescaled_similarity_across_iterations = min_rescaled_similarity_across_iterations

    def update_min_rescaled_similarity_to_determine_high_reliability_region(
            self, min_rescaled_similarity_to_determine_high_reliability_region: float):
        self.min_rescaled_similarity_across_iterations.append(
            min_rescaled_similarity_to_determine_high_reliability_region)

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
        if count_non_finite > 0:
            print(f"WARNING: In {count_non_finite} training iterations out of "
                  f"{len(self.min_rescaled_similarity_across_iterations)}, a suitable threshold was not found at the "
                  f"given alpha value. The model and/or data may be too weak to reliably determine the High "
                  f"Reliability region.")
        else:
            print(f"Thresholds were found at the given alpha value for all "
                  f"{len(self.min_rescaled_similarity_across_iterations)} training iterations.")
            print(f"Across iterations, the median absolute deviation around the median for the "
                  f"rescaled similarity (q') to determine the high reliability region is: "
                  f"{self._get_min_valid_rescaled_similarity_mad()}")

    def export_properties_to_dict(self):

        json_dict = {constants.STORAGE_KEY_version: constants.ProgramIdentifiers_version,
                     constants.STORAGE_KEY_globalUncertaintyModelUUID: self.globalUncertaintyModelUUID,
                     constants.STORAGE_KEY_numberOfClasses: self.numberOfClasses,
                     constants.STORAGE_KEY_min_rescaled_similarity_across_iterations:
                         self.min_rescaled_similarity_across_iterations,
                     }
        return json_dict
