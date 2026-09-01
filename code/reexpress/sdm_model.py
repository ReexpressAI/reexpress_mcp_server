# Copyright Reexpress AI, Inc. All rights reserved.

import constants

import torch
import torch.nn as nn
import numpy as np
import faiss
from collections import namedtuple


# Steps for constructing Similarity-Distance-Magnitude activations:
#
# 1. Train model against training set, using soft_sdm_max.
#     CDF(d_nearest) for training is over training, and q is calculated against training, excluding the identity match.
#         (The first epoch does not rescale and uses the equivalent of a standard softmax and CrossEntropy loss.)
#     CDF(d_nearest) for calibration is over calibration, and q is calculated against training
#     (subsequently, CDF(d_nearest) is over calibration for new, unseen test instances,
#     and q is calculated against training)
#     Note that the class-wise CDFs for d_nearest are calculated excluding q=0 instances, which are considered OOD.
#     They are considered OOD because with q=0, the distance to the nearest match is undefined,
#     since the nearest match is not a similar instance, by definition.
# 2. Calculate the thresholds (over calibration) to detect the high-reliability regions. In this version, we
#    consider multiple regions (corresponding to decreasing values of alpha, each time re-running Alg. 1). I.e., this
#    is a descending-alpha loop: Alg. 1 is run at alpha = 1 - alpha_resolution, and if q'_min is finite, the
#    (alpha, q'_min, class-wise output thresholds) triple is recorded and the calibration points that are
#    members of that region (q' >= q'_min with a singleton prediction set via the thresholds, exactly as at
#    test-time) are excluded; Alg. 1 is then altogether re-run at the next lower alpha (decremented by
#    alpha_resolution) as if those points are gone, continuing while alpha > 0.5 (as required by Alg. 1).
#    Rungs of the ladder without a finite q'_min are skipped without exclusion. (To put it another way, if alpha=0.9,
#    but a suitable q'_min is not found, alpha is decremented and Alg. 1 is rerun again with that lower value of
#    alpha.) Alg. 1 determines the region
#    boundaries: the q'_min and threshold values of lower alpha regions may be higher or lower than those of
#    the higher HR regions. The recorded regions in self.hr_regions are the sole source of the region
#    parameters (the alpha values, the class-wise output thresholds, and the per-region q'_min values).
#    This is described in the research note "Nested Similarity-Distance-Magnitude Estimators".
# 3. Collect the sample size summary statistics. The effective sample
#    size is assumed to be increasing in q', class-wise over the calibration set. In high-risk settings, it is
#    recommended to also explicitly take into account the error from the effective sample size.
#
# At test-time (as calculated for a single instance in `single_pass_forward`):
#     1. Calculate the SDM High Reliability region (centroid and lower), as well as the nested-region
#         assignments ("hr_region_alpha" and "hr_region_alpha_lower"): the most conservative (i.e., closest
#         to 1) alpha among the recorded regions whose gates the instance passes, with 0 indicating that no
#         region's gates pass. For each region, the DKW error term used for the lower assignment is pinned to
#         that region's alpha.
#     2. The non-rejected points from (1) are those suitable for final decision-making. If needed to triage the
#         remedial actions of the rejected points, the output from sdm() can be used directly with the
#         understanding that the estimates are of unspecified reliability. The
#         points with floor(q') == 0 and/or d=0 are strictly OOD, but the typical convention is to also treat any
#         point not assigned to a class- and prediction-conditional region with alpha > 0.5 to also be effectively OOD.


ModelCalibrationTrainingStage = namedtuple("ModelCalibrationTrainingStage",
                                           ["init", "base_model", "rescaler", "complete"])
modelCalibrationTrainingStages = ModelCalibrationTrainingStage(0, 1, 2, 3)

class SimilarityDistanceMagnitudeCalibrator(nn.Module):
    def __init__(self,
                 version: str,
                 uncertaintyModelUUID: str,
                 numberOfClasses: int,
                 embedding_size: int,
                 train_labels,
                 train_predicted_labels,
                 train_uuids,
                 exemplar_vector_dimension: int = constants.keyModelDimension,
                 trueClass_To_dCDF = None,
                 trueClass_To_qCumulativeSampleSizeArray = None,
                 maxQAvailableFromIndexer: int = constants.maxQAvailableFromIndexer,
                 calibration_training_stage: int = 0,
                 training_embedding_summary_stats = None,
                 is_sdm_network_verification_layer=False,
                 # Resolution of the nested high-reliability regions: Alg. 1 is run at
                 # alpha = 1 - k*alpha_resolution for k = 1, 2, ..., while alpha > 0.5.
                 alpha_resolution: float = constants.defaultAlphaResolution,
                 hr_regions=None,
                 # the following can be None at test-time to save memory, if desired:
                 calibration_labels = None,
                 calibration_predicted_labels = None,
                 calibration_uuids = None,
                 calibration_sdm_outputs = None,
                 calibration_rescaled_similarity_values = None,
                 calibration_is_ood_indicators = None,
                 # These are None on re-load to avoid overwriting learned weights.
                 train_trueClass_To_dCDF = None
                 ):

        super(SimilarityDistanceMagnitudeCalibrator, self).__init__()

        self.version = version
        self.uncertaintyModelUUID = uncertaintyModelUUID
        self.numberOfClasses = numberOfClasses

        # If shuffled, all must be shuffled together.
        self.train_labels = train_labels
        self.train_predicted_labels = train_predicted_labels  # must be set before calculating q, d0
        self.train_uuids = train_uuids
        assert training_embedding_summary_stats is not None
        self.training_embedding_summary_stats = training_embedding_summary_stats

        # These can be None at inference to save memory, but we save these values as part of the model during training
        # since they are needed to calculate the parameters for rescaling and the output class-conditional thresholds.
        # This is done for convenience, since dataset shuffling can alter the indexes relative to
        # the original orders. See load_uncertainty_statistics_from_disk()'s `load_for_inference` argument.
        self.calibration_labels = calibration_labels
        self.calibration_predicted_labels = calibration_predicted_labels
        self.calibration_uuids = calibration_uuids  # JSON
        self.calibration_sdm_outputs = calibration_sdm_outputs
        self.calibration_rescaled_similarity_values = calibration_rescaled_similarity_values
        if calibration_is_ood_indicators is None:
            self.calibration_is_ood_indicators = []
        else:
            # The semantics of "OOD" (as used elsewhere in eval) has changed from earlier versions, but
            # specifically, this list records for reference during training:
            # 0 if q != 0 and 1 if q == 0
            self.calibration_is_ood_indicators = calibration_is_ood_indicators

        if trueClass_To_dCDF is None:
            self.trueClass_To_dCDF = {}
        else:
            self.trueClass_To_dCDF = trueClass_To_dCDF

        if train_trueClass_To_dCDF is None:  # see self.set_train_trueClass_To_dCDF()
            self.train_trueClass_To_dCDF = {}
        else:
            self.train_trueClass_To_dCDF = train_trueClass_To_dCDF

        if trueClass_To_qCumulativeSampleSizeArray is None:
            self.trueClass_To_qCumulativeSampleSizeArray = {}
        else:
            self.trueClass_To_qCumulativeSampleSizeArray = trueClass_To_qCumulativeSampleSizeArray

        self.maxQAvailableFromIndexer = maxQAvailableFromIndexer

        self.q_rescale_offset = constants.q_rescale_offset  # This typically should not change.
        self.ood_limit = constants.ood_limit  # This typically should not change.

        self.alpha_resolution = alpha_resolution
        # Nested high-reliability regions ("alpha ladder"), sorted descending by alpha. Each element is a dict:
        # {"alpha": float, "min_rescaled_similarity": float,
        #  "output_thresholds": torch.Tensor of shape [numberOfClasses]}.
        # self.hr_regions is the sole source of the region parameters (the alpha values, the class-wise output
        # thresholds, and the per-region q'_min values). See calculateOutputThresholdsAdaptive().
        if hr_regions is None:
            self.hr_regions = []
        else:
            self.hr_regions = hr_regions

        self.exemplar_vector_dimension = exemplar_vector_dimension
        self.embedding_size = embedding_size

        self.is_sdm_network_verification_layer = is_sdm_network_verification_layer

        # Input:
        # [composition attributes (optional)] :: [Cumulative average LLM embeddings (optional)] :: [LLM embedding]
        # Typically:
        # [Cumulative average LLM embeddings (up to and including t)] :: [LLM embedding at current token t]
        exemplar_network_input_size = self.embedding_size
        self.conv = nn.Conv1d(1, self.exemplar_vector_dimension, exemplar_network_input_size,
                              stride=exemplar_network_input_size)
        self.fc = nn.Linear(self.exemplar_vector_dimension, self.numberOfClasses)  # for router / verificationLayer

        # Support index is saved separately, as it may be quite large. See setters and getters below.
        self.support_index = None

        self.calibration_training_stage = calibration_training_stage

    @property
    def device(self):
        return self.fc.weight.device

    @property
    def on_gpu(self):
        return self.device.type == 'cuda'

    def get_most_conservative_high_reliability_region_stats(self):
        if len(self.hr_regions) > 0:
            return {"most_conservative_hr_alpha": self.hr_regions[0]["alpha"],
                    "most_conservative_hr_output_thresholds": self.hr_regions[0]["output_thresholds"],
                    "most_conservative_hr_min_rescaled_similarity": self.hr_regions[0]["min_rescaled_similarity"]}
        else:
            return {"most_conservative_hr_alpha": 0.0,
                    "most_conservative_hr_output_thresholds": torch.zeros(self.numberOfClasses),
                    "most_conservative_hr_min_rescaled_similarity": torch.inf}

    def increment_model_calibration_training_stage(self, set_value=None):
        self.calibration_training_stage = set_value

    def set_train_predicted_labels(self, train_predicted_labels):
        self.train_predicted_labels = train_predicted_labels

    def set_calibration_predicted_labels(self, calibration_predicted_labels):
        self.calibration_predicted_labels = calibration_predicted_labels

    def set_train_trueClass_To_dCDF(self, train_trueClass_To_dCDF):
        # convenience for training SDM networks, since the distance from the generated output to the
        # force-decoded output is needed when training, but this is not needed for standard classification, and isn't
        # saved to conserve space in that case
        if self.is_sdm_network_verification_layer:
            self.train_trueClass_To_dCDF = train_trueClass_To_dCDF
        else:
            self.train_trueClass_To_dCDF = {}

    def construct_support_index(self,
                                support_exemplar_vectors_numpy=None, calibration_exemplar_vectors_numpy=None,
                                k=None,
                                ood_support_exemplar_vectors_numpy=None,
                                ood_support_labels=None,
                                ood_support_predicted_labels=None,
                                ood_support_document_ids=None
                                ):
        # Note that FAISS uses numpy arrays.
        # Note that any existing support index will be overwritten
        assert support_exemplar_vectors_numpy is not None
        assert calibration_exemplar_vectors_numpy is not None
        dimensions = self.exemplar_vector_dimension
        assert support_exemplar_vectors_numpy.shape[1] == self.exemplar_vector_dimension
        assert calibration_exemplar_vectors_numpy.shape[1] == self.exemplar_vector_dimension
        if k is None:
            k = self.maxQAvailableFromIndexer
        support_index = faiss.IndexFlatL2(dimensions)  # build the index
        support_index.add(support_exemplar_vectors_numpy)  # add exemplar vectors to the index
        if ood_support_exemplar_vectors_numpy is not None and ood_support_labels is not None and \
            ood_support_predicted_labels is not None and ood_support_document_ids is not None and \
            len(ood_support_document_ids) > 0:
            assert ood_support_exemplar_vectors_numpy.shape[1] == self.exemplar_vector_dimension
            self.add_to_support_batch(labels=ood_support_labels,
                                      predicted_labels=ood_support_predicted_labels,
                                      document_ids=ood_support_document_ids,
                                      exemplar_vectors=ood_support_exemplar_vectors_numpy)
            print(f">Added {len(ood_support_document_ids)} OOD/additional instances to the training support.<")
        if k > support_index.ntotal:
            k = support_index.ntotal  # indexes will be -1 if exceeds, so hard constraint here
        if self.on_gpu:
            # start move to gpu
            gpu_id = self.device.index
            res = faiss.StandardGpuResources()
            support_index = faiss.index_cpu_to_gpu(res, gpu_id, support_index)
            print(f"Model is on a CUDA device, so the new FAISS index has been moved to cuda:{gpu_id}.")
            # end move
        top_k_distances, top_k_distances_idx = support_index.search(calibration_exemplar_vectors_numpy, k)
        self.support_index = support_index
        return support_index, top_k_distances, top_k_distances_idx

    def set_support_index(self, support_index):
        self.support_index = support_index

    def add_to_support(self, label: int, predicted_label: int, document_id: str, exemplar_vector):
        # We assume the caller has checked that d0 != 0
        assert exemplar_vector is not None
        # FAISS expects numpy
        if isinstance(exemplar_vector, torch.Tensor):
            self.support_index.add(exemplar_vector.detach().cpu().numpy())
        else:
            self.support_index.add(exemplar_vector)
        label = torch.tensor([label], device=self.train_labels.device)
        self.train_labels = torch.cat([self.train_labels, label])
        predicted_label = torch.tensor([predicted_label], device=self.train_predicted_labels.device)
        self.train_predicted_labels = torch.cat([self.train_predicted_labels, predicted_label])
        self.train_uuids.append(document_id)

    def add_to_support_batch(self, labels, predicted_labels, document_ids: list[str], exemplar_vectors):
        assert isinstance(labels, torch.Tensor)
        assert isinstance(predicted_labels, torch.Tensor)
        assert labels.dim() == 1, f"Expected 1D labels, got shape {labels.shape}"
        assert predicted_labels.dim() == 1, f"Expected 1D predicted_labels, got shape {predicted_labels.shape}"
        assert labels.shape[0] == predicted_labels.shape[0]
        assert predicted_labels.shape[0] == exemplar_vectors.shape[0]

        # FAISS expects numpy
        if isinstance(exemplar_vectors, torch.Tensor):
            self.support_index.add(exemplar_vectors.detach().cpu().numpy())
        else:
            self.support_index.add(exemplar_vectors)

        # Concatenate tensors on same device
        self.train_labels = torch.cat([self.train_labels, labels.to(self.train_labels.device)])
        self.train_predicted_labels = torch.cat([self.train_predicted_labels,
                                                 predicted_labels.to(self.train_predicted_labels.device)])
        self.train_uuids.extend(document_ids)

    def get_top_support_distances(self, batch_eval_exemplar_vectors_numpy, k=None):
        assert self.support_index is not None
        assert len(batch_eval_exemplar_vectors_numpy.shape) == 2
        assert batch_eval_exemplar_vectors_numpy.shape[1] == self.exemplar_vector_dimension
        if k is None:
            k = self.maxQAvailableFromIndexer
        if k > self.support_index.ntotal:
            k = self.support_index.ntotal  # indexes will be -1 if exceeds, so hard constraint here
        top_k_distances, top_k_distances_idx = self.support_index.search(batch_eval_exemplar_vectors_numpy, k)
        return top_k_distances, top_k_distances_idx

    def soft_sdm_max_log_to_probability(self, batch_input, q):
        """
        Convert from log space, with q as the base, to probability space, taking into account the rescale offset.
        This can be used during training when the sdm() output from one network needs to be re-composed with another
        model that takes input in the probability space.

        Parameters
        ----------
        batch_input
            Output from self.soft_sdm_max(batch_input, q, log=True, change_of_base=True).
        q
            Same as in soft_sdm_max()

        Returns
        -------
            (self.q_rescale_offset + q) ** batch_input
        """

        assert len(batch_input.shape) == 2
        assert batch_input.shape[0] == q.shape[0]
        assert q.shape[1] == 1
        q_factor = self.q_rescale_offset + q
        return q_factor ** batch_input

    def soft_sdm_max(self, batch_input, q, distance_quantile_per_class=None, log=False,
                     change_of_base=True):
        """
        Instead of standard softmax e^val/sum(e^val), this normalizes via
        q^(val_y*(1-CDF(d_nearest)_y))/sum(q^(val_y*(1-CDF(d_nearest)_y)),
        increasing the relative amplification/sharpness of the distribution for higher Similarity (q) values
        and larger Distance quantiles (d). Each column of distance_quantile_per_class for a given row must be the same
        value; in the canonical SDM activation, each value is the min distance quantile across classes for a given
        instance.

        In this version, as a best practice, we avoid reimplementing numerical stability code and
        instead use standard softmax in Pytorch
        by using the relation (2+q)^(z'*d) == e^(z'*d*log_e(2+q)), given that 2+q > 0. The change of base is
        achieved by multiplying by the factor 1/log_e(2+q) prior to calculating the loss.

        Parameters
        ----------
        batch_input
            torch.tensor
            shape == [batch size, self.numberOfClasses]; if, e.g., batch_size == 1, [1, self.numberOfClasses]
        q
            torch.tensor
            shape == [batch size, 1], with each value in [0, constants.maxQAvailableFromIndexer]. This function then
            adds self.q_rescale_offset to q. For the standard softmax (assuming self.q_rescale_offset==2, as
            is typical), use q=torch.tensor([[torch.e-2],...]).
        distance_quantile_per_class
            torch.tensor, or None
            If not None, shape == [batch size, self.numberOfClasses], with each quantile in [0,1]. Each column of
            a given row must contain the same value.
        log
            If True, take the log (useful for training)
        change_of_base
            If log == True, use q as the base of the logarithm. Should always be True in practice; only included
            for reference/debugging.

        Returns
        -------
        [batch size, self.numberOfClasses]
        """

        assert change_of_base
        assert len(batch_input.shape) == 2
        if not self.is_sdm_network_verification_layer:
            assert batch_input.shape[1] == self.numberOfClasses
        assert batch_input.shape[0] == q.shape[0]
        assert q.shape[1] == 1
        if distance_quantile_per_class is not None:
            assert batch_input.shape == distance_quantile_per_class.shape
        q_factor = self.q_rescale_offset + q
        log_q_factor = torch.log(q_factor)
        if distance_quantile_per_class is not None:
            scaled_logits = batch_input * distance_quantile_per_class * log_q_factor
        else:
            scaled_logits = batch_input * log_q_factor
        if log:
            log_probs = torch.nn.functional.log_softmax(scaled_logits, dim=-1)  # log_softmax for numerical stability
            if change_of_base:
                return log_probs / log_q_factor
            else:
                return log_probs
        else:
            return torch.softmax(scaled_logits, dim=-1)

    def get_quantile(self, float_list, quantileProportion: float):
        quantileIndex = min(int(quantileProportion * len(float_list)), len(float_list) - 1)
        return torch.sort(torch.tensor(float_list, dtype=torch.float32)).values[quantileIndex].item()

    def getCdfThresholdForClass(self, normalized_output_for_true_class, alpha):
        if len(normalized_output_for_true_class) > 0:
            # Note: round() guards against float accumulation in the quantile proportion (e.g.,
            # 1 - 0.9 = 0.09999999999999998), which would otherwise truncate the quantile index down by one
            # order statistic in get_quantile() for some alpha values:
            return max(self.get_quantile(normalized_output_for_true_class, round(1 - alpha, 10)), 0.0)
        return 0.0  # conservative (no information about class, so always included)

    def get_ladder_alphas(self):
        # The descending alpha values at which Alg. 1 in 'SDM Activations' is run to construct the nested
        # high-reliability regions: alpha = 1 - k*self.alpha_resolution, for k = 1, 2, ..., stopping before
        # alpha <= 0.5, since Alg. 1 requires alpha in (0.5, 1.0]. For example, with the default
        # alpha_resolution of 0.05: [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55].
        assert 0.0 < self.alpha_resolution < 0.5, \
            f"ERROR: --alpha_resolution must be in (0, 0.5), so that at least one alpha value " \
            f"(1 - k*alpha_resolution, for k = 1, 2, ...) is in (0.5, 1.0], as required by Alg. 1 in " \
            f"'SDM Activations'"
        ladder_alphas = []
        k = 1
        while True:
            # Round to guard against float accumulation (e.g., 0.55000000000000004):
            candidate_alpha = round(1.0 - k * self.alpha_resolution, 10)
            if candidate_alpha <= 0.5:
                break
            ladder_alphas.append(candidate_alpha)
            k += 1
        return ladder_alphas

    def calculateOutputThresholdsForAlpha(self, trueClass_To_rescaled_OutputCDF_non_ood, all_bins, alpha):
        # The single-alpha search of Alg. 1 in 'SDM Activations', over the provided (possibly residual) points.
        # Note: trueClass_To_rescaled_OutputCDF_non_ood must have values from a
        # categorical distribution for this to be valid.
        # Note: This function destructively reduces the provided trueClass_To_rescaled_OutputCDF_non_ood
        # structure (as an optimization, since the candidate bins are ascending), so the caller should provide
        # a (shallow) per-rung copy if the structure is to be reused, as in calculateOutputThresholdsAdaptive().
        # No class properties are set here; this returns the tuple (q'_min, thresholds, diagnostics), with
        # (torch.inf, None, diagnostics) if no finite q'_min obtains the class-wise thresholds at the given
        # alpha. The diagnostics dictionary records the best achievable value (over the candidate bins) of the
        # minimum class-wise threshold, which determines acceptance (torch.all(thresholds >= alpha) is
        # equivalent to torch.min(thresholds) >= alpha), so that skipped rungs of the nested construction are
        # inspectable (see calculateOutputThresholdsAdaptive()).
        all_bins = list(set(all_bins))
        all_bins.sort()
        diagnostics = {"best_min_class_threshold": -torch.inf, "best_candidate_bin": None,
                       "best_output_thresholds": None}
        for candidate_bin in all_bins:
            trueClass_To_CDF = {}
            for trueLabel in range(self.numberOfClasses):
                trueClass_To_CDF[trueLabel] = []
                if trueLabel in trueClass_To_rescaled_OutputCDF_non_ood:
                    filtered = []
                    filtered_rescaled_outputs = []
                    for tuple_of_output_and_rescaled_similarity in trueClass_To_rescaled_OutputCDF_non_ood[trueLabel]:
                        output = tuple_of_output_and_rescaled_similarity[0]
                        rescaled_similarity = tuple_of_output_and_rescaled_similarity[1]
                        if rescaled_similarity >= candidate_bin:
                            filtered_rescaled_outputs.append(output)
                            filtered.append(tuple_of_output_and_rescaled_similarity)
                    trueClass_To_CDF[trueLabel] = filtered_rescaled_outputs
                    trueClass_To_rescaled_OutputCDF_non_ood[trueLabel] = filtered  # reduce
            thresholds = torch.zeros(self.numberOfClasses)
            for trueLabel in range(self.numberOfClasses):
                if trueLabel in trueClass_To_CDF:
                    rescaled_outputs = trueClass_To_CDF[trueLabel]
                    threshold = self.getCdfThresholdForClass(normalized_output_for_true_class=rescaled_outputs,
                                                             alpha=alpha)
                    thresholds[trueLabel] = threshold
            min_class_threshold = torch.min(thresholds).item()
            if min_class_threshold > diagnostics["best_min_class_threshold"]:
                diagnostics["best_min_class_threshold"] = min_class_threshold
                diagnostics["best_candidate_bin"] = candidate_bin
                diagnostics["best_output_thresholds"] = thresholds
            if torch.all(thresholds >= alpha):
                return candidate_bin, thresholds, diagnostics
        return torch.inf, None, diagnostics

    def calculateOutputThresholdsAdaptive(self, trueClass_To_rescaled_OutputCDF_non_ood, all_bins,
                                          calibration_sdm_outputs=None,
                                          calibration_rescaled_similarity_values=None):
        # Alg. 1 in 'SDM Activations', run as a descending-alpha loop to construct nested high-reliability
        # regions: Alg. 1 is run at each alpha of get_ladder_alphas() (from the most to the least conservative)
        # over the residual points. If q'_min is finite at a given alpha, the
        # (alpha, q'_min, class-wise output thresholds) triple is recorded in self.hr_regions, and the
        # calibration points that are members of the recorded region (i.e., q' >= q'_min with a singleton
        # prediction set determined by the thresholds, exactly as at test-time; see
        # get_high_reliability_region_indicator_vectorized()) are excluded, with Alg. 1 then re-run over the
        # remaining points at the next lower alpha. If q'_min is not finite at a given alpha, no points are
        # excluded and the search continues at the next lower alpha. At test-time, a point is assigned to the
        # region of the most conservative (i.e., closest to 1) alpha whose gates it passes (see
        # get_hr_region_alpha_vectorized()).
        # This is described in the research note "Nested Similarity-Distance-Magnitude Estimators".
        assert calibration_sdm_outputs is not None and calibration_rescaled_similarity_values is not None, \
            f"ERROR: The full calibration SDM outputs and rescaled Similarity values are required to " \
            f"determine region membership (for exclusion) when constructing the nested regions."
        assert calibration_sdm_outputs.dim() == 2 and \
               calibration_sdm_outputs.shape[1] == self.numberOfClasses, \
            f"Expected calibration_sdm_outputs of shape [N, numberOfClasses], got " \
            f"{calibration_sdm_outputs.shape}"
        # Normalize to a 1-D tensor (e.g., accepting a column vector of shape [N, 1]). This is required, as
        # a non-1-D tensor would otherwise silently broadcast to an [N, N] mask in the vectorized membership
        # evaluation below (via [N] & [N, 1]), which would incorrectly exclude every calibration point:
        calibration_rescaled_similarity_values = calibration_rescaled_similarity_values.reshape(-1)
        assert calibration_rescaled_similarity_values.shape[0] == calibration_sdm_outputs.shape[0], \
            f"Expected one rescaled Similarity value per calibration row: " \
            f"{calibration_rescaled_similarity_values.shape[0]} vs {calibration_sdm_outputs.shape[0]}"
        # Reset the existing class properties, if present:
        self.hr_regions = []
        if all_bins is None or len(all_bins) == 0:
            print(constants.ERROR_MESSAGES_NO_THRESHOLD_FOUND)
            return
        # Predictions over the calibration set, for determining region membership:
        # Edge case: Note that elsewhere, we take care to use the argmax of the underlying (unrescaled) logits as the
        # predicted class, since flips are possible when going to parity. However, it's ok here, since such
        # cases will not survive thresholding, since alpha is required to be > 0.5.
        calibration_predictions = torch.argmax(calibration_sdm_outputs, dim=1)

        residual_trueClass_To_rescaled_OutputCDF_non_ood = {}
        for trueLabel in trueClass_To_rescaled_OutputCDF_non_ood:
            residual_trueClass_To_rescaled_OutputCDF_non_ood[trueLabel] = \
                list(trueClass_To_rescaled_OutputCDF_non_ood[trueLabel])

        for ladder_alpha in self.get_ladder_alphas():
            # The candidate bins are the rescaled Similarity values present among the residual points:
            residual_bins = []
            for trueLabel in residual_trueClass_To_rescaled_OutputCDF_non_ood:
                for tuple_of_output_and_rescaled_similarity in \
                        residual_trueClass_To_rescaled_OutputCDF_non_ood[trueLabel]:
                    residual_bins.append(tuple_of_output_and_rescaled_similarity[1])
            if len(residual_bins) == 0:
                # all points have been excluded by the higher regions
                print(f"The residual is empty (all remaining calibration points were members of the higher "
                      f"regions); ending the nested search before alpha={ladder_alpha}.")
                break
            # A per-rung shallow copy, since calculateOutputThresholdsForAlpha() reduces its argument:
            rung_trueClass_To_rescaled_OutputCDF_non_ood = {}
            for trueLabel in residual_trueClass_To_rescaled_OutputCDF_non_ood:
                rung_trueClass_To_rescaled_OutputCDF_non_ood[trueLabel] = \
                    list(residual_trueClass_To_rescaled_OutputCDF_non_ood[trueLabel])
            residual_class_sizes = \
                [len(residual_trueClass_To_rescaled_OutputCDF_non_ood[trueLabel])
                 for trueLabel in sorted(residual_trueClass_To_rescaled_OutputCDF_non_ood)]
            min_rescaled_similarity, thresholds, diagnostics = \
                self.calculateOutputThresholdsForAlpha(rung_trueClass_To_rescaled_OutputCDF_non_ood,
                                                       residual_bins, ladder_alpha)
            if min_rescaled_similarity != torch.inf:
                self.hr_regions.append({"alpha": ladder_alpha,
                                        "min_rescaled_similarity": min_rescaled_similarity,
                                        "output_thresholds": thresholds})
                # Exclude the members of the recorded region before continuing to the next lower alpha. The
                # membership criteria exactly match those applied at test-time:
                _, is_region_member, _ = self.get_high_reliability_region_indicator_vectorized(
                    rescaled_similarities=calibration_rescaled_similarity_values,
                    batch_sdm_outputs=calibration_sdm_outputs,
                    predictions=calibration_predictions,
                    min_rescaled_similarity=min_rescaled_similarity,
                    hr_output_thresholds=thresholds,
                    hr_class_conditional_accuracy=ladder_alpha)
                is_region_member_list = is_region_member.tolist()
                for trueLabel in residual_trueClass_To_rescaled_OutputCDF_non_ood:
                    residual_trueClass_To_rescaled_OutputCDF_non_ood[trueLabel] = \
                        [tuple_of_output_and_rescaled_similarity for tuple_of_output_and_rescaled_similarity in
                         residual_trueClass_To_rescaled_OutputCDF_non_ood[trueLabel]
                         if not is_region_member_list[tuple_of_output_and_rescaled_similarity[2]]]
                print(
                    f"Min rescaled Similarity to achieve class-conditional accuracy of {ladder_alpha}: "
                    f"{min_rescaled_similarity} "
                    f"(searched residual with per-class sizes {residual_class_sizes})")
                print(f"Thresholds: {thresholds}")
            else:
                print(f"No finite min rescaled Similarity found at alpha={ladder_alpha}; continuing to the "
                      f"next lower alpha without excluding points. "
                      f"(Searched residual with per-class sizes {residual_class_sizes}; the best achievable "
                      f"minimum class-wise threshold was {diagnostics['best_min_class_threshold']} at "
                      f"q'={diagnostics['best_candidate_bin']}, with class-wise thresholds "
                      f"{diagnostics['best_output_thresholds']} there, against the required {ladder_alpha}.)")

        if len(self.hr_regions) > 0:
            print(f"Class-conditional accuracy estimate of the most conservative region: "
                  f"{self.hr_regions[0]['alpha']}")
            print(f"Total nested high-reliability regions: {len(self.hr_regions)}, with alpha values: "
                  f"{[hr_region['alpha'] for hr_region in self.hr_regions]}")
        else:
            print(constants.ERROR_MESSAGES_NO_THRESHOLD_FOUND)

    def set_high_reliability_region_thresholds(self, calibration_sdm_outputs: torch.Tensor,
                                               calibration_rescaled_similarity_values: torch.Tensor,
                                               true_labels: torch.Tensor):
        # Note: The alpha values of the nested regions are determined by self.alpha_resolution (validated in
        # get_ladder_alphas(), called via calculateOutputThresholdsAdaptive() below); each is in (0.5, 1.0], as
        # required by Alg. 1 in 'SDM Activations'.
        assert 0.0 < self.alpha_resolution < 0.5, \
            f"ERROR: --alpha_resolution must be in (0, 0.5)"

        trueClass_To_sdm_outputs_non_ood = {}

        for label in range(self.numberOfClasses):
            trueClass_To_sdm_outputs_non_ood[label] = []
            self.trueClass_To_qCumulativeSampleSizeArray[label] = []
        all_non_ood_rescaled_similarities = []
        self.eval()
        with torch.no_grad():
            self.calibration_is_ood_indicators = []  # reset OOD indicators, if present
            for dataset_row_index, (calibration_sdm_output, calibration_rescaled_similarity_value, true_label) \
                    in enumerate(zip(
                    calibration_sdm_outputs,
                    calibration_rescaled_similarity_values, true_labels)):
                true_label = true_label.item()
                is_ood = False
                floor_rescaled_similarity = int(calibration_rescaled_similarity_value.item())
                if floor_rescaled_similarity <= self.ood_limit:
                    is_ood = True

                if not is_ood:
                    # indexed by *true label*
                    trueClass_To_sdm_outputs_non_ood[true_label].append(
                        (
                            calibration_sdm_output[true_label].item(),
                            calibration_rescaled_similarity_value.item(),
                            # the dataset row index, for region-membership exclusion when constructing the
                            # nested regions (see calculateOutputThresholdsAdaptive()):
                            dataset_row_index
                        )
                    )
                    all_non_ood_rescaled_similarities.append(calibration_rescaled_similarity_value.item())
                self.calibration_is_ood_indicators.append(int(is_ood))
                self.trueClass_To_qCumulativeSampleSizeArray[true_label].append(
                    calibration_rescaled_similarity_value.item())

        assert len(self.calibration_is_ood_indicators) == self.calibration_labels.shape[0]
        total_ood = torch.sum(torch.tensor(self.calibration_is_ood_indicators))
        print(f"Total q==0 instances in the calibration set: {total_ood} "
              f"out of {len(self.calibration_is_ood_indicators)}: "
              f"{100*(total_ood.item()/len(self.calibration_is_ood_indicators))}%")

        for label in range(self.numberOfClasses):
            trueClass_To_sdm_outputs_non_ood[label].sort(key=lambda x: x[1])  # sort by rescaled similarity
            self.trueClass_To_qCumulativeSampleSizeArray[label].sort()
        self.calculateOutputThresholdsAdaptive(trueClass_To_sdm_outputs_non_ood, all_non_ood_rescaled_similarities,
                                               calibration_sdm_outputs=calibration_sdm_outputs,
                                               calibration_rescaled_similarity_values=
                                               calibration_rescaled_similarity_values)
        self.increment_model_calibration_training_stage(set_value=modelCalibrationTrainingStages.complete)

    def get_cumulative_effective_sample_sizes_and_errors_vectorized(self, rescaled_similarities: torch.Tensor,
                                                                    alpha_value: float):
        """Construct a band around the per-class empirical CDFs using the DKW inequality, given the
            modeling assumption that the effective sample is increasing in the rescaled Similarity, class-wise over
            the calibration set.

        Parameters
        ----------
        rescaled_similarities : torch.Tensor
            Shape [batch_size] containing rescaled similarity values
        alpha_value : float
            The alpha at which the DKW error term is calculated (required; in (0.5, 1.0)). For the nested
            regions, the error term is pinned to each region's alpha (see
            get_hr_region_alpha_lower_vectorized()), so more conservative (higher alpha) regions require wider
            intervals, ceteris paribus. When only the effective sample sizes are needed (they do not depend on
            alpha_value), any valid alpha may be passed (see get_batch_eval_output_dictionary()).

        Returns
        -------
        cumulative_effective_sample_sizes : torch.Tensor
            Shape [batch_size, numberOfClasses]
        effective_cdf_sample_size_errors : torch.Tensor
            Shape [batch_size, numberOfClasses]
        """
        assert isinstance(rescaled_similarities, torch.Tensor)
        assert rescaled_similarities.dim() == 1, f"Expected 1D tensor, got shape {rescaled_similarities.shape}"

        batch_size = rescaled_similarities.shape[0]

        # Move to correct device
        rescaled_similarities = rescaled_similarities.to(self.device)

        # Initialize output tensors
        cumulative_effective_sample_sizes = \
            torch.zeros(batch_size, self.numberOfClasses, device=self.device)  # default is 0
        effective_cdf_sample_size_errors = \
            torch.ones(batch_size, self.numberOfClasses, device=self.device)  # default is 1

        alpha = 1 - alpha_value  # Note how alpha is defined
        assert alpha < 0.5, "ERROR: The alpha value is likely misspecified. " \
                            "Check that it should not be 1-(the provided value). If such a low alpha value is " \
                            "desired, comment this assert. Note that the nested high-reliability regions " \
                            "(Alg. 1 in 'SDM Activations') require each alpha to be > 0.5 " \
                            "(see get_ladder_alphas())."

        # Process all classes
        for label in range(self.numberOfClasses):
            if label not in self.trueClass_To_qCumulativeSampleSizeArray or \
                    len(self.trueClass_To_qCumulativeSampleSizeArray[label]) == 0:
                # If no data for this class, keep defaults (0 for sizes, 1 for errors)
                continue

            # Convert CDF array to tensor on the same device
            cdf_tensor = torch.tensor(self.trueClass_To_qCumulativeSampleSizeArray[label],
                                      dtype=torch.float32, device=self.device)
            # cdf_len = len(cdf_tensor)
            # # Use PyTorch's searchsorted for GPU acceleration
            # indices = torch.searchsorted(cdf_tensor, rescaled_similarities, side='left')
            #
            # # The indices are the sample sizes, so we just need to apply the max constraint
            # sample_sizes = torch.minimum(
            #     indices,
            #     torch.tensor(max(0, cdf_len - 1), device=self.device, dtype=torch.long)
            # )
            # Earlier versions used the above commented-out convention (inherited from the distance eCDF convention
            # explicitly noted in A.4.2), but the following is arguably a more typical computational
            # interpretation/implementation of Eq. 11. Here, side='right' counts the calibration points of
            # the class with a rescaled similarity <= the eval instance's value (the standard,
            # right-continuous eCDF convention: The cumulative effective sample at the instance's level
            # includes the points at exactly that level). Since
            # min(q, (2+q)^{p}) clamps confident points to integer q, exact-tie
            # groups that belong to the cumulative effective
            # sample can result. (However, in practice, with the data from release v2.5.0 of the MCP Server, for
            # example, the difference in final assignments for our eval/analysis datasets is negligible.):
            indices = torch.searchsorted(cdf_tensor, rescaled_similarities, side='right')
            # The indices are the sample sizes:
            sample_sizes = indices

            cumulative_effective_sample_sizes[:, label] = sample_sizes

            # Calculate DKW errors for non-zero sample sizes
            if alpha > 0:
                # Create mask for positive sample sizes
                positive_mask = sample_sizes > 0

                # Calculate errors only for positive sample sizes to avoid division by zero
                if positive_mask.any():
                    effective_cdf_sample_size_errors[positive_mask, label] = torch.sqrt(
                        torch.log(torch.tensor(2.0 / alpha, device=self.device)) /
                        (2.0 * sample_sizes[positive_mask].float())
                    )

        return cumulative_effective_sample_sizes, effective_cdf_sample_size_errors

    def get_distance_quantiles_vectorized(self, dataset_d0_values, train_trueClass_To_dCDF=None):

        take_min_across_percentiles = True

        assert isinstance(dataset_d0_values, torch.Tensor)
        # Guard against numerical issues.
        d0_values_tensor = torch.clamp(dataset_d0_values, min=0.0).to(self.device)

        dataset_distance_quantile_per_class = torch.zeros(d0_values_tensor.shape[0], self.numberOfClasses,
                                                          device=self.device)

        # Use the appropriate CDF dictionary
        cdf_dict = train_trueClass_To_dCDF if train_trueClass_To_dCDF is not None else self.trueClass_To_dCDF

        # Process all classes at once
        for label in range(self.numberOfClasses):
            if label not in cdf_dict or len(cdf_dict[label]) == 0:
                # If no CDF data for this class, use 0.0
                dataset_distance_quantile_per_class[:, label] = 0.0
            else:
                # Convert CDF to tensor on the same device
                cdf_tensor = torch.tensor(cdf_dict[label], dtype=torch.float32, device=self.device)
                cdf_len = len(cdf_tensor)

                # Use PyTorch's searchsorted for GPU acceleration
                indices = torch.searchsorted(cdf_tensor, d0_values_tensor, side='left')

                # Calculate quantiles (reverse=True for distances)
                quantiles = 1.0 - indices.float() / cdf_len

                # Store results
                dataset_distance_quantile_per_class[:, label] = quantiles

        if take_min_across_percentiles:
            # Take minimum across all classes for each instance
            min_quantiles = torch.min(dataset_distance_quantile_per_class, dim=1)[0]
            dataset_distance_quantile_per_class[:, :] = min_quantiles.unsqueeze(1)

        return dataset_distance_quantile_per_class

    def get_summary_stats_for_eval_vectorized(self, eval_set_size, top_k_distances, top_k_distances_idx,
                                              eval_logits, is_training_support=False):
        # This is similar to set_summary_stats_for_support_vectorized(), but here the values are collected for the
        # held-out evaluation set, so we do not need to set class properties.
        assert self.train_predicted_labels is not None
        if is_training_support:
            # at least two support indexes must be present for the training split,
            # since the first match will be identity
            assert top_k_distances_idx.shape[1] > 1
        else:
            # Equivalently, at least one support index must be present for other dataset splits
            assert top_k_distances_idx.shape[1] > 0
        assert eval_set_size == top_k_distances.shape[0]
        assert eval_set_size == top_k_distances_idx.shape[0]
        assert eval_set_size == eval_logits.shape[0]
        # Ensure train labels are on GPU
        self.train_labels = self.train_labels.to(self.device)
        self.train_predicted_labels = self.train_predicted_labels.to(self.device)
        # Move all inputs to GPU
        if isinstance(top_k_distances_idx, np.ndarray):
            top_k_distances_idx = torch.from_numpy(top_k_distances_idx).to(self.device)
        else:
            top_k_distances_idx = top_k_distances_idx.to(self.device)

        if isinstance(top_k_distances, np.ndarray):
            top_k_distances_torch = torch.from_numpy(top_k_distances).float().to(self.device)
        else:
            top_k_distances_torch = top_k_distances.to(self.device).float()

        if not isinstance(eval_logits, torch.Tensor):
            eval_logits = torch.from_numpy(eval_logits).to(self.device)
        elif eval_logits.device != self.device:
            eval_logits = eval_logits.to(self.device)

        # Get predicted labels on GPU
        eval_predicted_labels = torch.argmax(eval_logits, dim=1)

        # Efficient gathering
        k = top_k_distances_idx.shape[1]
        batch_size = eval_set_size  # full eval in one pass
        flat_idx = top_k_distances_idx.reshape(-1)

        matched_true_labels = self.train_labels[flat_idx].reshape(batch_size, k)
        matched_predicted_labels = self.train_predicted_labels[flat_idx].reshape(batch_size, k)

        # Comparison mask
        eval_pred_expanded = eval_predicted_labels.unsqueeze(1)
        match_mask = (matched_true_labels == matched_predicted_labels) & \
                     (matched_predicted_labels == eval_pred_expanded)

        # Calculate q values
        if is_training_support:
            # For training, we assume the first match is identity, so we skip when calculating q and d0.
            match_mask_subset = match_mask[:, 1:].float()  # Boolean mask converted to 1's and 0's (as floats).
            # A cumulative product of 1's and 0's, so the first non-match and all subsequent positions will be 0.
            consecutive_mask = torch.cumprod(match_mask_subset, dim=1)
            # Given the cumulative product above, this sum only considers the matching indexes into the support set.
            q_values = consecutive_mask.sum(dim=1, keepdim=True)
        else:
            consecutive_mask = torch.cumprod(match_mask.float(), dim=1)
            q_values = consecutive_mask.sum(dim=1, keepdim=True)

        # Extract d0 values
        d0_values = top_k_distances_torch[:, 1 if is_training_support else 0]
        # This handles the numerical edge case where the exact match is a very small negative value.
        # In principle such cases would be correctly handled by the empirical CDFs,
        # but they could cause unexpected surprises downstream
        # in future changes to the codebase (as well as when viewing analysis output), so we check here.
        d0_values = torch.clamp(d0_values, min=0.0)

        return q_values, d0_values

    def set_summary_stats_for_support_vectorized(self, eval_set_size, top_k_distances, top_k_distances_idx,
                                                 eval_logits, eval_labels, is_training_support=False):
        """GPU version"""
        assert self.train_predicted_labels is not None
        if is_training_support:
            # at least two support indexes must be present for the training split,
            # since the first match will be identity
            assert top_k_distances_idx.shape[1] > 1
        else:
            # Equivalently, at least one support index must be present for other dataset splits
            assert top_k_distances_idx.shape[1] > 0
        assert eval_set_size == top_k_distances.shape[0]
        assert eval_set_size == top_k_distances_idx.shape[0]
        assert eval_set_size == eval_logits.shape[0]
        assert eval_set_size == eval_labels.shape[0]
        # Ensure train labels are on GPU
        self.train_labels = self.train_labels.to(self.device)
        self.train_predicted_labels = self.train_predicted_labels.to(self.device)
        # Move all inputs to GPU
        if isinstance(top_k_distances_idx, np.ndarray):
            top_k_distances_idx = torch.from_numpy(top_k_distances_idx).to(self.device)
        else:
            top_k_distances_idx = top_k_distances_idx.to(self.device)

        if isinstance(top_k_distances, np.ndarray):
            top_k_distances_torch = torch.from_numpy(top_k_distances).float().to(self.device)
        else:
            top_k_distances_torch = top_k_distances.to(self.device).float()

        if not isinstance(eval_logits, torch.Tensor):
            eval_logits = torch.from_numpy(eval_logits).to(self.device)
        elif eval_logits.device != self.device:
            eval_logits = eval_logits.to(self.device)

        if not isinstance(eval_labels, torch.Tensor):
            eval_labels_torch = torch.from_numpy(eval_labels).to(self.device)
        elif eval_labels.device != self.device:
            eval_labels_torch = eval_labels.to(self.device)
        else:
            eval_labels_torch = eval_labels

        # Get predicted labels on GPU
        eval_predicted_labels = torch.argmax(eval_logits, dim=1)

        # Efficient gathering
        k = top_k_distances_idx.shape[1]
        batch_size = eval_set_size  # full eval in one pass
        flat_idx = top_k_distances_idx.reshape(-1)

        matched_true_labels = self.train_labels[flat_idx].reshape(batch_size, k)
        matched_predicted_labels = self.train_predicted_labels[flat_idx].reshape(batch_size, k)

        # Comparison mask
        eval_pred_expanded = eval_predicted_labels.unsqueeze(1)
        match_mask = (matched_true_labels == matched_predicted_labels) & \
                     (matched_predicted_labels == eval_pred_expanded)

        # Calculate q values
        if is_training_support:
            # For training, we assume the first match is identity, so we skip when calculating q and d0.
            match_mask_subset = match_mask[:, 1:].float()  # Boolean mask converted to 1's and 0's (as floats).
            # A cumulative product of 1's and 0's, so the first non-match and all subsequent positions will be 0.
            consecutive_mask = torch.cumprod(match_mask_subset, dim=1)
            # Given the cumulative product above, this sum only considers the matching indexes into the support set.
            q_values = consecutive_mask.sum(dim=1, keepdim=True)
        else:
            consecutive_mask = torch.cumprod(match_mask.float(), dim=1)
            q_values = consecutive_mask.sum(dim=1, keepdim=True)

        # Extract d0 values
        d0_values = top_k_distances_torch[:, 1 if is_training_support else 0]
        # This handles the numerical edge case where the exact match is a very small negative value.
        # In principle such cases would be correctly handled by the empirical CDFs,
        # but they could cause unexpected surprises downstream
        # in future changes to the codebase (as well as when viewing analysis output), so we check here.
        d0_values = torch.clamp(d0_values, min=0.0)

        # Valid mask to exclude unlabeled (-1) and OOD-labeled (-99) documents.
        # Note: This is currently redundant since we iterate through range(numberOfClasses),
        # but kept for semantic clarity and to document the special label values.
        valid_mask = (eval_labels_torch >= 0) & (eval_labels_torch < self.numberOfClasses)

        trueClass_To_dCDF = {}  # Used in downstream calculations to determine the distance quantile for each point.
        trueClass_To_dataset_total_q_ood = {}  # Informational for analysis
        trueClass_To_total_labels = {}  # Informational for analysis

        # Note: Documents with special labels (unlabeled=-1, OOD-labeled=-99) are automatically
        # excluded by the equality check in the loop below since we only iterate through valid class indices.
        for label in range(self.numberOfClasses):
            class_mask = (eval_labels_torch == label) & valid_mask
            trueClass_To_total_labels[label] = int(class_mask.sum().item())

            class_ood_mask = class_mask & (q_values.squeeze() <= self.ood_limit)
            trueClass_To_dataset_total_q_ood[label] = int(class_ood_mask.sum().item())

            # Non-OOD d0 values for this class. OOD (i.e., q=0) points are excluded from the class-wise distance CDFs.
            class_non_ood_mask = class_mask & (q_values.squeeze() > self.ood_limit)
            if class_non_ood_mask.any():
                class_d0 = d0_values[class_non_ood_mask]
                # Sort on GPU then transfer. This sort is critical for subsequent
                # binary search to determine the distance quantiles.
                sorted_d0, _ = torch.sort(class_d0)
                trueClass_To_dCDF[label] = sorted_d0.cpu().numpy().tolist()
            else:
                trueClass_To_dCDF[label] = []

        if not is_training_support:
            self.trueClass_To_dCDF = trueClass_To_dCDF
            return q_values, trueClass_To_dataset_total_q_ood, trueClass_To_total_labels, d0_values, None
        else:
            self.set_train_trueClass_To_dCDF(train_trueClass_To_dCDF=trueClass_To_dCDF)
            return q_values, trueClass_To_dataset_total_q_ood, trueClass_To_total_labels, d0_values, trueClass_To_dCDF

    def get_rescaled_similarity_vectorized(self, q, sdm_output_for_predicted_class):
        """
        Compute rescaled similarity.

        Parameters
        ----------
        q : torch.Tensor
            Similarity value(s)
        sdm_output_for_predicted_class : torch.Tensor
            SDM output value(s) for the predicted class

        Returns
        -------
        torch.Tensor
            Rescaled similarity value(s). shape: torch.Size([batch_size])
        """
        assert isinstance(q, torch.Tensor)
        # Vectorized version
        rescaled_values = (self.q_rescale_offset + q) ** sdm_output_for_predicted_class
        rescaled_similarity = torch.minimum(q, rescaled_values)
        return rescaled_similarity

    def get_rescaled_similarity_for_eval_batch(self, cached_f_outputs,
                                               dataset_q_values, sdm_outputs,
                                               return_tensors_on_cpu=True,
                                               keepdim=False, return_sdm_outputs_for_predicted=False):
        # rescaled_similarities has shape: torch.Size([batch_size]) if keepdim=False.
        # rescaled_similarities.unsqueeze(1) thus matches the shape of
        # dataset_q_values (i.e., torch.Size([batch_size, 1])).
        # Set keep_dims=True to return the values as torch.Size([batch_size, 1])
        predictions = torch.argmax(cached_f_outputs, dim=1)

        # Extract SDM outputs for predicted classes using advanced indexing
        # Create indices for gathering the correct SDM output values
        batch_indices = torch.arange(len(predictions))
        sdm_outputs_for_predicted = sdm_outputs[batch_indices, predictions].to(self.device)

        # Ensure q_values is the right shape - squeeze if needed
        if dataset_q_values.dim() > 1:
            q_values_squeezed = dataset_q_values.squeeze(-1)
        else:
            q_values_squeezed = dataset_q_values

        # Vectorized computation of rescaled similarities
        rescaled_similarities = self.get_rescaled_similarity_vectorized(
            q=q_values_squeezed,
            sdm_output_for_predicted_class=sdm_outputs_for_predicted
        )
        if keepdim:
            rescaled_similarities = rescaled_similarities.unsqueeze(1)
            predictions = predictions.unsqueeze(1)
            if return_sdm_outputs_for_predicted:
                sdm_outputs_for_predicted = sdm_outputs_for_predicted.unsqueeze(1)
        if return_tensors_on_cpu:
            rescaled_similarities = rescaled_similarities.detach().cpu()
            predictions = predictions.detach().cpu()
            if return_sdm_outputs_for_predicted:
                sdm_outputs_for_predicted = sdm_outputs_for_predicted.detach().cpu()
        if return_sdm_outputs_for_predicted:
            return rescaled_similarities, predictions, sdm_outputs_for_predicted
        else:
            return rescaled_similarities, predictions

    def get_high_reliability_region_indicator_vectorized(self, rescaled_similarities, batch_sdm_outputs, predictions,
                                                         min_rescaled_similarity=None, hr_output_thresholds=None,
                                                         hr_class_conditional_accuracy=None):
        """
        Vectorized version to determine high reliability regions for a batch

        Parameters
        ----------
        rescaled_similarities : torch.Tensor
            Shape [batch_size] containing rescaled similarity values
        batch_sdm_outputs : torch.Tensor
            Shape [batch_size, numberOfClasses] containing SDM outputs
        predictions : torch.Tensor
            Shape [batch_size] containing predicted class indices
        min_rescaled_similarity : float, or None
            If None (along with the other two region parameters below), the most conservative recorded region
            (self.hr_regions[0]) is used, with no recorded regions yielding an all-False
            is_high_reliability_region (floor_rescaled_similarities and is_ood are returned regardless).
            Provide a region's value to evaluate membership in that nested region (see
            get_hr_region_alpha_vectorized()). The three region parameters must be provided together (or all
            omitted).
        hr_output_thresholds : torch.Tensor, or None
            If None, as above. Provide a region's thresholds to evaluate membership in that nested region.
        hr_class_conditional_accuracy : float, or None
            If None, as above. Provide a region's alpha to evaluate membership in that nested region. (A value
            of 0.0 indicates no region is available, in which case all instances are out of the region.)

        Returns
        -------
        floor_rescaled_similarities : torch.Tensor
            Shape [batch_size] with integer floor values
        is_high_reliability_region : torch.Tensor
            Shape [batch_size] boolean tensor
        is_ood : torch.Tensor
            Shape [batch_size] boolean tensor
        """
        assert rescaled_similarities.dim() == 1, f"Expected 1D tensor, got shape {rescaled_similarities.shape}"
        assert batch_sdm_outputs.dim() == 2 and batch_sdm_outputs.shape[0] == rescaled_similarities.shape[0], \
            f"Expected batch_sdm_outputs of shape [batch_size, numberOfClasses], got {batch_sdm_outputs.shape}"
        assert predictions.dim() == 1 and predictions.shape[0] == rescaled_similarities.shape[0], \
            f"Expected 1D predictions of length {rescaled_similarities.shape[0]}, got shape {predictions.shape}"
        batch_size = rescaled_similarities.shape[0]
        device = rescaled_similarities.device
        region_parameters = [min_rescaled_similarity, hr_output_thresholds, hr_class_conditional_accuracy]
        assert all(parameter is None for parameter in region_parameters) or \
               all(parameter is not None for parameter in region_parameters), \
            f"The region parameters (min_rescaled_similarity, hr_output_thresholds, " \
            f"hr_class_conditional_accuracy) must be provided together, or all omitted (in which case the " \
            f"most conservative recorded region is used)."
        if min_rescaled_similarity is None:
            if len(self.hr_regions) > 0:
                min_rescaled_similarity = self.hr_regions[0]["min_rescaled_similarity"]
                hr_output_thresholds = self.hr_regions[0]["output_thresholds"]
                hr_class_conditional_accuracy = self.hr_regions[0]["alpha"]
            else:
                # No recorded regions: is_high_reliability_region is all-False below
                # (hr_output_thresholds is then never read, given the hr_class_conditional_accuracy > 0.0 and
                # valid_bins gates):
                min_rescaled_similarity = torch.inf
                hr_class_conditional_accuracy = 0.0

        # Compute floor of rescaled similarities
        floor_rescaled_similarities = torch.floor(rescaled_similarities).long()

        # Check OOD condition
        is_ood = floor_rescaled_similarities <= self.ood_limit

        # Check valid bin condition (not OOD AND rescaled >= min threshold)
        valid_bins = (~is_ood) & (
                    rescaled_similarities >= min_rescaled_similarity)

        # Initialize high reliability region indicators as False
        is_high_reliability_region = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Only check singleton condition for valid bins
        if valid_bins.any() and hr_class_conditional_accuracy > 0.0:
            # Ensure hr_output_thresholds is on the same device
            thresholds = hr_output_thresholds.to(device) if torch.is_tensor(hr_output_thresholds) else \
                torch.tensor(hr_output_thresholds, dtype=torch.float32, device=device)

            # Create mask where SDM outputs >= thresholds
            # Shape: [batch_size, numberOfClasses]
            above_threshold = batch_sdm_outputs >= thresholds.unsqueeze(0)

            # Count how many classes are in the prediction set for each sample
            prediction_set_sizes = above_threshold.sum(dim=1)

            # Check if predicted class is in the prediction set
            # Use gather to get the threshold status for each sample's predicted class
            pred_in_set = above_threshold.gather(1, predictions.unsqueeze(1)).squeeze(1)

            # A singleton containing the predicted class means:
            # 1. Only one class passes threshold (size == 1)
            # 2. The predicted class passes threshold
            # 3. The bin is valid
            is_singleton = (prediction_set_sizes == 1) & pred_in_set & valid_bins

            # The high reliability region is where we have singleton sets in the region bounded by the min
            # rescaled Similarity:
            is_high_reliability_region = is_singleton

        return floor_rescaled_similarities, is_high_reliability_region, is_ood

    def get_hr_region_alpha_vectorized(self, rescaled_similarities, batch_sdm_outputs, predictions):
        """
        Assign each instance to the most conservative (i.e., closest to 1) nested high-reliability region
        whose gates it passes (see calculateOutputThresholdsAdaptive()). The regions in self.hr_regions are
        sorted descending by alpha, so the first region whose gates pass is the assignment.

        Parameters
        ----------
        rescaled_similarities : torch.Tensor
            Shape [batch_size] containing rescaled similarity values
        batch_sdm_outputs : torch.Tensor
            Shape [batch_size, numberOfClasses] containing SDM outputs
        predictions : torch.Tensor
            Shape [batch_size] containing predicted class indices

        Returns
        -------
        hr_region_alphas : torch.Tensor
            Shape [batch_size] containing the alpha of the assigned region for each instance, with 0.
            indicating that no region's gates pass (i.e., a rejected point).
        """
        batch_size = rescaled_similarities.shape[0]
        # float64, so that the exact decimal ladder alpha values (e.g., 0.85) are preserved in the returned
        # values (and downstream JSON serialization), rather than the nearest float32 (e.g., 0.8500000238...):
        hr_region_alphas = torch.zeros(batch_size, dtype=torch.float64, device=rescaled_similarities.device)
        for hr_region in self.hr_regions:
            _, is_in_region, _ = self.get_high_reliability_region_indicator_vectorized(
                rescaled_similarities=rescaled_similarities,
                batch_sdm_outputs=batch_sdm_outputs,
                predictions=predictions,
                min_rescaled_similarity=hr_region["min_rescaled_similarity"],
                hr_output_thresholds=hr_region["output_thresholds"],
                hr_class_conditional_accuracy=hr_region["alpha"])
            newly_assigned = is_in_region.to(hr_region_alphas.device) & (hr_region_alphas == 0.)
            hr_region_alphas[newly_assigned] = hr_region["alpha"]
        return hr_region_alphas

    def get_hr_region_alpha_lower_vectorized(self, rescaled_similarities, predictions, batch_f, batch_q,
                                             batch_distance_quantile_per_class):
        """
        The lower (DKW-band) counterpart to get_hr_region_alpha_vectorized(): assign each instance to the most
        conservative nested region whose gates it passes when the Distance quantile is replaced by its lower
        estimate given the effective sample size. For each region, the DKW error term is pinned to that
        region's alpha, so more conservative (higher alpha) regions require wider intervals, ceteris paribus,
        while the less stringent claims of the lower alpha regions are not subjected to vacuously wide bounds.
        The SDM output, and by extension the rescaled Similarity (q'_{lower}), are recalculated per-region
        accordingly.

        In addition to the assignment, the per-instance band quantities are returned, selected at the alpha of
        each instance's assigned region (i.e., the values with which the instance's banded admission actually
        holds). Instances not assigned to any region reflect maximum uncertainty: the error term is 1 (the
        maximum, matching the default of get_cumulative_effective_sample_sizes_and_errors_vectorized()), so
        after clamping, d_lower = 0 and d_upper = 1, the corresponding lower SDM output goes to parity, and
        q'_{lower} follows accordingly.

        Parameters
        ----------
        rescaled_similarities : torch.Tensor
            Shape [batch_size] containing the (centroid) rescaled similarity values, from which the effective
            sample sizes are estimated
        predictions : torch.Tensor
            Shape [batch_size] containing predicted class indices
        batch_f : torch.Tensor
            Shape [batch_size, numberOfClasses] containing the un-normalized logits (z')
        batch_q : torch.Tensor
            Shape [batch_size, 1] containing the Similarity (q) values
        batch_distance_quantile_per_class : torch.Tensor
            Shape [batch_size, numberOfClasses] containing the (centroid) Distance quantiles

        Returns
        -------
        hr_region_alphas_lower : torch.Tensor
            Shape [batch_size] (float64, so that the exact decimal ladder alpha values are preserved)
            containing the alpha of the assigned region for each instance, with 0. indicating that no region's
            gates pass (i.e., a rejected point).
        selected_effective_cdf_sample_size_error : torch.Tensor
            Shape [batch_size, numberOfClasses]
        selected_batch_d_cdf_lower : torch.Tensor
            Shape [batch_size, numberOfClasses]
        selected_batch_d_cdf_upper : torch.Tensor
            Shape [batch_size, numberOfClasses]
        selected_batch_sdm_d_cdf_lower : torch.Tensor
            Shape [batch_size, numberOfClasses]
        selected_batch_sdm_d_cdf_upper : torch.Tensor
            Shape [batch_size, numberOfClasses]
        selected_rescaled_similarities_lower : torch.Tensor
            Shape [batch_size]
        """
        batch_size = rescaled_similarities.shape[0]
        hr_region_alphas_lower = torch.zeros(batch_size, dtype=torch.float64,
                                             device=rescaled_similarities.device)
        # Initialize the selected quantities with the maximum-uncertainty band (an error term of 1), which is
        # retained for any instance that is not assigned to a region:
        selected_effective_cdf_sample_size_error = torch.ones(batch_size, self.numberOfClasses,
                                                              device=self.device)
        selected_batch_d_cdf_lower, selected_batch_d_cdf_upper, \
            selected_batch_sdm_d_cdf_lower, selected_batch_sdm_d_cdf_upper = \
            self.get_sdm_output_for_d_cdf_lower_and_upper(
                batch_effective_cdf_sample_size_error=
                selected_effective_cdf_sample_size_error.to(batch_f.device),
                batch_f=batch_f,
                batch_q=batch_q,
                batch_distance_quantile_per_class=batch_distance_quantile_per_class)
        selected_rescaled_similarities_lower, _ = \
            self.get_rescaled_similarity_for_eval_batch(
                cached_f_outputs=batch_f,
                dataset_q_values=batch_q,
                sdm_outputs=selected_batch_sdm_d_cdf_lower,
                return_tensors_on_cpu=False)
        for hr_region in self.hr_regions:
            _, effective_cdf_sample_size_error = \
                self.get_cumulative_effective_sample_sizes_and_errors_vectorized(
                    rescaled_similarities=rescaled_similarities,
                    alpha_value=hr_region["alpha"])
            batch_d_cdf_lower, batch_d_cdf_upper, batch_sdm_d_cdf_lower, batch_sdm_d_cdf_upper = \
                self.get_sdm_output_for_d_cdf_lower_and_upper(
                    batch_effective_cdf_sample_size_error=effective_cdf_sample_size_error.to(batch_f.device),
                    batch_f=batch_f,
                    batch_q=batch_q,
                    batch_distance_quantile_per_class=batch_distance_quantile_per_class)
            rescaled_similarities_lower, _ = \
                self.get_rescaled_similarity_for_eval_batch(
                    cached_f_outputs=batch_f,
                    dataset_q_values=batch_q,
                    sdm_outputs=batch_sdm_d_cdf_lower,
                    return_tensors_on_cpu=False)
            _, is_in_region_lower, _ = self.get_high_reliability_region_indicator_vectorized(
                rescaled_similarities=rescaled_similarities_lower,
                batch_sdm_outputs=batch_sdm_d_cdf_lower,
                predictions=predictions.to(batch_sdm_d_cdf_lower.device),
                min_rescaled_similarity=hr_region["min_rescaled_similarity"],
                hr_output_thresholds=hr_region["output_thresholds"],
                hr_class_conditional_accuracy=hr_region["alpha"])
            newly_assigned = is_in_region_lower.to(hr_region_alphas_lower.device) & (hr_region_alphas_lower == 0.)
            hr_region_alphas_lower[newly_assigned] = hr_region["alpha"]
            # Select this region's band quantities for the newly assigned instances:
            newly_assigned_error_device = newly_assigned.to(selected_effective_cdf_sample_size_error.device)
            selected_effective_cdf_sample_size_error[newly_assigned_error_device] = \
                effective_cdf_sample_size_error[newly_assigned_error_device]
            newly_assigned_band_device = newly_assigned.to(selected_batch_d_cdf_lower.device)
            selected_batch_d_cdf_lower[newly_assigned_band_device] = \
                batch_d_cdf_lower[newly_assigned_band_device]
            selected_batch_d_cdf_upper[newly_assigned_band_device] = \
                batch_d_cdf_upper[newly_assigned_band_device]
            selected_batch_sdm_d_cdf_lower[newly_assigned_band_device] = \
                batch_sdm_d_cdf_lower[newly_assigned_band_device]
            selected_batch_sdm_d_cdf_upper[newly_assigned_band_device] = \
                batch_sdm_d_cdf_upper[newly_assigned_band_device]
            newly_assigned_rescaled_device = newly_assigned.to(selected_rescaled_similarities_lower.device)
            selected_rescaled_similarities_lower[newly_assigned_rescaled_device] = \
                rescaled_similarities_lower[newly_assigned_rescaled_device]
        return hr_region_alphas_lower, selected_effective_cdf_sample_size_error, \
            selected_batch_d_cdf_lower, selected_batch_d_cdf_upper, \
            selected_batch_sdm_d_cdf_lower, selected_batch_sdm_d_cdf_upper, \
            selected_rescaled_similarities_lower

    def get_sdm_output_for_d_cdf_lower_and_upper(self, batch_effective_cdf_sample_size_error, batch_f, batch_q,
                                                 batch_distance_quantile_per_class):
        # Take the max error across classes:
        batch_max_sample_size_error_across_classes = torch.amax(batch_effective_cdf_sample_size_error,
                                                                dim=1,
                                                                keepdim=True)
        batch_d_cdf_lower = \
            torch.clamp(batch_distance_quantile_per_class - batch_max_sample_size_error_across_classes,
                        min=0.0, max=1.0)
        batch_d_cdf_upper = \
            torch.clamp(batch_distance_quantile_per_class + batch_max_sample_size_error_across_classes,
                        min=0.0, max=1.0)
        # Same as the standard SDM calculation, but now with the
        # lower and upper estimates for the distance quantile:
        batch_sdm_d_cdf_lower = self.soft_sdm_max(batch_f, batch_q,
                                                  distance_quantile_per_class=
                                                  batch_d_cdf_lower)
        batch_sdm_d_cdf_upper = self.soft_sdm_max(batch_f, batch_q,
                                                  distance_quantile_per_class=
                                                  batch_d_cdf_upper)
        return batch_d_cdf_lower, batch_d_cdf_upper, batch_sdm_d_cdf_lower, batch_sdm_d_cdf_upper

    def get_batch_eval_output_dictionary(self, rescaled_similarities: torch.Tensor, sdm_batch_outputs: torch.Tensor,
                                         predictions: torch.Tensor, batch_f: torch.Tensor, batch_q: torch.Tensor,
                                         batch_distance_quantile_per_class: torch.Tensor, d0_values: torch.Tensor,
                                         nearest_support_idx_values: torch.Tensor):
        with torch.no_grad():
            floor_rescaled_similarity_tensor, is_high_reliability_region_tensor, is_ood_tensor = \
                self.get_high_reliability_region_indicator_vectorized(
                    rescaled_similarities=rescaled_similarities,
                    batch_sdm_outputs=sdm_batch_outputs,
                    predictions=predictions.to(sdm_batch_outputs.device))

            # Effective sample sizes (these are alpha-independent; reported for reference). The error term of
            # the returned pair is discarded here: the per-instance error terms are instead selected at the
            # alpha of each instance's assigned region below. Since alpha_value only affects the
            # error term (which is discarded here),
            # the ladder's most conservative alpha is passed to satisfy the required argument:
            cumulative_effective_sample_sizes, _ = \
                self.get_cumulative_effective_sample_sizes_and_errors_vectorized(
                    rescaled_similarities=rescaled_similarities,
                    alpha_value=self.get_ladder_alphas()[0])

            # The current convention assumes self.ood_limit == 0 (e.g., the is_ood return values of the
            # indicator calls inside the helpers below are equivalent to is_ood_tensor above). We check that
            # assumption here in case it changes in the future:
            assert self.ood_limit == 0, "The current convention assumes self.ood_limit == 0."
            # Nested-region assignments: the most conservative alpha among the recorded regions whose gates
            # pass, with 0 indicating rejection at all recorded regions. For the lower counterpart, the DKW
            # error term is pinned to each region's alpha, and the per-instance band quantities
            # (effective_cdf_sample_size_error, batch_d_cdf_lower/upper, batch_sdm_d_cdf_lower/upper, and
            # rescaled_similarities_lower) are those of each instance's assigned region, with instances not
            # assigned to any region reflecting maximum uncertainty (an error term of 1, so d_lower = 0 and
            # d_upper = 1 after clamping, with the lower SDM output at parity):
            hr_region_alpha_tensor = self.get_hr_region_alpha_vectorized(
                rescaled_similarities=rescaled_similarities,
                batch_sdm_outputs=sdm_batch_outputs,
                predictions=predictions.to(sdm_batch_outputs.device))
            hr_region_alpha_lower_tensor, effective_cdf_sample_size_error, \
                batch_d_cdf_lower, batch_d_cdf_upper, batch_sdm_d_cdf_lower, batch_sdm_d_cdf_upper, \
                rescaled_similarities_lower = \
                self.get_hr_region_alpha_lower_vectorized(
                    rescaled_similarities=rescaled_similarities,
                    predictions=predictions,
                    batch_f=batch_f,
                    batch_q=batch_q,
                    batch_distance_quantile_per_class=batch_distance_quantile_per_class)
            floor_rescaled_similarity_lower_tensor = torch.floor(rescaled_similarities_lower).long()
            # The legacy single-region lower indicator is equivalent to assignment to the most conservative
            # region under the band, since that region's gates (with the DKW error term at its alpha) are
            # exactly the first test of the descending walk in get_hr_region_alpha_lower_vectorized():
            if len(self.hr_regions) > 0:
                is_high_reliability_region_lower_tensor = \
                    (hr_region_alpha_lower_tensor == self.hr_regions[0]["alpha"])
            else:
                is_high_reliability_region_lower_tensor = \
                    torch.zeros(rescaled_similarities.shape[0], dtype=torch.bool,
                                device=rescaled_similarities.device)
            results = []
            for rescaled_similarity, sdm_output, prediction, f, q, distance_quantile_per_class, \
                    d0_value, nearest_support_idx_value, floor_rescaled_similarity, is_high_reliability_region, \
                    is_ood, cumulative_effective_sample_sizes_per_class, effective_cdf_sample_size_error_per_class, \
                    d_cdf_lower, d_cdf_upper, sdm_output_d_lower, sdm_output_d_upper,\
                    rescaled_similarity_lower, \
                    floor_rescaled_similarity_lower, \
                    is_high_reliability_region_lower, \
                    hr_region_alpha, \
                    hr_region_alpha_lower in \
                    zip(rescaled_similarities, sdm_batch_outputs, predictions, batch_f,
                        batch_q, batch_distance_quantile_per_class,
                        d0_values,
                        nearest_support_idx_values, floor_rescaled_similarity_tensor,
                        is_high_reliability_region_tensor, is_ood_tensor, cumulative_effective_sample_sizes,
                        effective_cdf_sample_size_error, batch_d_cdf_lower, batch_d_cdf_upper,
                        batch_sdm_d_cdf_lower, batch_sdm_d_cdf_upper,
                        rescaled_similarities_lower,
                        floor_rescaled_similarity_lower_tensor,
                        is_high_reliability_region_lower_tensor,
                        hr_region_alpha_tensor,
                        hr_region_alpha_lower_tensor):

                prediction_meta_data = {
                        # Similarity value: q:
                        "q": q.item(),
                        # raw Distance value: d_nearest:
                        "d0": d0_value.item(),
                        # raw Magnitude value (un-normalized logits):
                        "f": f,  # tensor
                        # this is the predicted class, which may differ from the argmax of the sdm output iff the output
                        # goes to parity (e.g., if d=0):
                        # \hat{y}:
                        "prediction": prediction.item(),
                        # Already min (among Distance quantiles across classes), so take the first index:
                        "d": distance_quantile_per_class[0].item(),
                        "sdm_output": sdm_output,  # tensor
                        "rescaled_similarity": rescaled_similarity.item(),
                        "top_distance_idx": nearest_support_idx_value.item(),
                        # These use the DKW inequality based on the effective sample size to put bounds on the
                        # Distance empirical CDFs, with the error term pinned to the alpha of the region
                        # assigned under the band (i.e., "hr_region_alpha_lower" below). If the instance is not
                        # assigned to any region under the band (hr_region_alpha_lower == 0.), these reflect
                        # maximum uncertainty: an error term of 1, so d_lower = 0 and d_upper = 1 (after
                        # clamping), with sdm_output_d_lower at parity and rescaled_similarity_lower following
                        # accordingly:
                        "d_lower": d_cdf_lower[0].item(),
                        "d_upper": d_cdf_upper[0].item(),
                        "sdm_output_d_lower": sdm_output_d_lower,  # tensor
                        "sdm_output_d_upper": sdm_output_d_upper,  # tensor
                        "rescaled_similarity_lower": rescaled_similarity_lower.item(),
                        # Nested-region assignments: the most conservative (i.e., closest to 1) alpha among
                        # self.hr_regions whose gates pass, with 0. indicating rejection at all recorded
                        # regions. When at least one region exists,
                        # hr_region_alpha == self.hr_regions[0]["alpha"] iff is_high_reliability_region (and
                        # analogously for the lower counterparts), since the default (single-region) indicator
                        # evaluates exactly the most conservative recorded region:
                        "hr_region_alpha": hr_region_alpha.item(),
                        "hr_region_alpha_lower": hr_region_alpha_lower.item(),
                        # This is the effective sample size across classes using q'. Note the error term itself
                        #   is tied to the assigned
                        #   membership in the alpha (lower) region.
                        #   Note: There is deliberately no "_lower" counterpart of this field: The effective
                        #   sample size is anchored at the (centroid) rescaled similarity (Eq. 11 in 'SDM
                        #   Activations'), and the DKW error terms of Eq. 12. As such, all of the banded
                        #   ("_lower"/"_upper") quantities above are downstream of this single \hat{n}. A
                        #   sample size anchored at q'_{lower} would be circular (q'_{lower} depends on the
                        #   error term, which depends on \hat{n}):
                        "cumulative_effective_sample_sizes": cumulative_effective_sample_sizes_per_class,  # tensor

                        # TODO: streamline. These are convenience hold-overs from earlier versions (and are
                        #   currently still used by the eval scripts), but now
                        #   "hr_region_alpha" and "hr_region_alpha_lower" (and related) are sufficient (from which
                        #   these values, if needed, can be calculated by the caller).
                        # floor_rescaled_similarity is an int
                        "floor_rescaled_similarity": floor_rescaled_similarity.item(),
                        "is_high_reliability_region": is_high_reliability_region.item(),  # bool
                        "is_high_reliability_region_lower": is_high_reliability_region_lower.item(),  # bool
                        # floor_rescaled_similarity is an int
                        "floor_rescaled_similarity_lower": floor_rescaled_similarity_lower.item(),
                        # "OOD" can also be viewed as the points not assigned to a region, including d=0. This is up to
                        # the caller. This is_ood simply checks if q==0.
                        # is_ood is Bool.
                        "is_ood": is_ood.item(),  # bool
                        }
                results.append(prediction_meta_data)
            return results

    def get_q_and_d_from_exemplars(self, batch_f, exemplar_vectors, is_training_support=False,
                                   return_exemplar_vectors=False):
        # Arguments are currently assumed to be on cpu.
        # Fetch the distances. This will include the identity match if is_training_support=True, which is handled below.
        # Currently, we assume there are no duplicates in the data splits (or at least there are very few).
        eval_top_k_distances__including_self_if_training_document, \
            eval_top_k_distances_idx__including_self_if_training_document = \
            self.get_top_support_distances(exemplar_vectors.numpy())
        # FAISS uses numpy, but otherwise we aim to keep data structures in pytorch for consistency:
        d0_values_tensor = torch.tensor(
            eval_top_k_distances__including_self_if_training_document[:, 1 if is_training_support else 0])
        nearest_support_idx_tensor = \
            torch.tensor(
                eval_top_k_distances_idx__including_self_if_training_document[:, 1 if is_training_support else 0])

        # get q values a dn d_nearest; is_training_support=True will discard the first (identity) match
        eval_dataset_q_values, eval_dataset_d0_values = \
            self.get_summary_stats_for_eval_vectorized(
                eval_set_size=exemplar_vectors.shape[0],
                top_k_distances=eval_top_k_distances__including_self_if_training_document,
                top_k_distances_idx=eval_top_k_distances_idx__including_self_if_training_document,
                eval_logits=batch_f,
                is_training_support=is_training_support)

        eval_dataset_distance_quantile_per_class = \
            self.get_distance_quantiles_vectorized(eval_dataset_d0_values,
                                                    train_trueClass_To_dCDF=self.train_trueClass_To_dCDF if is_training_support else None)
        # Typically with an SDM network, eval_dataset_distance_quantile_per_class will need to be expanded to the
        # size of the language model's output vocabulary (or a multiple thereof), which we leave to the caller.
        # Note that
        # each column of eval_dataset_distance_quantile_per_class is the same value, so expansion can just use the
        # first column, as needed.
        if return_exemplar_vectors:
            return eval_dataset_q_values, eval_dataset_distance_quantile_per_class, batch_f, \
                d0_values_tensor, nearest_support_idx_tensor, exemplar_vectors
        else:
            return eval_dataset_q_values, eval_dataset_distance_quantile_per_class, batch_f, \
                d0_values_tensor, nearest_support_idx_tensor

    def single_pass_forward(self, batch_exemplar_vectors, batch_f,
                            return_k_nearest_training_idx_in_prediction_metadata=1, is_training_support=False):
        # Note: Currently we always return the nearest 1 document idx from training and
        # return_k_nearest_training_idx_in_prediction_metadata is ignored.
        main_device = batch_exemplar_vectors.device
        with torch.no_grad():
            # get summary stats and run inference all in one pass
            # we assume batch size one:
            assert batch_exemplar_vectors.shape[0] == 1
            assert batch_f.shape[0] == 1
            # Currently this first function runs on cpu, since FAISS expects numpy:
            batch_q, batch_distance_quantile_per_class, batch_f, \
                batch_d0_values_tensor, batch_nearest_support_idx_tensor = \
                self.get_q_and_d_from_exemplars(batch_f=batch_f.cpu(),
                                                exemplar_vectors=batch_exemplar_vectors.cpu(),
                                                is_training_support=is_training_support)
            batch_f = batch_f.to(main_device)
            batch_q = batch_q.to(main_device)
            batch_distance_quantile_per_class = batch_distance_quantile_per_class.to(main_device)
            batch_sdm = \
                self.soft_sdm_max(batch_f, batch_q, distance_quantile_per_class=batch_distance_quantile_per_class)

            rescaled_similarities, predictions = \
                self.get_rescaled_similarity_for_eval_batch(
                    cached_f_outputs=batch_f,
                    dataset_q_values=batch_q,
                    sdm_outputs=batch_sdm,
                    return_tensors_on_cpu=False)

            results = self.get_batch_eval_output_dictionary(
                rescaled_similarities=rescaled_similarities,
                sdm_batch_outputs=batch_sdm,
                predictions=predictions,
                batch_f=batch_f.to(main_device),
                batch_q=batch_q.to(main_device),
                batch_distance_quantile_per_class=batch_distance_quantile_per_class.to(main_device),
                d0_values=batch_d0_values_tensor,
                nearest_support_idx_values=batch_nearest_support_idx_tensor)
            return results[0]

    def normalize_embeddings(self, embeddings):
        # (optional) mean centering of the input to the 1-D CNN of the sdm activation:
        return (embeddings - self.training_embedding_summary_stats[constants.STORAGE_KEY_SUMMARY_STATS_EMBEDDINGS_training_embedding_mean]) / \
            self.training_embedding_summary_stats[constants.STORAGE_KEY_SUMMARY_STATS_EMBEDDINGS_training_embedding_std]

    def forward(self, input, batch_q=None, batch_f=None, batch_distance_quantile_per_class=None,
                forward_type=constants.FORWARD_TYPE_SENTENCE_LEVEL_PREDICTION, train=False, normalize_embeddings=True,
                return_k_nearest_training_idx_in_prediction_metadata=1):
        # The point-estimate prediction is always determined by batch_f.
        if batch_f is None or forward_type == constants.FORWARD_TYPE_GENERATE_EXEMPLAR_VECTORS:
            # input corresponds to:
            # [composition attributes (optional)] :: [Cumulative average LLM embeddings] :: [LLM embedding]
            # In the current version, this is not a convolution over sequence positions (i.e., the
            # width of the 1-D CNN is equivalent to the length of the input vector). However, this can be readily
            # adapted to the sequence case, as well (e.g., by adding a maxpool), but that is not currently implemented.
            batch_exemplar_vectors = input.unsqueeze(1)
            # global norm
            if normalize_embeddings:
                with torch.no_grad():
                    batch_exemplar_vectors = \
                        self.normalize_embeddings(batch_exemplar_vectors)
            batch_exemplar_vectors = self.conv(batch_exemplar_vectors).squeeze(2)
            batch_f = self.fc(batch_exemplar_vectors)

            assert len(batch_exemplar_vectors.shape) != 1
        if len(batch_f.shape) == 1:
            batch_f = batch_f.unsqueeze(0)
        if forward_type in [constants.FORWARD_TYPE_SINGLE_PASS_TEST,
                            constants.FORWARD_TYPE_SINGLE_PASS_TEST_WITH_EXEMPLAR]:
            prediction_meta_data = self.single_pass_forward(batch_exemplar_vectors, batch_f,
                                                            return_k_nearest_training_idx_in_prediction_metadata=
                                                            return_k_nearest_training_idx_in_prediction_metadata)
            if forward_type == constants.FORWARD_TYPE_SINGLE_PASS_TEST_WITH_EXEMPLAR:
                prediction_meta_data["exemplar_vector"] = batch_exemplar_vectors
            return prediction_meta_data

        assert batch_q is not None
        sdm_batch_output = \
            self.soft_sdm_max(batch_f, batch_q,
                              distance_quantile_per_class=batch_distance_quantile_per_class,
                              log=train, change_of_base=True)
        if forward_type == constants.FORWARD_TYPE_SENTENCE_LEVEL_PREDICTION:
            return batch_f, sdm_batch_output
        elif forward_type == constants.FORWARD_TYPE_GENERATE_EXEMPLAR_VECTORS:
            return batch_f, sdm_batch_output, batch_exemplar_vectors

    def export_properties_to_dict(self):
        json_dict = {constants.STORAGE_KEY_version: self.version,
                     constants.STORAGE_KEY_uncertaintyModelUUID: self.uncertaintyModelUUID,
                     constants.STORAGE_KEY_maxQAvailableFromIndexer: self.maxQAvailableFromIndexer,
                     constants.STORAGE_KEY_numberOfClasses: self.numberOfClasses,
                     constants.STORAGE_KEY_q_rescale_offset: self.q_rescale_offset,
                     constants.STORAGE_KEY_ood_limit: self.ood_limit,
                     constants.STORAGE_KEY_exemplar_vector_dimension: self.exemplar_vector_dimension,
                     constants.STORAGE_KEY_embedding_size: self.embedding_size,
                     constants.STORAGE_KEY_calibration_training_stage: self.calibration_training_stage,
                     constants.STORAGE_KEY_calibration_is_ood_indicators: self.calibration_is_ood_indicators,
                     constants.STORAGE_KEY_SUMMARY_STATS_EMBEDDINGS_training_embedding_summary_stats: self.training_embedding_summary_stats,
                     constants.STORAGE_KEY_is_sdm_network_verification_layer: self.is_sdm_network_verification_layer,
                     constants.STORAGE_KEY_alpha_resolution: self.alpha_resolution,
                     # Nested high-reliability regions, sorted descending by alpha. The output thresholds are
                     # converted to standard lists for JSON serialization (bit-exactly; see
                     # import_properties_from_dict() for the inverse):
                     constants.STORAGE_KEY_hr_regions: [
                         {"alpha": hr_region["alpha"],
                          "min_rescaled_similarity": float(hr_region["min_rescaled_similarity"]),
                          "output_thresholds": [float(threshold) for threshold in hr_region["output_thresholds"]]}
                         for hr_region in self.hr_regions],
                     }

        trueClass_To_dCDF_json_flat = {}
        for label in self.trueClass_To_dCDF.keys():
            trueClass_To_dCDF_json_flat[label] = self.trueClass_To_dCDF[label]
        json_dict[constants.STORAGE_KEY_trueClass_To_dCDF] = trueClass_To_dCDF_json_flat

        train_trueClass_To_dCDF_json_flat = {}
        for label in self.train_trueClass_To_dCDF.keys():
            train_trueClass_To_dCDF_json_flat[label] = self.train_trueClass_To_dCDF[label]
        json_dict[constants.STORAGE_KEY_train_trueClass_To_dCDF] = train_trueClass_To_dCDF_json_flat

        trueClass_To_qCumulativeSampleSizeArray_json_flat = {}
        for label in self.trueClass_To_qCumulativeSampleSizeArray.keys():
            trueClass_To_qCumulativeSampleSizeArray_json_flat[label] = self.trueClass_To_qCumulativeSampleSizeArray[label]
        json_dict[constants.STORAGE_KEY_trueClass_To_qCumulativeSampleSizeArray] = trueClass_To_qCumulativeSampleSizeArray_json_flat
        return json_dict

    def import_properties_from_dict(self, json_dict, load_for_inference=False):
        # When loading from disk, this must be called after class init before calibrating new data points.
        # Note that in JSON, int dictionary keys become strings

        # The nested regions are the sole persisted source of the region parameters. The float values
        # round-trip bit-exactly through JSON (binary32 -> binary64 -> shortest-repr decimal string ->
        # binary64 -> binary32 are all exact conversions), so this is equivalent to persisting torch tensors.
        self.hr_regions = []
        for hr_region_json in json_dict[constants.STORAGE_KEY_hr_regions]:
            self.hr_regions.append({
                "alpha": hr_region_json["alpha"],
                "min_rescaled_similarity": hr_region_json["min_rescaled_similarity"],
                "output_thresholds": torch.tensor(hr_region_json["output_thresholds"], dtype=torch.float32),
            })

        trueClass_To_dCDF_json_flat = json_dict[constants.STORAGE_KEY_trueClass_To_dCDF]
        for trueClass in range(self.numberOfClasses):
            trueClass_str = str(trueClass)
            if trueClass_str in trueClass_To_dCDF_json_flat:
                self.trueClass_To_dCDF[trueClass] = trueClass_To_dCDF_json_flat[trueClass_str]
            else:
                self.trueClass_To_dCDF[trueClass] = []

        trueClass_To_qCumulativeSampleSizeArray_json_flat = \
            json_dict[constants.STORAGE_KEY_trueClass_To_qCumulativeSampleSizeArray]
        for trueClass in range(self.numberOfClasses):
            trueClass_str = str(trueClass)
            if trueClass_str in trueClass_To_qCumulativeSampleSizeArray_json_flat:
                self.trueClass_To_qCumulativeSampleSizeArray[trueClass] = \
                    trueClass_To_qCumulativeSampleSizeArray_json_flat[trueClass_str]
            else:
                self.trueClass_To_qCumulativeSampleSizeArray[trueClass] = []

        if self.is_sdm_network_verification_layer and not load_for_inference:
            train_trueClass_To_dCDF_json_flat = json_dict[constants.STORAGE_KEY_train_trueClass_To_dCDF]
            for trueClass in range(self.numberOfClasses):
                trueClass_str = str(trueClass)
                if trueClass_str in train_trueClass_To_dCDF_json_flat:
                    self.train_trueClass_To_dCDF[trueClass] = train_trueClass_To_dCDF_json_flat[trueClass_str]
                else:
                    self.train_trueClass_To_dCDF[trueClass] = []
        else:
            self.train_trueClass_To_dCDF = {}

# Internal notes: When re-implementing in other languages, here are some things to remember to check:
# - Always use the ground-truth class for the main CDF structures when collecting the original statistics over
#   the calibration set.
# - Remember to properly address prediction flips, which can happen when the SDM output goes to parity when d=0. This
#   is straightforward to address: As in the original paper, the predicted class is always determined by the
#   argmax over z'.
# - Don't forget to sort the CDF structures.
# - Properly handle the boundaries when determining the distance quantiles. See the note in the appendix of the
#   original paper.
# - Currently we are inconsistent with variable casing, as a consequence of simplifying conversions
#    between the Swift and Python codebases. (Swift and Python use different conventions.)
