# Copyright Reexpress AI, Inc. All rights reserved.

import constants
import data_validator

import torch
import torch.nn as nn
# import torch.nn.functional as F
import numpy as np
import faiss
from collections import namedtuple


# Steps for robust Similarity-Distance-Magnitude Calibration:
# References are to https://arxiv.org/abs/2502.20167
#
# 1. Train model against training set, using soft_sdm_max.
#     CDF(d_nearest) for training is over training, and q is calculated against training, excluding the identity match
#         (the first epoch does not rescale and uses the equivalent of a standard softmax and CrossEntropy loss)
#     CDF(d_nearest) for calibration is over calibration, and q is calculated against training
#     (subsequently, CDF(d_nearest) is over calibration for new, unseen test instances,
#     and q is calculated against training)
#     Note that the class-wise CDFs for d_nearest are calculated excluding q=0 instances, which are considered OOD.
#     They are considered OOD because with q=0, the distance to the nearest match is undefined,
#     since the nearest match is not a similar instance, by definition.
# 2. Train model_rescaler against calibration. The convention is to use a batch size of 1 to train the numberOfClasses^2
#     parameters. (Note that model_rescaler does not have bias parameters.)
# 3. Calculate the threshold (over calibration) to detect the high-probability region. (Alg. 3)
# 4. Collect the sample size summary statistics (needed at test-time).
# 5. Repeat the process J times, collecting the summary statistics for the robust estimates. These are collected in
#     the class UncertaintyStatistics.
#
# At test-time (as calculated for a single instance in `single_pass_forward`):
#     1. Calculate the index-conditional uncertainty estimate (Eq. 28)
#     2. The non-rejected points from (1) are those suitable for final decision-making. If needed to triage the
#         remedial actions of the rejected points, max(0, p(\hat{y}_lower)-m^\hat{y}_floor(soft q) (i.e., the
#         left condition of Eq. 28) can be used with the understanding that the
#         estimates are of unspecified reliability. The
#         points with floor(log((2+q)^{unnormalized-output-CDF_yhat})) == 0 are strictly OOD.


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
                 cdfThresholdTolerance: float = constants.defaultCdfThresholdTolerance,
                 exemplar_vector_dimension: int = constants.keyModelDimension,
                 trueClass_To_dCDF = None,
                 trueClass_To_qCumulativeSampleSizeArray = None,
                 trueClass_To_unrescaledOutputCDF = None,
                 non_odd_thresholds = None,
                 non_odd_class_conditional_accuracy: float = 0.0,
                 alpha: float = constants.defaultCdfAlpha,
                 maxQAvailableFromIndexer: int = constants.maxQAvailableFromIndexer,
                 calibration_training_stage: int = 0,
                 min_valid_qbin_for_class_conditional_accuracy: int = np.inf,
                 training_embedding_summary_stats = None,
                 is_gen_ai=False,
                 gen_ai_vocab=0,
                 global_embedding_size=0,
                 composition_attributes_size=0,
                 top_logits_k=constants.top_logits_k,
                 # the following can be None at test-time to save memory, if desired:
                 calibration_labels = None,
                 calibration_predicted_labels = None,
                 calibration_uuids = None,
                 calibration_unrescaled_CDFquantiles = None,
                 calibration_soft_qbins = None,
                 calibration_is_ood_indicators = None,
                 # These are None on re-load to avoid overwriting learned weights.
                 gen_ai_model_lm_head_weights = None,
                 train_trueClass_To_dCDF = None
                 ):

        super(SimilarityDistanceMagnitudeCalibrator, self).__init__()

        self.uncertaintyModelUUID = uncertaintyModelUUID
        self.cdfThresholdTolerance = cdfThresholdTolerance
        self.numberOfClasses = numberOfClasses

        # these are currently numpy arrays:
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
        self.calibration_labels = calibration_labels  # UNLIKE TRAIN, THESE ARE TORCH TENSORS
        self.calibration_predicted_labels = calibration_predicted_labels  # TORCH TENSOR
        self.calibration_uuids = calibration_uuids  # JSON
        self.calibration_unrescaled_CDFquantiles = calibration_unrescaled_CDFquantiles
        self.calibration_soft_qbins = calibration_soft_qbins
        if calibration_is_ood_indicators is None:
            self.calibration_is_ood_indicators = []
        else:
            self.calibration_is_ood_indicators = calibration_is_ood_indicators  # list: 0 == not OOD; 1 == is OOD

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
        if trueClass_To_unrescaledOutputCDF is None:
            self.trueClass_To_unrescaledOutputCDF = {}
        else:
            self.trueClass_To_unrescaledOutputCDF = trueClass_To_unrescaledOutputCDF

        self.maxQAvailableFromIndexer = maxQAvailableFromIndexer

        self.q_rescale_offset = constants.q_rescale_offset  # This typically should not change.
        self.ood_limit = constants.ood_limit  # This typically should not change.
        self.min_valid_qbin_for_class_conditional_accuracy = min_valid_qbin_for_class_conditional_accuracy

        # self.trueClass_To_normalized_OutputCDF_non_ood = trueClass_To_normalized_OutputCDF_non_ood  # this does not need to be a class attribute; we just save thresholds
        self.non_odd_thresholds = non_odd_thresholds
        if self.non_odd_thresholds is None:
            self.non_odd_thresholds = np.zeros(self.numberOfClasses)

        # non_odd_class_conditional_accuracy is per-class, but this value is tied across classes.
        self.non_odd_class_conditional_accuracy = non_odd_class_conditional_accuracy
        self.alpha = alpha

        self.exemplar_vector_dimension = exemplar_vector_dimension
        self.embedding_size = embedding_size

        # additional parameters for genai
        self.global_embedding_size = global_embedding_size
        self.composition_attributes_size = composition_attributes_size
        self.is_gen_ai = is_gen_ai
        self.gen_ai_vocab = gen_ai_vocab
        self.top_logits_k = top_logits_k

        # weights:
        # [composition attributes (optional)] :: [Cumulative average LLM embeddings (optional)] :: [LLM embedding]
        # for gen ai, typically:
        # [Cumulative average LLM embeddings (up to and including t)] :: [LLM embedding at current token t]
        exemplar_network_input_size = self.composition_attributes_size + self.global_embedding_size + self.embedding_size
        self.conv = nn.Conv1d(1, self.exemplar_vector_dimension, exemplar_network_input_size,
                              stride=exemplar_network_input_size)
        self.fc = nn.Linear(self.exemplar_vector_dimension, self.numberOfClasses)  # for router / verificationLayer
        if self.is_gen_ai:
            gen_ai_dtype = torch.bfloat16
            # gen_ai_dtype = torch.float32
            assert self.numberOfClasses == 2, f"Only binary preferences are currently supported."
            # The considered LLM model has no bias.
            # These linear layers should exactly match the LLM model being fine-tuned. For the more general setting
            # of updating all weights of M^pos, the full weights of M^ref and M^neg would be needed, instead of just
            # the linear layers.
            self.fc_original = nn.Linear(self.embedding_size, self.gen_ai_vocab,
                                         bias=False)
            self.fc_negative = nn.Linear(self.embedding_size, self.gen_ai_vocab,
                                         bias=False)
            self.fc_positive = nn.Linear(self.embedding_size, self.gen_ai_vocab,
                                         bias=False)

            if gen_ai_model_lm_head_weights is not None:
                print(f"Initializing and freezing original lm weights and duplicating the final linear layers for "
                      f"updating.")
                assert self.gen_ai_vocab == gen_ai_model_lm_head_weights.shape[0]
                assert self.embedding_size == gen_ai_model_lm_head_weights.shape[1], \
                    f"{self.embedding_size}, {gen_ai_model_lm_head_weights.shape[1]}"
                # transfer and upcast: currently the sdm() numerical stability assumes full precision
                self.fc_original.weight = nn.Parameter(gen_ai_model_lm_head_weights.to(
                    gen_ai_dtype).detach().clone())
                self.fc_negative.weight = nn.Parameter(gen_ai_model_lm_head_weights.to(
                    gen_ai_dtype).detach().clone())
                self.fc_positive.weight = nn.Parameter(gen_ai_model_lm_head_weights.to(
                    gen_ai_dtype).detach().clone())
                self.fc_original.weight.requires_grad = False  # original model is not updated and is used for regularization
                self.fc_negative.weight.requires_grad = False
                self.fc_positive.weight.requires_grad = True

        self.model_rescaler = nn.Linear(self.numberOfClasses, self.numberOfClasses, bias=False)

        # Support index is saved separately, as it may be quite large. See setters and getters below.
        self.support_index = None

        self.calibration_training_stage = calibration_training_stage

        # self.kEPS = 1e-12  # Apple M2 Ultra;
        # adjust as applicable for platform; conservatively can use, for example, torch.finfo(torch.float32).eps
        self.kEPS = torch.finfo(torch.float32).eps

    def reset_llm_weights(self, gen_ai_model_lm_head_weights, reset_to_float32=True):
        print("Resetting LLM weights")
        assert self.gen_ai_vocab == gen_ai_model_lm_head_weights.shape[0]
        assert self.embedding_size == gen_ai_model_lm_head_weights.shape[1], \
            f"{self.embedding_size}, {gen_ai_model_lm_head_weights.shape[1]}"
        # transfer and upcast: currently the sdm() numerical stability assumes full precision
        if reset_to_float32:
            print(f"\tResetting to torch.float32")
            gen_ai_dtype = torch.float32
        else:
            gen_ai_dtype = torch.bfloat16

        self.fc_original.weight = nn.Parameter(gen_ai_model_lm_head_weights.to(
            gen_ai_dtype).detach().clone())
        self.fc_negative.weight = nn.Parameter(gen_ai_model_lm_head_weights.to(
            gen_ai_dtype).detach().clone())
        self.fc_positive.weight = nn.Parameter(gen_ai_model_lm_head_weights.to(
            gen_ai_dtype).detach().clone())
        self.fc_original.weight.requires_grad = False  # original model is not updated and is used for regularization
        self.fc_negative.weight.requires_grad = False
        self.fc_positive.weight.requires_grad = True

    def increment_model_calibration_training_stage(self, set_value=None):
        self.calibration_training_stage = set_value

    def set_train_predicted_labels(self, train_predicted_labels):
        # is a numpy() array; TODO: update all occurrences to torch tensor
        self.train_predicted_labels = train_predicted_labels

    def set_calibration_predicted_labels(self, calibration_predicted_labels):
        # torch tensor:
        self.calibration_predicted_labels = calibration_predicted_labels

    def set_train_trueClass_To_dCDF(self, train_trueClass_To_dCDF):
        # convenience for training gen ai model, since the distance from the generated output to the
        # force-decoded output is needed when training, but this is not needed for standard classification, and isn't
        # saved to save space in that case
        if self.is_gen_ai:
            self.train_trueClass_To_dCDF = train_trueClass_To_dCDF
        else:
            self.train_trueClass_To_dCDF = {}

    # def construct_support_index(self,
    #                             support_exemplar_vectors_numpy=None, calibration_exemplar_vectors_numpy=None,
    #                             k=None):
    #     # Note that any existing support index will be overwritten
    #     assert support_exemplar_vectors_numpy is not None
    #     assert calibration_exemplar_vectors_numpy is not None
    #     dimensions = self.exemplar_vector_dimension
    #     assert support_exemplar_vectors_numpy.shape[1] == self.exemplar_vector_dimension
    #     assert calibration_exemplar_vectors_numpy.shape[1] == self.exemplar_vector_dimension
    #     if k is None:
    #         k = self.maxQAvailableFromIndexer
    #     support_index = faiss.IndexFlatL2(dimensions)  # build the index
    #     support_index.add(support_exemplar_vectors_numpy)  # add exemplar vectors to the index
    #     if k > support_index.ntotal:
    #         k = support_index.ntotal  # indexes will be -1 if exceeds, so hard constraint here
    #     if constants.USE_GPU_FAISS_INDEX:
    #         # start move to gpu
    #         gpu_id = 0
    #         res = faiss.StandardGpuResources()
    #         support_index = faiss.index_cpu_to_gpu(res, gpu_id, support_index)
    #         # end move
    #     top_k_distances, top_k_distances_idx = support_index.search(calibration_exemplar_vectors_numpy, k)
    #     self.support_index = support_index
    #     return support_index, top_k_distances, top_k_distances_idx
    def construct_support_index(self,
                                support_exemplar_vectors_numpy=None, calibration_exemplar_vectors_numpy=None,
                                k=None,
                                ood_support_exemplar_vectors_numpy=None,
                                ood_support_labels=None,
                                ood_support_predicted_labels=None,
                                ood_support_document_ids=None
                                ):
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
            for ood_i, ood_support_document_id in enumerate(ood_support_document_ids):
                self.add_to_support(label=ood_support_labels[ood_i],
                                    predicted_label=ood_support_predicted_labels[ood_i],
                                    document_id=ood_support_document_id,
                                    exemplar_vector=None)  # exemplar_vector is None, since we batch import in next line
            support_index.add(ood_support_exemplar_vectors_numpy)
            print(f">Added {len(ood_support_document_ids)} OOD/additional instances to the training support.<")
        if k > support_index.ntotal:
            k = support_index.ntotal  # indexes will be -1 if exceeds, so hard constraint here
        if constants.USE_GPU_FAISS_INDEX:
            # start move to gpu
            gpu_id = 0
            res = faiss.StandardGpuResources()
            support_index = faiss.index_cpu_to_gpu(res, gpu_id, support_index)
            # end move
        top_k_distances, top_k_distances_idx = support_index.search(calibration_exemplar_vectors_numpy, k)
        self.support_index = support_index
        return support_index, top_k_distances, top_k_distances_idx

    def set_support_index(self, support_index):
        self.support_index = support_index

    def add_to_support(self, label: int, predicted_label: int, document_id: str, exemplar_vector=None):
        # We assume the caller has checked that d0 != 0
        # exemplar_vector is a numpy array
        if exemplar_vector is not None:
            self.support_index.add(exemplar_vector)
        self.train_labels = np.append(self.train_labels, label)
        self.train_predicted_labels = np.append(self.train_predicted_labels, predicted_label)
        self.train_uuids.append(document_id)

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

    def soft_sdm_max(self, batch_input, q, distance_quantile_per_class=None, log=False, change_of_base=True):
        """
        Instead of softmax e^val/sum(e^val), we normalize via q^(val_y*(1-CDF(d)_y))/sum(q^(val_y*(1-CDF(d)_y)),
        increasing the relative amplification/sharpness of the distribution for higher Similarity (q) values
        and lower distances (d). distance_quantile_per_class is assumed to be the same across classes; in this way,
        the argmax does not change relative to argmax(batch_input, dim=1). In practice, it typically is
        recommended to take the min across classes as the distance quantile and use the same value across classes.

        Parameters
        ----------
        batch_input
            torch.tensor
            shape == [batch size, self.numberOfClasses]; if, e.g., batch_size == 1, [1, self.numberOfClasses]
        q
            torch.tensor
            shape == [batch size, 1], with each value in [0, constants.maxQAvailableFromIndexer]. This function then
            adds self.q_rescale_offset to q. For the standard softmax (assuming self.q_rescale_offset==2, as
            is typical), use q=torch.tensor([[np.e-2],...]).
        distance_quantile_per_class
            torch.tensor, or None
            If not None, shape == [batch size, self.numberOfClasses], with each quantile in [0,1].
        log
            If True, take the log (useful for training)
        change_of_base
            If log == True, use q as the base of the logarithm. Should always be True in practice; only included
            for reference/debugging.

        Returns
        -------
        [batch size, self.numberOfClasses]
        """

        assert len(batch_input.shape) == 2
        if not self.is_gen_ai:
            assert batch_input.shape[1] == self.numberOfClasses
        assert batch_input.shape[0] == q.shape[0]
        assert q.shape[1] == 1
        if distance_quantile_per_class is not None:
            assert batch_input.shape == distance_quantile_per_class.shape
        q_factor = self.q_rescale_offset + q
        batch_input = batch_input - torch.amax(batch_input, dim=1, keepdim=True)  # for numerical stability
        if distance_quantile_per_class is not None:
            rescaled_distribution = q_factor ** (batch_input * distance_quantile_per_class)
        else:
            rescaled_distribution = q_factor ** batch_input
        if log:  # log_base{q}
            # self.kEPS  # for numerical stability
            rescaled_distribution = torch.log(rescaled_distribution+self.kEPS) - \
                                    torch.log(torch.sum(rescaled_distribution, dim=1)+self.kEPS).unsqueeze(1)
            if change_of_base:
                # q_factor is always at least self.q_rescale_offset = 2
                return rescaled_distribution / torch.log(q_factor)
            else:
                return rescaled_distribution
        else:
            return rescaled_distribution / torch.sum(rescaled_distribution, dim=1).unsqueeze(1)

    def swiftCodeQuantile(self, float_list, quantileProportion: float):  # matches Swift v1
        quantileIndex = min(int(quantileProportion * len(float_list)), len(float_list) - 1)
        return np.sort(float_list)[quantileIndex]  # could sort once, as with CDF structures, but currently not called at inference, so not an issue

    def getCdfThresholdForClass(self, normalized_output_for_true_class, alpha):
        if len(normalized_output_for_true_class) > 0:
            return max(self.swiftCodeQuantile(normalized_output_for_true_class, 1 - alpha), 0.0)
        return 0.0  # conservative (no information about class, so always included)

    def calculateOutputThresholdsAdaptive(self, trueClass_To_rescaled_OutputCDF_non_ood, all_bins):
        if len(all_bins) is None:
            print(constants.ERROR_MESSAGES_NO_THRESHOLD_FOUND)
            return
        all_bins = list(set(all_bins))
        all_bins.sort()
        # trueClass_To_rescaled_OutputCDF_non_ood must have values from a categorical distribution for this to be valid.
        # reset if present:
        self.non_odd_class_conditional_accuracy = 0.0
        self.non_odd_thresholds = np.zeros(self.numberOfClasses)
        self.min_valid_qbin_for_class_conditional_accuracy = np.inf

        for candidate_bin in all_bins:
            trueClass_To_CDF = {}
            for trueLabel in range(self.numberOfClasses):
                trueClass_To_CDF[trueLabel] = []
                if trueLabel in trueClass_To_rescaled_OutputCDF_non_ood:
                    filtered = []
                    filtered_rescaled_outputs = []
                    for output_qbin_pair in trueClass_To_rescaled_OutputCDF_non_ood[trueLabel]:
                        output = output_qbin_pair[0]
                        qbin = output_qbin_pair[1]
                        if qbin >= candidate_bin:
                            filtered_rescaled_outputs.append(output)
                            filtered.append(output_qbin_pair)
                    trueClass_To_CDF[trueLabel] = filtered_rescaled_outputs
                    trueClass_To_rescaled_OutputCDF_non_ood[trueLabel] = filtered  # reduce
            thresholds = np.zeros(self.numberOfClasses)
            for trueLabel in range(self.numberOfClasses):
                if trueLabel in trueClass_To_CDF:
                    rescaled_outputs = trueClass_To_CDF[trueLabel]
                    threshold = self.getCdfThresholdForClass(normalized_output_for_true_class=rescaled_outputs,
                                                             alpha=self.alpha)
                    thresholds[trueLabel] = threshold
            if np.all(thresholds >= self.alpha):
                self.non_odd_thresholds = thresholds
                self.min_valid_qbin_for_class_conditional_accuracy = candidate_bin
                self.non_odd_class_conditional_accuracy = self.alpha
                print(
                    f"Min q bin to achieve class-conditional accuracy of {self.alpha}: {self.min_valid_qbin_for_class_conditional_accuracy}")
                print(f"Thresholds: {self.non_odd_thresholds}")
                print(f"Class-conditional accuracy estimate: {self.non_odd_class_conditional_accuracy}")
                break

        if self.non_odd_class_conditional_accuracy == 0.0:
            print(constants.ERROR_MESSAGES_NO_THRESHOLD_FOUND)

    def set_class_conditional_non_ood_threshold(self, calibration_predicted_labels, calibration_unrescaled_CDFquantiles, calibration_soft_qbins, true_labels):
        # Threshold for points >= self.min_valid_qbin_for_class_conditional_accuracy
        assert self.alpha >= (1.0 / self.numberOfClasses), \
            f"ERROR: --alpha must be great than 1/(total number of classes)"
        trueClass_To_rescaled_OutputCDF_non_ood = {}
        predicted_class_to_bin_to_output_magnitudes = {}
        predicted_class_to_bin_to_median_output_magnitude = {}
        for label in range(self.numberOfClasses):
            trueClass_To_rescaled_OutputCDF_non_ood[label] = []
            self.trueClass_To_qCumulativeSampleSizeArray[label] = []
            predicted_class_to_bin_to_output_magnitudes[label] = {}
            predicted_class_to_bin_to_median_output_magnitude[label] = {}
            for hard_bin in range(constants.default_max_hard_bin):
                predicted_class_to_bin_to_output_magnitudes[label][hard_bin] = []
                predicted_class_to_bin_to_median_output_magnitude[label][hard_bin] = None
        all_non_ood_qbins = []
        self.eval()
        with torch.no_grad():
            # below is per-instance (i.e., batch size of 1)
            for label in range(self.numberOfClasses):
                self.trueClass_To_qCumulativeSampleSizeArray[label].sort()
            # reset OOD indicators, if present
            self.calibration_is_ood_indicators = []
            for logit_pred_y, unrescaled_CDFquantiles, soft_qbin, true_label in zip(calibration_predicted_labels,
                                                                                    calibration_unrescaled_CDFquantiles,
                                                                                    calibration_soft_qbins, true_labels):
                true_label = true_label.item()
                logit_pred_y = logit_pred_y.item()
                is_ood = False
                hard_bin = int(soft_qbin.item())
                if hard_bin <= self.ood_limit:
                    is_ood = True

                rescaled_output = self.forward(unrescaled_CDFquantiles.unsqueeze(0),
                                               soft_qbin.unsqueeze(1),
                                               forward_type=constants.FORWARD_TYPE_RESCALE_CACHED_CALIBRATION)
                if not is_ood:
                    rescaled_pred_y = torch.argmax(rescaled_output[0, :]).item()
                    if logit_pred_y != rescaled_pred_y:  # flipped
                        is_ood = True
                    else:
                        # indexed by *true label*
                        trueClass_To_rescaled_OutputCDF_non_ood[true_label].append(
                            (
                                rescaled_output[0, true_label].item(),
                                soft_qbin.item()
                            )
                        )
                        all_non_ood_qbins.append(soft_qbin.item())
                self.calibration_is_ood_indicators.append(int(is_ood))
                self.trueClass_To_qCumulativeSampleSizeArray[true_label].append(soft_qbin.item())
                if is_ood:
                    predicted_class_to_bin_to_output_magnitudes[logit_pred_y][0].append(
                        rescaled_output[0, logit_pred_y].item())
                else:
                    predicted_class_to_bin_to_output_magnitudes[logit_pred_y][hard_bin].append(
                        rescaled_output[0, logit_pred_y].item())

        assert len(self.calibration_is_ood_indicators) == self.calibration_labels.shape[0]
        print(f"Total OOD instances in the calibration set: {np.sum(self.calibration_is_ood_indicators)} "
              f"out of {len(self.calibration_is_ood_indicators)}: "
              f"{100*(np.sum(self.calibration_is_ood_indicators)/len(self.calibration_is_ood_indicators))}%")

        for label in range(self.numberOfClasses):
            trueClass_To_rescaled_OutputCDF_non_ood[label].sort(key=lambda x: x[1])  # sort by bin
            self.trueClass_To_qCumulativeSampleSizeArray[label].sort()
        self.calculateOutputThresholdsAdaptive(trueClass_To_rescaled_OutputCDF_non_ood, all_non_ood_qbins)
        self.increment_model_calibration_training_stage(set_value=modelCalibrationTrainingStages.complete)

        for label in range(self.numberOfClasses):
            for hard_bin in range(constants.default_max_hard_bin):
                if len(predicted_class_to_bin_to_output_magnitudes[label][hard_bin]) > 0:
                    predicted_class_to_bin_to_median_output_magnitude[label][hard_bin] = \
                        np.median(predicted_class_to_bin_to_output_magnitudes[label][hard_bin])
                    print(f"Label: {label}, hard bin: {hard_bin}, median magnitude: "
                          f"{predicted_class_to_bin_to_median_output_magnitude[label][hard_bin]}")
        return predicted_class_to_bin_to_median_output_magnitude

    def constructPredictionSetFromThresholds(self, numberOfClasses: int, softmax, thresholds):  # -> set(int)
        # assumes softmax and thresholds are numpy arrays
        if len(softmax) == len(thresholds) and numberOfClasses == len(softmax):
            return set((softmax >= thresholds).nonzero()[0])
        return set()

    def get_cumulative_effective_sample_sizes_and_errors(self, soft_qbin):
        # uses the DKW inequality to construct a band around the per-class empirical CDFs, given the effective sample
        # size for the soft-q-bin
        cumulative_effective_sample_sizes = torch.zeros(self.numberOfClasses)  # Note default is 0
        effective_cdf_sample_size_errors = torch.ones(self.numberOfClasses)  # Note default is 1
        alpha = 1 - self.alpha  # Note how we define alpha
        assert alpha < 0.5, "ERROR: The alpha value is likely misspecified. " \
                            "Check that it should not be 1-(the provided value). If such a high alpha value is " \
                            "desired, comment this assert."
        for label in range(self.numberOfClasses):
            sample_size_percentile = self.getCDFIndex(self.trueClass_To_qCumulativeSampleSizeArray,
                             soft_qbin,
                             label,
                             reverse=False)
            sample_size = min(int(sample_size_percentile * len(self.trueClass_To_qCumulativeSampleSizeArray[label])),
                                len(self.trueClass_To_qCumulativeSampleSizeArray[label]) - 1)
            cumulative_effective_sample_sizes[label] = sample_size
            if sample_size > 0 and alpha > 0:
                effective_cdf_sample_size_errors[label] = np.sqrt(np.log(2 / alpha) / (2 * sample_size))
        return cumulative_effective_sample_sizes, effective_cdf_sample_size_errors

    def getCDFIndex(self, trueClass_To_CDF, val, prediction, reverse=False, val_in_0to1=False):
        # np.searchsorted assumes ascending sort of initial argument
        if prediction not in trueClass_To_CDF or len(trueClass_To_CDF[prediction]) == 0:
            return 0.0
        if val_in_0to1 and len(trueClass_To_CDF[prediction]) > 0 and val >= trueClass_To_CDF[prediction][-1]:  # saturation guard
            assert not reverse
            return 1.0
        index = np.searchsorted(trueClass_To_CDF[prediction], val, side="left")  # will be 0 for len() == 0
        if reverse:  # use for distances
            return 1 - index / len(trueClass_To_CDF[prediction])
        else:
            return index / len(trueClass_To_CDF[prediction])

    def get_distance_quantiles(self, dataset_d0_values, train_trueClass_To_dCDF=None):
        take_min_across_percentiles = True
        dataset_distance_quantile_per_class = torch.zeros(dataset_d0_values.shape[0], self.numberOfClasses)
        for eval_i in range(dataset_d0_values.shape[0]):
            d0 = dataset_d0_values[eval_i].item()
            d_percentiles = torch.zeros(self.numberOfClasses)
            for label in range(self.numberOfClasses):
                if train_trueClass_To_dCDF is None:
                    d_percentiles[label] = self.getCDFIndex(self.trueClass_To_dCDF, d0, label,
                                                            reverse=True)
                else:
                    d_percentiles[label] = self.getCDFIndex(train_trueClass_To_dCDF, d0, label,
                                                            reverse=True)
            if take_min_across_percentiles:
                dataset_distance_quantile_per_class[eval_i, :] = torch.zeros(self.numberOfClasses) + torch.min(d_percentiles)
            else:
                # in the following cases, flips are possible in the sdm operator and typically the input
                # should be normalized to be in [0,infinity)
                dataset_distance_quantile_per_class[eval_i, :] = d_percentiles
        return dataset_distance_quantile_per_class

    def set_summary_stats_for_support(self, eval_set_size, top_k_distances, top_k_distances_idx,
                                      eval_logits,
                                      eval_labels, is_training_support=False, main_device=None):
        # Requires true labels. This overwrites self.trueClass_To_dCDF and
        # self.trueClass_To_qCumulativeSampleSizeArray, if is_training_support == False.
        # set_train_predicted_labels() must be called before running this function.
        # q is determined by the original logits; if subsequent renorming causes a flip, q is set to 0
        if not main_device:
            main_device = torch.device("cpu")
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
        trueClass_To_dCDF = {}
        trueClass_To_dataset_total_q_ood = {}
        trueClass_To_total_labels = {}
        for trueLabel in range(self.numberOfClasses):
            trueClass_To_dCDF[trueLabel] = []
            trueClass_To_dataset_total_q_ood[trueLabel] = 0
            trueClass_To_total_labels[trueLabel] = 0
        eval_predicted_labels = np.argmax(eval_logits, axis=1)
        dataset_q_values = torch.zeros(eval_set_size, 1)
        dataset_d0_values = torch.zeros(eval_set_size)
        for eval_i in range(eval_set_size):
            matched_true_labels = self.train_labels[top_k_distances_idx[eval_i]]
            matched_predicted_labels = self.train_predicted_labels[top_k_distances_idx[eval_i]]
            eval_predicted_label = eval_predicted_labels[eval_i].item()
            q_instance_value = 0
            for matching_i in range(top_k_distances_idx.shape[1]):
                if (not is_training_support) or (is_training_support and matching_i > 0):
                    if matched_true_labels[matching_i] == matched_predicted_labels[matching_i] and \
                            matched_predicted_labels[matching_i] == eval_predicted_label:
                        q_instance_value += 1
                    else:
                        break
            if is_training_support:
                d0 = top_k_distances[eval_i, 1].item()
            else:
                d0 = top_k_distances[eval_i, 0].item()
            # Note: If the prediction flips after subsequent re-normalization, q is set to 0 in eval
            dataset_q_values[eval_i, 0] = q_instance_value
            dataset_d0_values[eval_i] = d0
            is_valid = q_instance_value > self.ood_limit  # q == 0 are OOD
            true_label = eval_labels[eval_i].item()
            if is_valid and data_validator.isKnownValidLabel(label=true_label, numberOfClasses=self.numberOfClasses):
                trueClass_To_dCDF[true_label].append(d0)
            if q_instance_value <= self.ood_limit and data_validator.isKnownValidLabel(label=true_label, numberOfClasses=self.numberOfClasses):
                trueClass_To_dataset_total_q_ood[true_label] += 1
            if data_validator.isKnownValidLabel(label=true_label, numberOfClasses=self.numberOfClasses):
                trueClass_To_total_labels[true_label] += 1
        for trueLabel in range(self.numberOfClasses):
            trueClass_To_dCDF[trueLabel].sort()  # this is from lowest to highest, as needed for search sorted. It is reversed when calculating the quantile.
        if not is_training_support:
            self.trueClass_To_dCDF = trueClass_To_dCDF
            return dataset_q_values.to(main_device), trueClass_To_dataset_total_q_ood, trueClass_To_total_labels, dataset_d0_values, None
        else:
            # dCDF for training is not saved as a class attribute
            return dataset_q_values.to(main_device), trueClass_To_dataset_total_q_ood, trueClass_To_total_labels, dataset_d0_values, trueClass_To_dCDF

    def get_summary_stats_for_eval(self, eval_set_size, top_k_distances, top_k_distances_idx,
                                   eval_logits, is_training_support=False, train_trueClass_To_dCDF=None):
        # Note that is_training_support should be set to True for post-hoc analysis of training to avoid the
        # identity match. Note that dCDF will be calculated against the calibration set dCDF, unlike during training,
        # which uses the empirical CDF over training, unless the original train_trueClass_To_dCDF is provided.
        # q is determined by the original logits; if subsequent renorming causes a flip, q should be set to 0
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

        # eval_predicted_labels = np.argmax(eval_logits, axis=1)
        eval_predicted_labels = torch.argmax(eval_logits, dim=1)
        dataset_q_values = torch.zeros(eval_set_size, 1)
        dataset_d0_values = torch.zeros(eval_set_size)
        dataset_distance_quantile_per_class = torch.zeros(eval_set_size, self.numberOfClasses)
        for eval_i in range(eval_set_size):
            matched_true_labels = self.train_labels[top_k_distances_idx[eval_i]]
            matched_predicted_labels = self.train_predicted_labels[top_k_distances_idx[eval_i]]
            eval_predicted_label = eval_predicted_labels[eval_i].item()
            q_instance_value = 0
            for matching_i in range(top_k_distances_idx.shape[1]):
                if (not is_training_support) or (is_training_support and matching_i > 0):
                    if matched_true_labels[matching_i] == matched_predicted_labels[matching_i] and \
                            matched_predicted_labels[matching_i] == eval_predicted_label:
                        q_instance_value += 1
                    else:
                        break
            if is_training_support:
                d0 = top_k_distances[eval_i, 1].item()
            else:
                d0 = top_k_distances[eval_i, 0].item()
            # Note: If the prediction flips after subsequent re-normalization, q is set to 0 in eval
            dataset_q_values[eval_i, 0] = q_instance_value
            dataset_d0_values[eval_i] = d0

            d_percentiles = self.get_distance_quantiles(dataset_d0_values[eval_i].unsqueeze(0),
                                                        train_trueClass_To_dCDF=train_trueClass_To_dCDF).squeeze()
            # d_percentiles = torch.zeros(self.numberOfClasses)
            # for label in range(self.numberOfClasses):
            #     d_percentiles[label] = self.getCDFIndex(self.trueClass_To_dCDF, d0, label, reverse=True)
            dataset_distance_quantile_per_class[eval_i, :] = d_percentiles
            # is_valid = q_instance_value > self.ood_limit  # q == 0 are OOD
        return dataset_q_values, dataset_d0_values, dataset_distance_quantile_per_class

    def prediction_set_is_singleton(self, valid_bin, rescaled_output_numpy_array, logit_pred_y):
        is_singleton = False
        if valid_bin:
            prediction_set = self.constructPredictionSetFromThresholds(
                self.numberOfClasses, rescaled_output_numpy_array, self.non_odd_thresholds)
            if len(prediction_set) == 1 and logit_pred_y in prediction_set and self.non_odd_class_conditional_accuracy > 0.0:
                is_singleton = True
        return is_singleton

    def get_rescaled_output(self, q_prime, logit_pred_y, unrescaled_CDFquantiles):
        main_device = unrescaled_CDFquantiles.device
        # uses updated q and logit prediction; this, in effect, handles flips:
        soft_qbin = (self.q_rescale_offset + q_prime) ** unrescaled_CDFquantiles[0, logit_pred_y].item()
        soft_qbin = np.log(soft_qbin)
        soft_qbin = torch.tensor([soft_qbin]).unsqueeze(1)
        if main_device != torch.device('cpu'):
            soft_qbin = soft_qbin.to(torch.float32).to(main_device)
        rescaled_output = self.soft_sdm_max(self.model_rescaler(unrescaled_CDFquantiles), soft_qbin,
                                            distance_quantile_per_class=None)
        return soft_qbin, rescaled_output

    def get_rescaled_min_max(self, q_prime, logit_pred_y, unrescaled_CDFquantiles, effective_cdf_sample_size_error):
        # for min: increase non-predictions, reduce prediction
        unrescaled_CDFquantiles_min = torch.clamp(
            unrescaled_CDFquantiles + effective_cdf_sample_size_error.unsqueeze(0),
            min=0.0, max=1.0)
        unrescaled_CDFquantiles_min[0, logit_pred_y] = torch.clamp(
            unrescaled_CDFquantiles[0, logit_pred_y] - effective_cdf_sample_size_error[logit_pred_y],
            min=0.0, max=1.0)
        # for max: reduce non-predictions, increase prediction
        unrescaled_CDFquantiles_max = torch.clamp(unrescaled_CDFquantiles - effective_cdf_sample_size_error.unsqueeze(0),
                                                  min=0.0, max=1.0)
        unrescaled_CDFquantiles_max[0, logit_pred_y] = torch.clamp(
            unrescaled_CDFquantiles[0, logit_pred_y] + effective_cdf_sample_size_error[logit_pred_y],
            min=0.0, max=1.0)
        soft_qbin_min, rescaled_output_min = self.get_rescaled_output(q_prime, logit_pred_y,
                                                                      unrescaled_CDFquantiles_min)
        soft_qbin_max, rescaled_output_max = self.get_rescaled_output(q_prime, logit_pred_y,
                                                                      unrescaled_CDFquantiles_max)
        return soft_qbin_min, rescaled_output_min, soft_qbin_max, rescaled_output_max


    def _process_rescaled_output(self, soft_qbin, rescaled_output, logit_pred_y, sdm_pred_y,
                                      min_valid_qbin_for_class_conditional_accuracy_with_bounded_error=None,
                                      predicted_class_to_bin_to_output_magnitude_with_bounded_error_lower_offset_by_bin=None):

        rescaled_pred_y = torch.argmax(rescaled_output[0, :]).item()

        hard_qbin = int(soft_qbin)
        is_ood = hard_qbin <= self.ood_limit

        q_was_reduced_in_final_rescaling = False
        if rescaled_pred_y != sdm_pred_y:  # flipped, possibly for a second time
            # When displaying prediction-conditional estimate, this will be reflected in an estimate
            # <= 1/self.numberOfClasses
            q_was_reduced_in_final_rescaling = True
            # OOD status:
            is_ood = True
            hard_qbin = 0
            soft_qbin[0, 0] = 0.0
        valid_bin = (not is_ood) and (soft_qbin.item() >= self.min_valid_qbin_for_class_conditional_accuracy)
        lower_offset = 0.0
        if predicted_class_to_bin_to_output_magnitude_with_bounded_error_lower_offset_by_bin is not None and \
                logit_pred_y in predicted_class_to_bin_to_output_magnitude_with_bounded_error_lower_offset_by_bin and hard_qbin in \
                predicted_class_to_bin_to_output_magnitude_with_bounded_error_lower_offset_by_bin[logit_pred_y]:
            lower_offset = \
            predicted_class_to_bin_to_output_magnitude_with_bounded_error_lower_offset_by_bin[logit_pred_y][
                hard_qbin]
        rescaled_output[0, logit_pred_y] = torch.clamp(rescaled_output[0, logit_pred_y] - lower_offset,
                                                       min=0.0, max=1.0)
        is_singleton = self.prediction_set_is_singleton(valid_bin, rescaled_output.squeeze().cpu().numpy(),
                                                        logit_pred_y)
        if min_valid_qbin_for_class_conditional_accuracy_with_bounded_error is not None:
            # cast on right side for consistency with is_singleton
            # to avoid <class 'numpy.bool_'> with comparison with <class 'numpy.float64'>
            is_valid_index_conditional = is_singleton and bool(
                        soft_qbin.item() >= min_valid_qbin_for_class_conditional_accuracy_with_bounded_error)
        else:
            is_valid_index_conditional = is_singleton
        return rescaled_output[0, :], soft_qbin, is_valid_index_conditional, lower_offset

    def single_pass_forward(self, batch_exemplar_vectors, batch_f_positive,
                            min_valid_qbin_for_class_conditional_accuracy_with_bounded_error=None,
                            predicted_class_to_bin_to_output_magnitude_with_bounded_error_lower_offset_by_bin=None,
                            return_k_nearest_training_idx_in_prediction_metadata=1):
        main_device = batch_exemplar_vectors.device
        with torch.no_grad():
            # get summary stats and run inference all in one pass
            # we assume batch size one:
            assert batch_exemplar_vectors.shape[0] == 1
            assert batch_f_positive.shape[0] == 1
            # prediction is the argmax over Magnitude
            logit_pred_y = torch.argmax(batch_f_positive[0], dim=0).item()
            # get distances:
            top_k_distances, top_k_distances_idx = \
                self.get_top_support_distances(batch_exemplar_vectors.cpu().detach().numpy())
            # get q and distance quantiles
            batch_q, batch_d0_values, batch_distance_quantile_per_class = \
                self.get_summary_stats_for_eval(1, top_k_distances, top_k_distances_idx,
                                                batch_f_positive, is_training_support=False)
            d0 = batch_d0_values[0].item()
            # output from final layer sdm activation function (Eq. 7)
            soft_sdm_max_unrescaled_batch_output = self.soft_sdm_max(batch_f_positive, batch_q.to(main_device),
                                                                     distance_quantile_per_class=
                                                                     batch_distance_quantile_per_class.to(main_device))
            # prediction from sdm activation function. No flip is possible in the abstract, but could occur in rare
            # cases due to numerical reasons (for output close to decision boundary), so we apply a simple check
            # for consistency, reverting q to 0 (i.e., OOD) in those cases.
            sdm_pred_y = torch.argmax(soft_sdm_max_unrescaled_batch_output[0], dim=0).item()

            # q_was_reduced = False
            original_q = batch_q[0, 0].item()
            q_prime = original_q
            if logit_pred_y != sdm_pred_y:
                # q_was_reduced = True
                q_prime = 0
            # per-class quantiles of the output from the sdm activation function above (Eq. 9)
            unrescaled_CDFquantiles = torch.zeros(1, self.numberOfClasses).to(main_device)
            for label in range(self.numberOfClasses):
                unrescaled_CDFquantiles[0, label] = self.getCDFIndex(self.trueClass_To_unrescaledOutputCDF,
                                                                     soft_sdm_max_unrescaled_batch_output[0][
                                                                         label].item(),
                                                                     label,
                                                                     reverse=False,
                                                                     val_in_0to1=True)
            # rescaled q (based on the per-class quantiles) and the rescaled output from the rescaling function;
            # (Eq. 10, 11):
            soft_qbin, rescaled_output = self.get_rescaled_output(q_prime, logit_pred_y, unrescaled_CDFquantiles)
            # This accounts for flips after rescaling; applies the magnitude error correction/offset from
            # iterated learning (if applicable); and determines if the output distribution is a valid
            # index-conditional estimate based on the provided alpha' value. Note that
            # is_valid_index_conditional__centroid may not be a categorical distribution due to applying
            # the magnitude error correction/offset.
            # rescaled_prediction_conditional_distribution__centroid[logit_pred_y] += iterated_lower_offset__centroid
            # recovers the normalized distribution, if needed. (However, it typically isn't needed, since
            # rescaled_prediction_conditional_distribution__centroid and is_valid_index_conditional__centroid
            # --- and lower, upper below --- are intended as the final decision to present to human users
            # in a decision-making pipeline.) The convention is to use lower as the primary estimator (see paper).
            rescaled_prediction_conditional_distribution__centroid, \
                soft_qbin__centroid, \
                is_valid_index_conditional__centroid, iterated_lower_offset__centroid = self._process_rescaled_output(
                soft_qbin, rescaled_output, logit_pred_y, sdm_pred_y,
                min_valid_qbin_for_class_conditional_accuracy_with_bounded_error=
                min_valid_qbin_for_class_conditional_accuracy_with_bounded_error,
                predicted_class_to_bin_to_output_magnitude_with_bounded_error_lower_offset_by_bin=
                predicted_class_to_bin_to_output_magnitude_with_bounded_error_lower_offset_by_bin)

            cumulative_effective_sample_sizes, effective_cdf_sample_size_error = \
                    self.get_cumulative_effective_sample_sizes_and_errors(soft_qbin__centroid.item())  # Eq. 12 and 13
            soft_qbin_min, rescaled_output_min, soft_qbin_max, rescaled_output_max = \
                self.get_rescaled_min_max(q_prime, logit_pred_y, unrescaled_CDFquantiles,
                                          effective_cdf_sample_size_error.to(main_device))

            # same as above, but now for the min and max boundaries from DKW (which accounts for the sample size)
            rescaled_prediction_conditional_distribution__lower, \
                soft_qbin__lower, \
                is_valid_index_conditional__lower, iterated_lower_offset__lower = self._process_rescaled_output(
                soft_qbin_min, rescaled_output_min, logit_pred_y, sdm_pred_y,
                min_valid_qbin_for_class_conditional_accuracy_with_bounded_error=
                min_valid_qbin_for_class_conditional_accuracy_with_bounded_error,
                predicted_class_to_bin_to_output_magnitude_with_bounded_error_lower_offset_by_bin=
                predicted_class_to_bin_to_output_magnitude_with_bounded_error_lower_offset_by_bin)  # Eq. 28

            rescaled_prediction_conditional_distribution__upper, \
                soft_qbin__upper, \
                is_valid_index_conditional__upper, iterated_lower_offset__upper = self._process_rescaled_output(
                soft_qbin_max, rescaled_output_max, logit_pred_y, sdm_pred_y,
                min_valid_qbin_for_class_conditional_accuracy_with_bounded_error=
                min_valid_qbin_for_class_conditional_accuracy_with_bounded_error,
                predicted_class_to_bin_to_output_magnitude_with_bounded_error_lower_offset_by_bin=
                predicted_class_to_bin_to_output_magnitude_with_bounded_error_lower_offset_by_bin)

            # lower, estimate centroid, upper -- these can flip when predictions flip when rescaling,
            # so remember to re-order and convey that
            # information to end-users, as applicable. We use "prediction" as the reference prediction. That is,
            # flips in rescaling of the other estimates are then conveyed as probabilities <= 1/|Y| (e.g., prob. <= 0.5
            # for binary classification). (These types of flips are very rare for valid index conditional
            # estimates, and primarily occur when the prediction-conditional distribution estimate of the predicted
            # class is near 1/|Y|.)

            # 2025-07-23: Adding additional values here for convenience and consistency,
            # rather than calculating them downstream:
            hard_qbin_lower = int(soft_qbin__lower[0].item())
            is_ood_lower = hard_qbin_lower <= self.ood_limit
            return {
                    # raw (i.e., un-rescaled) Similarity value: q:
                    "original_q": original_q,
                    # raw Distance value: d_nearest:
                    "d0": d0,
                    # raw Magnitude value (un-normalized logits):
                    "f": batch_f_positive[0, :],
                    # this is the predicted class, which may differ from the argmax of the rescaled distributions:
                    # \hat{y}:
                    "prediction": logit_pred_y,
                    # min (among Distance quantiles across classes): d:
                    "distance_quantiles": batch_distance_quantile_per_class.squeeze(),
                    # centroid -- prediction-conditional distribution:
                    "rescaled_prediction_conditional_distribution__centroid":
                        rescaled_prediction_conditional_distribution__centroid,
                    # centroid -- soft rescaled Similarity value
                    #    Note: int(soft_qbin__centroid) produces the hard version
                    "soft_qbin__centroid": soft_qbin__centroid,
                    # centroid -- bool: whether the prediction is valid index conditional
                    "is_valid_index_conditional__centroid": is_valid_index_conditional__centroid,
                    # centroid -- float: this is the cauchy distributed offset over data/learning iterations. This has
                    #   already been subtracted from rescaled_prediction_conditional_distribution__centroid[prediction],
                    #   which means the distribution will not (necessarily) be normalized.
                    #   (rescaled_prediction_conditional_distribution__centroid[prediction] +
                    #   iterated_lower_offset__centroid) recovers the normalized distribution, if needed (and
                    #   similarly for lower, upper below).
                    "iterated_lower_offset__centroid": iterated_lower_offset__centroid,
                    # lower and upper are analogous to centroid, but use the DKW offset of the CDFs
                    # lower: i.e., Eq. 28 for max(0, p(\hat{y})_lower-m^{\hat{y}}_hardqbin):
                    "rescaled_prediction_conditional_distribution__lower":
                        rescaled_prediction_conditional_distribution__lower,
                    "soft_qbin__lower": soft_qbin__lower,
                    "is_valid_index_conditional__lower": is_valid_index_conditional__lower,
                    "iterated_lower_offset__lower": iterated_lower_offset__lower,
                    # upper:
                    "rescaled_prediction_conditional_distribution__upper":
                        rescaled_prediction_conditional_distribution__upper,
                    "soft_qbin__upper": soft_qbin__upper,
                    "is_valid_index_conditional__upper": is_valid_index_conditional__upper,
                    "iterated_lower_offset__upper": iterated_lower_offset__upper,
                    # effective sample size across classes: Eq. 12:
                    "cumulative_effective_sample_sizes": cumulative_effective_sample_sizes,
                    # Added 2025-07-23:
                    # hard_qbin_lower is an int
                    "hard_qbin_lower": hard_qbin_lower,
                    # is_ood_lower is Bool. Note that when an instance is not
                    #    is_valid_index_conditional__lower, there are two possibilities: It is or isn't
                    #    is_ood_lower. That is, not all non-index-conditional instances are OOD.
                    "is_ood_lower": is_ood_lower,
                    "top_k_distances_idx": top_k_distances_idx[0, 0:max(1, return_k_nearest_training_idx_in_prediction_metadata)]
                    }

    def normalize_embeddings(self, embeddings):
        # (optional) mean centering of the input to the 1-D CNN of the sdm activation:
        return (embeddings - self.training_embedding_summary_stats[constants.STORAGE_KEY_SUMMARY_STATS_EMBEDDINGS_training_embedding_mean]) / \
            self.training_embedding_summary_stats[constants.STORAGE_KEY_SUMMARY_STATS_EMBEDDINGS_training_embedding_std]

    def forward(self, input, batch_q=None, batch_f_positive=None, batch_distance_quantile_per_class=None,
                forward_type=constants.FORWARD_TYPE_SENTENCE_LEVEL_PREDICTION, train=False, normalize_embeddings=True,
                debug=False,
                min_valid_qbin_for_class_conditional_accuracy_with_bounded_error=None,
                predicted_class_to_bin_to_output_magnitude_with_bounded_error_lower_offset_by_bin=None,
                return_k_nearest_training_idx_in_prediction_metadata=1):
        # The point-estimate prediction is always determined by batch_f_positive.

        if forward_type == constants.FORWARD_TYPE_TRAIN_RESCALER:
            assert train
            assert batch_q is not None
            # input corresponds to calibration_unrescaled_CDFquantiles
            return self.soft_sdm_max(self.model_rescaler(input), batch_q,
                                                                     distance_quantile_per_class=None,
                                                                     log=train, change_of_base=True)
        elif forward_type == constants.FORWARD_TYPE_RESCALE_CACHED_CALIBRATION:
            assert not train
            assert batch_q is not None
            # input corresponds to calibration_unrescaled_CDFquantiles
            return self.soft_sdm_max(self.model_rescaler(input), batch_q,
                                                                     distance_quantile_per_class=None,
                                                                     log=train)

        if forward_type == constants.FORWARD_TYPE_GENAI_WITH_ROUTER_TOKEN_LEVEL_PREDICTION:  # only need to calculate when training
            batch_f_genai = torch.cat([self.fc_negative(input[:, -self.embedding_size:]),
                                       self.fc_positive(input[:, -self.embedding_size:])], dim=1)
            # batch_distance_quantile_per_class is the min distance per instance, so we expand to gen ai vocab
            batch_f_genai = self.soft_sdm_max(batch_f_genai,
                                              batch_q,
                                              distance_quantile_per_class=
                                              batch_distance_quantile_per_class[:, 0, None].expand(
                                                  batch_f_genai.shape[0], self.gen_ai_vocab*2) if
                                              batch_distance_quantile_per_class is not None else None,
                                              log=train, change_of_base=True)
            if train:
                with torch.no_grad():
                    # need original reference distribution for regularization
                    batch_f_original = self.soft_sdm_max(torch.cat([self.fc_original(input[:, -self.embedding_size:]),
                                                                    self.fc_original(input[:, -self.embedding_size:])],
                                                                   dim=1),
                                                         batch_q,
                                                         distance_quantile_per_class=
                                                         batch_distance_quantile_per_class[:, 0, None].expand(
                                                             batch_f_genai.shape[0], self.gen_ai_vocab * 2) if
                                                         batch_distance_quantile_per_class is not None else None,
                                                         log=train, change_of_base=True)

                return batch_f_genai, batch_f_original
            else:
                assert False
        if batch_f_positive is None or forward_type == constants.FORWARD_TYPE_GENERATE_EXEMPLAR_VECTORS:
            # input corresponds to:
            # [composition attributes (optional)] :: [Cumulative average LLM embeddings (optional)] :: [LLM embedding]
            batch_exemplar_vectors = input.unsqueeze(1)
            # global norm
            if normalize_embeddings:
                with torch.no_grad():
                    batch_exemplar_vectors = \
                        self.normalize_embeddings(batch_exemplar_vectors)
            batch_exemplar_vectors = self.conv(batch_exemplar_vectors).squeeze(2)
            batch_f_positive = self.fc(batch_exemplar_vectors)

            assert len(batch_exemplar_vectors.shape) != 1
        if len(batch_f_positive.shape) == 1:
            batch_f_positive = batch_f_positive.unsqueeze(0)
        if forward_type == constants.FORWARD_TYPE_SINGLE_PASS_TEST:
            return self.single_pass_forward(batch_exemplar_vectors, batch_f_positive,
                                            min_valid_qbin_for_class_conditional_accuracy_with_bounded_error=
                                            min_valid_qbin_for_class_conditional_accuracy_with_bounded_error,
                                            predicted_class_to_bin_to_output_magnitude_with_bounded_error_lower_offset_by_bin=
                                            predicted_class_to_bin_to_output_magnitude_with_bounded_error_lower_offset_by_bin,
                                            return_k_nearest_training_idx_in_prediction_metadata=
                                            return_k_nearest_training_idx_in_prediction_metadata)
        elif forward_type == constants.FORWARD_TYPE_SINGLE_PASS_TEST_WITH_EXEMPLAR:
            prediction_meta_data = self.single_pass_forward(batch_exemplar_vectors, batch_f_positive,
                                                            min_valid_qbin_for_class_conditional_accuracy_with_bounded_error=
                                                            min_valid_qbin_for_class_conditional_accuracy_with_bounded_error,
                                                            predicted_class_to_bin_to_output_magnitude_with_bounded_error_lower_offset_by_bin=
                                                            predicted_class_to_bin_to_output_magnitude_with_bounded_error_lower_offset_by_bin,
                                                            return_k_nearest_training_idx_in_prediction_metadata=
                                                            return_k_nearest_training_idx_in_prediction_metadata)
            prediction_meta_data["exemplar_vector"] = batch_exemplar_vectors.cpu().detach().numpy()
            return prediction_meta_data

        assert batch_q is not None
        soft_sdm_max_unrescaled_batch_output = self.soft_sdm_max(batch_f_positive, batch_q,
                                                  distance_quantile_per_class=batch_distance_quantile_per_class,
                                                  log=train, change_of_base=True)
        if forward_type == constants.FORWARD_TYPE_SENTENCE_LEVEL_PREDICTION:
            return batch_f_positive, soft_sdm_max_unrescaled_batch_output
        elif forward_type == constants.FORWARD_TYPE_GENERATE_EXEMPLAR_VECTORS:
            return batch_f_positive, soft_sdm_max_unrescaled_batch_output, batch_exemplar_vectors

    def export_properties_to_dict(self):
        json_dict = {constants.STORAGE_KEY_version: constants.ProgramIdentifiers_version,
                     constants.STORAGE_KEY_uncertaintyModelUUID: self.uncertaintyModelUUID,
                     constants.STORAGE_KEY_non_odd_class_conditional_accuracy: self.non_odd_class_conditional_accuracy,
                     constants.STORAGE_KEY_alpha: self.alpha,
                     constants.STORAGE_KEY_cdfThresholdTolerance: self.cdfThresholdTolerance,
                     constants.STORAGE_KEY_maxQAvailableFromIndexer: self.maxQAvailableFromIndexer,
                     constants.STORAGE_KEY_numberOfClasses: self.numberOfClasses,
                     constants.STORAGE_KEY_q_rescale_offset: self.q_rescale_offset,
                     constants.STORAGE_KEY_ood_limit: self.ood_limit,
                     constants.STORAGE_KEY_exemplar_vector_dimension: self.exemplar_vector_dimension,
                     constants.STORAGE_KEY_embedding_size: self.embedding_size,
                     constants.STORAGE_KEY_calibration_training_stage: self.calibration_training_stage,
                     constants.STORAGE_KEY_calibration_is_ood_indicators: self.calibration_is_ood_indicators,
                     constants.STORAGE_KEY_min_valid_qbin_for_class_conditional_accuracy: self.min_valid_qbin_for_class_conditional_accuracy,
                     constants.STORAGE_KEY_SUMMARY_STATS_EMBEDDINGS_training_embedding_summary_stats: self.training_embedding_summary_stats,
                     constants.STORAGE_KEY_is_gen_ai: self.is_gen_ai,
                     constants.STORAGE_KEY_gen_ai_vocab: self.gen_ai_vocab,
                     constants.STORAGE_KEY_global_embedding_size: self.global_embedding_size,
                     constants.STORAGE_KEY_composition_attributes_size: self.composition_attributes_size,
                     constants.STORAGE_KEY_top_logits_k: self.top_logits_k
                     }

        json_dict[constants.STORAGE_KEY_non_odd_thresholds] = list(self.non_odd_thresholds)

        trueClass_To_dCDF_json_flat = {}
        for label in self.trueClass_To_dCDF.keys():
            trueClass_To_dCDF_json_flat[label] = self.trueClass_To_dCDF[label]
        json_dict[constants.STORAGE_KEY_trueClass_To_dCDF] = trueClass_To_dCDF_json_flat

        train_trueClass_To_dCDF_json_flat = {}
        for label in self.train_trueClass_To_dCDF.keys():
            train_trueClass_To_dCDF_json_flat[label] = self.train_trueClass_To_dCDF[label]
        json_dict[constants.STORAGE_KEY_train_trueClass_To_dCDF] = train_trueClass_To_dCDF_json_flat

        trueClass_To_unrescaledOutputCDF_json_flat = {}
        for label in self.trueClass_To_unrescaledOutputCDF.keys():
            trueClass_To_unrescaledOutputCDF_json_flat[label] = self.trueClass_To_unrescaledOutputCDF[label]
        json_dict[constants.STORAGE_KEY_trueClass_To_unrescaledOutputCDF] = trueClass_To_unrescaledOutputCDF_json_flat

        trueClass_To_qCumulativeSampleSizeArray_json_flat = {}
        for label in self.trueClass_To_qCumulativeSampleSizeArray.keys():
            trueClass_To_qCumulativeSampleSizeArray_json_flat[label] = self.trueClass_To_qCumulativeSampleSizeArray[label]
        json_dict[constants.STORAGE_KEY_trueClass_To_qCumulativeSampleSizeArray] = trueClass_To_qCumulativeSampleSizeArray_json_flat
        return json_dict

    def import_properties_from_dict(self, json_dict, load_for_inference=False):
        # When loading from disk, this must be called after class init before calibrating new data points.
        # Note that in JSON, int dictionary keys become strings

        trueClass_To_dCDF_json_flat = json_dict[constants.STORAGE_KEY_trueClass_To_dCDF]
        for trueClass in range(self.numberOfClasses):
            trueClass_str = str(trueClass)
            if trueClass_str in trueClass_To_dCDF_json_flat:
                self.trueClass_To_dCDF[trueClass] = trueClass_To_dCDF_json_flat[trueClass_str]
            else:
                self.trueClass_To_dCDF[trueClass] = []

        trueClass_To_unrescaledOutputCDF_json_flat = \
            json_dict[constants.STORAGE_KEY_trueClass_To_unrescaledOutputCDF]
        for trueClass in range(self.numberOfClasses):
            trueClass_str = str(trueClass)
            if trueClass_str in trueClass_To_unrescaledOutputCDF_json_flat:
                self.trueClass_To_unrescaledOutputCDF[trueClass] = \
                    trueClass_To_unrescaledOutputCDF_json_flat[trueClass_str]
            else:
                self.trueClass_To_unrescaledOutputCDF[trueClass] = []

        trueClass_To_qCumulativeSampleSizeArray_json_flat = \
            json_dict[constants.STORAGE_KEY_trueClass_To_qCumulativeSampleSizeArray]
        for trueClass in range(self.numberOfClasses):
            trueClass_str = str(trueClass)
            if trueClass_str in trueClass_To_qCumulativeSampleSizeArray_json_flat:
                self.trueClass_To_qCumulativeSampleSizeArray[trueClass] = \
                    trueClass_To_qCumulativeSampleSizeArray_json_flat[trueClass_str]
            else:
                self.trueClass_To_qCumulativeSampleSizeArray[trueClass] = []

        if self.is_gen_ai and not load_for_inference:
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
# - Always use true class for the main CDF structures when collecting the original statistics over the calibration set;
# - Remember to properly address prediction flips (which can also happen when the model goes to parity)
# - Don't forget to sort cdf structures;
# - Properly handle the boundaries of determining the quantiles (e.g., when the output is saturated;
#    see getCDFIndex(), which covers the edge cases)
# TODO: Currently we are inconsistent with variable casing, as a consequence of simplifying conversions
#  between the Swift and Python codebases. (Swift and Python use different conventions.)
