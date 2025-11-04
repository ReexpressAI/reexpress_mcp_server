# Copyright Reexpress AI, Inc. All rights reserved.

import utils_model

import torch
import logging
import sys


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def set_model_rescaled_similarity_vectorized(model, calibration_cached_f_outputs,
                                             calibration_dataset_q_values, calibration_sdm_outputs):
    """
    Vectorized version of setting model rescaled similarity values for the calibration set. For eval sets, use
    get_rescaled_similarity_for_eval_batch().

    These class properties are used in the step to determine the High Reliability region.
    These are saved as class properties to enable fast recalibration, if needed, with different alpha values.
    """
    # Verify predictions match
    cached_predictions = torch.argmax(calibration_cached_f_outputs, dim=1)
    assert torch.all(cached_predictions == model.calibration_predicted_labels), \
        "Error: There is an unexpected mismatch between the model's saved calibration predictions and " \
        "the argmax logits here."

    # Extract SDM outputs for predicted classes using advanced indexing
    # Create indices for gathering the correct SDM output values
    batch_indices = torch.arange(len(model.calibration_predicted_labels))
    sdm_outputs_for_predicted = calibration_sdm_outputs[batch_indices, model.calibration_predicted_labels].to(model.device)

    # Ensure q_values is the right shape - squeeze if needed
    if calibration_dataset_q_values.dim() > 1:
        q_values_squeezed = calibration_dataset_q_values.squeeze(-1)
    else:
        q_values_squeezed = calibration_dataset_q_values

    # Vectorized computation of rescaled similarities
    rescaled_similarities = model.get_rescaled_similarity_vectorized(
        q=q_values_squeezed,
        sdm_output_for_predicted_class=sdm_outputs_for_predicted
    )

    # Reshape to match expected format [N, 1]
    model.calibration_rescaled_similarity_values = rescaled_similarities.unsqueeze(1)

    # Store the SDM outputs as before
    model.calibration_sdm_outputs = calibration_sdm_outputs


def calibrate_to_determine_high_reliability_region(options, model_dir=None):
    assert model_dir is not None
    # reload best epoch
    model = utils_model.load_model_torch(model_dir, torch.device("cpu"))
    if model.alpha != options.alpha:
        print(f">>Updating alpha from {model.alpha} (saved with the model) to {options.alpha} based on the "
              f"provided input arguments. However, note that the global statistics (across iterations) will "
              f"not be updated.<<")
        model.alpha = options.alpha
    model.set_high_reliability_region_thresholds(calibration_sdm_outputs=model.calibration_sdm_outputs,
                                                 calibration_rescaled_similarity_values=
                                                 model.calibration_rescaled_similarity_values,
                                                 true_labels=model.calibration_labels)

    utils_model.save_model(model, model_dir)
    logger.info(f"Model saved to {model_dir} with training and calibration complete. Ready for testing.")
    return model.min_rescaled_similarity_to_determine_high_reliability_region
