# Copyright Reexpress AI, Inc. All rights reserved.

from sdm_model import SimilarityDistanceMagnitudeCalibrator
import utils_pretraining_initialization

import logging
import sys


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def train(options, train_embeddings=None, calibration_embeddings=None,
          train_labels=None, calibration_labels=None,
          model_params=None,
          main_device=None, model_dir=None, model=None):
    import baseline_utils_model

    if model is None:
        print("Initializing model")
        model = SimilarityDistanceMagnitudeCalibrator(**model_params).to(main_device)

    # For the baseline adaptor, we use the 'pretrain' routine
    model, min_held_out_balanced_loss, min_held_out_balanced_loss_epoch = \
        utils_pretraining_initialization.pretrain(options, model=model, model_dir=model_dir,
                                                  held_out_embeddings=calibration_embeddings,
                                                  held_out_labels=calibration_labels,
                                                  train_embeddings=train_embeddings,
                                                  train_labels=train_labels,
                                                  pretraining_learning_rate=options.learning_rate,
                                                  return_min_held_out_balanced_loss=True,
                                                  main_device=main_device, use_main_device=True
                                                  )
    model = model.to(main_device)
    baseline_utils_model.save_baseline_model(model, model_dir)
    logger.info(f"Model saved to iteration model_dir")
    return min_held_out_balanced_loss, min_held_out_balanced_loss_epoch