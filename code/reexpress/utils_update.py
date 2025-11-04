# Copyright Reexpress AI, Inc. All rights reserved.

# Update the support set

import torch
import numpy as np

import constants
import utils_model
import utils_preprocess


def batch_support_update(options, main_device):
    if options.skip_updates_already_in_support:
        # In this case, we also need to load the calibration set document id's.
        model = utils_model.load_model_torch(options.model_dir, main_device, load_for_inference=False)
    else:
        model = utils_model.load_model_torch(options.model_dir, main_device, load_for_inference=True)

    print(f"Current support set cardinality: {model.support_index.ntotal}")
    test_meta_data, _ = \
        utils_preprocess.get_metadata_lines(options, options.input_eval_set_file,
                                            reduce=False,
                                            use_embeddings=options.use_embeddings,
                                            concat_embeddings_to_attributes=options.concat_embeddings_to_attributes,
                                            calculate_summary_stats=False, is_training=False)
    test_embeddings = test_meta_data["embeddings"]
    test_labels = torch.tensor(test_meta_data["labels"])
    document_ids = test_meta_data["uuids"]

    assert test_embeddings.shape[0] == test_labels.shape[0]
    print(f"test_embeddings.shape: {test_embeddings.shape}")
    count_already_present_documents = 0
    for test_embedding, test_label, document_id in zip(test_embeddings, test_labels, document_ids):
        if options.skip_updates_already_in_support:
            if document_id in model.train_uuids or document_id in model.calibration_uuids:
                count_already_present_documents += 1
                continue
        true_test_label = test_label.item()
        prediction_meta_data = \
            model(test_embedding.unsqueeze(0).to(main_device),
                  forward_type=constants.FORWARD_TYPE_SINGLE_PASS_TEST_WITH_EXEMPLAR)

        exemplar_vector = prediction_meta_data["exemplar_vector"].detach().cpu().numpy()
        model.add_to_support(label=true_test_label, predicted_label=prediction_meta_data["prediction"],
                             document_id=document_id, exemplar_vector=exemplar_vector)

    support_set_cardinality = model.support_index.ntotal
    assert model.train_labels.shape[0] == support_set_cardinality
    assert model.train_predicted_labels.shape[0] == support_set_cardinality
    assert len(model.train_uuids) == support_set_cardinality
    utils_model.save_support_set_updates(model, options.model_dir)
    print(f"Updated support set cardinality: {model.support_index.ntotal}")
    print(f"Note that this does not update the distance quantiles, nor the thresholds on the HR region. "
          f"This is intended for small, local changes. For more substantive changes, retrain the model.")
    if options.skip_updates_already_in_support:
        print(f"Count of skipped document id's already in the support set or calibration set: "
              f"{count_already_present_documents}")

