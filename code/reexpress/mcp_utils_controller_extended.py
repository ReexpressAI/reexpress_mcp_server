# Copyright Reexpress AI, Inc. All rights reserved.

# Additional functions when running outside of typical tool MCP calls.
import uuid
from pathlib import Path

import constants
import utils_model
import data_validator
from mcp_utils_controller import MCPServerStateController, AdaptationError


class ExtendedMCPServerStateController(MCPServerStateController):
    """
    Extends MCPServerStateController with convenience methods for use
    outside of typical tool calls.
    """

    def save_current_reexpression_to_file(self, document_id: str, label: int, output_filename: str) -> str:
        # Convenience function when calling the Server outside of typical tool calls. This saves the embedding and
        # generative AI output to a specified file (output_filename) with a given document_id and label.
        # For simplicity, this does not run an analysis of the the SDM estimator output, and only the output from
        # get_reexpress_view (as a convenience for feeding back into a test-time search graph).
        # To save the full SDM estimator output and run the standard
        # eval scripts, run reexpress.py using the saved file as input.
        message = "There is no existing reexpression in the cache."
        if self.model_list is None or self.current_reexpression is None:
            return message
        try:
            if label not in [0, 1, data_validator.oodLabel]:
                raise AdaptationError(f"The provided label is not in [0, 1, {data_validator.oodLabel}].", "LABEL_ERROR")
            running_updates_file_path = Path(output_filename)
            # if constants.ADMIN_LABEL_MODE:
            #     document_id = f"added_{str(uuid.uuid4())}"
            # else:
            #     document_id = f"user_added_{str(uuid.uuid4())}"
            # prediction_meta_data_across_models = \
            #     self.current_reexpression.get(
            #         "prediction_meta_data_dict", {}).get(
            #         "prediction_meta_data_across_models", {})
            # Currently only the model at index 0:
            # model = self.model_list[0]
            # prediction_meta_data = prediction_meta_data_across_models[0] if len(
            #     prediction_meta_data_across_models) > 0 else {}
            # exemplar_vector = prediction_meta_data["exemplar_vector"]

            reexpress_view_output = self.get_reexpress_view()

            json_for_archive = {}
            reexpression_input = self.current_reexpression["reexpression_input"]
            embedding = [float(x) for x in reexpression_input.squeeze().detach().cpu().tolist()]
            json_for_archive[constants.REEXPRESS_ID_KEY] = document_id
            json_for_archive[constants.REEXPRESS_DOCUMENT_KEY] = ""
            json_for_archive[constants.REEXPRESS_LABEL_KEY] = label
            json_for_archive[constants.REEXPRESS_EMBEDDING_KEY] = embedding
            json_for_archive[constants.REEXPRESS_QUESTION_KEY] = \
                self.current_reexpression[constants.REEXPRESS_QUESTION_KEY]
            json_for_archive[constants.REEXPRESS_AI_RESPONSE_KEY] = \
                self.current_reexpression[constants.REEXPRESS_AI_RESPONSE_KEY]

            json_for_archive[constants.REEXPRESS_MODEL1_CLASSIFICATION] = \
                self.current_reexpression[constants.REEXPRESS_MODEL1_CLASSIFICATION]
            json_for_archive[constants.REEXPRESS_MODEL2_CLASSIFICATION] = \
                self.current_reexpression[constants.REEXPRESS_MODEL2_CLASSIFICATION]

            json_for_archive[constants.REEXPRESS_MODEL1_EXPLANATION] = \
                self.current_reexpression[constants.REEXPRESS_MODEL1_EXPLANATION]
            json_for_archive[constants.REEXPRESS_MODEL2_EXPLANATION] = \
                self.current_reexpression[constants.REEXPRESS_MODEL2_EXPLANATION]

            json_for_archive[constants.REEXPRESS_AGREEMENT_MODEL_CLASSIFICATION] = \
                self.current_reexpression[constants.REEXPRESS_AGREEMENT_MODEL_CLASSIFICATION]
            json_for_archive[constants.REEXPRESS_MODEL1_TOPIC_SUMMARY] = \
                self.current_reexpression[constants.REEXPRESS_MODEL1_TOPIC_SUMMARY]

            json_for_archive[constants.REEXPRESS_ATTACHED_FILE_NAMES] = \
                self.current_reexpression[constants.REEXPRESS_ATTACHED_FILE_NAMES]
            json_for_archive[constants.REEXPRESS_ATTACHED_DOCUMENTS_CONTENT] = \
                self.current_reexpression[constants.REEXPRESS_ATTACHED_DOCUMENTS_CONTENT]
            json_for_archive[constants.REEXPRESS_ATTACHED_DOCUMENT_NOTE_KEY] = \
                self.current_reexpression[constants.REEXPRESS_ATTACHED_DOCUMENT_NOTE_KEY]

            json_for_archive[constants.REEXPRESS_MODEL1_NAME_KEY] = \
                constants.MCP_SERVER_MODEL1_NAME
            json_for_archive[constants.REEXPRESS_MODEL2_NAME_KEY] = \
                constants.MCP_SERVER_MODEL2_NAME
            json_for_archive[constants.REEXPRESS_AGREEMENT_MODEL_NAME_KEY] = \
                constants.MCP_SERVER_API_EMBEDDING_MODEL_NAME
            json_for_archive[constants.REEXPRESS_MCP_SERVER_VERSION_KEY] = \
                constants.REEXPRESS_MCP_SERVER_VERSION
            # for research purposes, we also save the verbalized uncertainty to the JSON archive:
            json_for_archive[constants.REEXPRESS_MODEL1_CONFIDENCE] = \
                self.current_reexpression[constants.REEXPRESS_MODEL1_CONFIDENCE]
            json_for_archive[constants.REEXPRESS_MODEL2_CONFIDENCE] = \
                self.current_reexpression[constants.REEXPRESS_MODEL2_CONFIDENCE]

            json_for_archive[constants.REEXPRESS_VIEW_OUTPUT_KEY] = reexpress_view_output

            json_for_archive[constants.REEXPRESS_SUBMITTED_TIME_KEY] = \
                self.current_reexpression[constants.REEXPRESS_SUBMITTED_TIME_KEY]

            utils_model.save_by_appending_json_lines(str(running_updates_file_path.as_posix()), [json_for_archive])
            # add to document database
            add_to_support_db_message = ""

            self.current_reexpression = None

            if label == 0:
                string_label = f"{constants.MCP_SERVER_NOT_VERIFIED_CLASS_LABEL} (Class 0)"
            elif label == 1:
                string_label = f"{constants.MCP_SERVER_VERIFIED_CLASS_LABEL} (Class 1)"
            elif label == data_validator.oodLabel:
                string_label = f"{data_validator.getDefaultLabelName(label=label, abbreviated=False)}"
            else:
                raise AdaptationError(f"The provided label is not in [0, 1, {data_validator.oodLabel}].",
                                      "LABEL_ERROR")

            message = f"Successfully added document id {document_id} to " \
                      f"{str(running_updates_file_path.as_posix())} with label: " \
                      f"{string_label}.{add_to_support_db_message}"
            return message
        except AdaptationError as e:
            self.current_reexpression = None  # clear the cache to avoid inconsistencies
            return f"ERROR: Unable to save. {e} The cache has been cleared."
        except:
            self.current_reexpression = None  # clear the cache to avoid inconsistencies
            return message
