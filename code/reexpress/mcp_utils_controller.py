# Copyright Reexpress AI, Inc. All rights reserved.

# Primary controller for MCP server

import os
from pathlib import Path

import torch

import asyncio
import uuid
from datetime import datetime

import constants
import utils_model
import mcp_utils_data_format
import mcp_utils_test
import mcp_utils_llm_api
import mcp_utils_file_access_manager
import mcp_utils_tool_limits_manager
import data_validator
import utils_visualization
import mcp_utils_sqlite_document_db_controller


class AdaptationError(Exception):
    def __init__(self, message="An adaptation error occurred", error_code=None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

    def __str__(self):
        if self.error_code is not None:
            return f"{self.error_code}: {self.message}"
        return self.message


class MCPServerStateController:
    """
    Primary controller for the Model-Context-Protocol (MCP) server.
        ExternalFileController() handles external files sent to the verification LLMs
        ToolCallLimitController() handles soft and hard tool-call limits
        mcp_utils_llm_api.py handles LLM API calls and transforms of the output
        The environment variables REEXPRESS_MCP_SERVER_REPO_DIR and REEXPRESS_MCP_MODEL_DIR, along with the LLM api
            variables, must be present. See the documentation for details.
    """
    def __init__(self):
        REEXPRESS_MCP_SERVER_REPO_DIR = os.getenv("REEXPRESS_MCP_SERVER_REPO_DIR")
        self.MODEL_DIR = os.getenv("REEXPRESS_MCP_MODEL_DIR")
        try:
            self.CREATE_HTML_VISUALIZATION = int(os.getenv("REEXPRESS_MCP_SAVE_OUTPUT")) == 1
            # Note the file must exist (content can be empty)
            self.HTML_VISUALIZATION_FILE = str(
                Path(self.MODEL_DIR, "visualize", "current_reexpression.html").as_posix())
        except:
            self.CREATE_HTML_VISUALIZATION = False
            self.HTML_VISUALIZATION_FILE = None
        # For provenance of any added instances (the file must exist---content can be empty):
        self.DATA_UPDATE_FILE = str(Path(self.MODEL_DIR, "adaptation", "running_updates.jsonl").as_posix())
        # load model
        self.main_device = torch.device("cpu")
        self.model = utils_model.load_model_torch(self.MODEL_DIR, torch.device("cpu"), load_for_inference=True)
        self.global_uncertainty_statistics = utils_model.load_global_uncertainty_statistics_from_disk(self.MODEL_DIR)
        # Data Access Manager: Controls file access and content sent directly to the verification LLMs
        self.mcp_file_access_manager_object = \
            mcp_utils_file_access_manager.ExternalFileController(mcp_server_dir=REEXPRESS_MCP_SERVER_REPO_DIR)
        self.mcp_utils_tool_limits_manager_object = \
            mcp_utils_tool_limits_manager.ToolCallLimitController(mcp_server_dir=REEXPRESS_MCP_SERVER_REPO_DIR)
        self.current_reexpression = None
        try:
            self.reexpress_mcp_server_support_documents_file = str(
                Path(self.MODEL_DIR, "reexpress_mcp_server_db", "reexpress_mcp_server_support_documents.db").as_posix())
            self.support_db = \
                mcp_utils_sqlite_document_db_controller.DocumentDatabase(
                    self.reexpress_mcp_server_support_documents_file)
        except:
            self.support_db = None

    def controller_directory_set(self, directory: str) -> str:
        return self.mcp_file_access_manager_object.add_environment_parent_directory(parent_directory=directory)

    def controller_file_set(self, filename: str) -> str:
        return self.mcp_file_access_manager_object.add_file(filename_with_path=filename)

    def controller_file_clear(self) -> str:
        return self.mcp_file_access_manager_object.remove_all_file_access()

    def controller_get_tool_availability(self) -> (bool, str):
        return self.mcp_utils_tool_limits_manager_object.get_tool_availability()

    def controller_update_counters(self):
        self.mcp_utils_tool_limits_manager_object.update_counters()

    def controller_reset_sequential_limit_counter(self) -> str:
        return self.mcp_utils_tool_limits_manager_object.reset_sequential_limit_counter()

    async def _run_tasks_with_taskgroup(self, task_configs):
        tasks = []
        async with asyncio.TaskGroup() as tg:
            for func, args in task_configs:
                task = tg.create_task(asyncio.to_thread(func, args))
                tasks.append(task)
        return [task.result() for task in tasks]

    async def get_reexpressed_verification(self, user_question: str, ai_response: str) -> str:
        content_xml, available_file_names = self.mcp_file_access_manager_object.get_current_external_file_content()
        if len(content_xml) > 0:
            attached_documents = f" {content_xml} "
        else:
            attached_documents = ""
            available_file_names = []
        attached_document_note = ""
        if len(attached_documents) > 0:
            attached_document_note = "(Note: I have included additional documents relevant to this discussion within the <attached_file></attached_file> XML tags.) "
        previous_query_and_response_to_verify_string = \
            f"<question> {attached_documents}{user_question} </question> <ai_response> {ai_response} {attached_document_note}</ai_response>"
        # currently identical:
        previous_query_and_response_to_verify_string_reasoning = \
            f"<question> {attached_documents}{user_question} </question> <ai_response> {ai_response} {attached_document_note}</ai_response>"
        # currently identical:
        previous_query_and_response_to_verify_string_gemini = \
            f"<question> {attached_documents}{user_question} </question> <ai_response> {ai_response} {attached_document_note}</ai_response>"
        task_configs = [(mcp_utils_llm_api.get_document_attributes,
                         previous_query_and_response_to_verify_string),
                        (mcp_utils_llm_api.get_document_attributes_from_reasoning,
                         previous_query_and_response_to_verify_string_reasoning),
                        (mcp_utils_llm_api.get_document_attributes_from_gemini_reasoning,
                         previous_query_and_response_to_verify_string_gemini)
                        ]
        try:
            results = await self._run_tasks_with_taskgroup(task_configs)
            log_prob_model_verification_dict, \
                reasoning_model_verification_dict, \
                gemini_model_verification_dict = results
            llm_api_error = log_prob_model_verification_dict[constants.REEXPRESS_ATTRIBUTES_KEY] is None
        except:
            llm_api_error = True
        # results = await asyncio.gather(
        #     asyncio.to_thread(mcp_utils_llm_api.get_document_attributes, previous_query_and_response_to_verify_string),
        #     asyncio.to_thread(mcp_utils_llm_api.get_document_attributes_from_reasoning, previous_query_and_response_to_verify_string)
        # )
        # log_prob_model_verification_dict, reasoning_model_verification_dict = results
        # llm_api_error = log_prob_model_verification_dict[constants.REEXPRESS_ATTRIBUTES_KEY] is None
        formatted_output_string = ""
        partial_reexpression = {}

        if not llm_api_error:
            # get embedding over model explanations
            log_prob_model_classification, log_prob_model_explanation, \
                reasoning_model_classification, reasoning_model_explanation, \
                gemini_model_classification, gemini_model_explanation = \
                mcp_utils_llm_api.get_model_explanations(log_prob_model_verification_dict,
                                                         reasoning_model_verification_dict,
                                                         gemini_model_verification_dict)
            agreement_model_embedding, agreement_model_classification = \
                mcp_utils_llm_api.llm_api_controller(log_prob_model_explanation=log_prob_model_explanation,
                                                     reasoning_model_explanation=reasoning_model_explanation,
                                                     gemini_model_explanation=gemini_model_explanation)
            llm_api_error = agreement_model_embedding is None or agreement_model_classification is None
            if not llm_api_error:
                # log_prob_model_verification_dict[constants.REEXPRESS_EMBEDDING_KEY] = log_prob_model_embedding
                # reasoning_model_verification_dict[constants.REEXPRESS_EMBEDDING_KEY] = reasoning_model_embedding
                reexpression_input = mcp_utils_data_format.construct_document_attributes_and_embedding(
                    log_prob_model_verification_dict, reasoning_model_verification_dict,
                    gemini_model_verification_dict, agreement_model_embedding)
                prediction_meta_data = mcp_utils_test.test(self.main_device, self.model,
                                                           self.global_uncertainty_statistics, reexpression_input)
                if prediction_meta_data is not None:
                    formatted_output_string = mcp_utils_test.format_sdm_estimator_output_for_mcp_tool(
                        prediction_meta_data, log_prob_model_explanation, reasoning_model_explanation,
                        gemini_model_explanation,
                        agreement_model_classification)
                    partial_reexpression["reexpression_input"] = reexpression_input
                    partial_reexpression["prediction_meta_data"] = prediction_meta_data
                    partial_reexpression[constants.REEXPRESS_MODEL1_CLASSIFICATION] = log_prob_model_classification
                    partial_reexpression[constants.REEXPRESS_MODEL2_CLASSIFICATION] = reasoning_model_classification
                    partial_reexpression[constants.REEXPRESS_MODEL3_CLASSIFICATION] = gemini_model_classification
                    partial_reexpression[constants.REEXPRESS_MODEL1_EXPLANATION] = log_prob_model_explanation
                    partial_reexpression[constants.REEXPRESS_MODEL2_EXPLANATION] = reasoning_model_explanation
                    partial_reexpression[constants.REEXPRESS_MODEL3_EXPLANATION] = gemini_model_explanation
                    partial_reexpression[constants.REEXPRESS_AGREEMENT_MODEL_CLASSIFICATION] = \
                        agreement_model_classification
                    now = datetime.now()
                    submitted_time = now.strftime("%Y-%m-%d %H:%M:%S")
                    partial_reexpression[constants.REEXPRESS_SUBMITTED_TIME_KEY] = submitted_time

        if formatted_output_string == "":
            formatted_output_string = (
                mcp_utils_test.get_formatted_sdm_estimator_output_string(
                    False, constants.CALIBRATION_RELIABILITY_LABEL_OOD,
                    constants.SHORT_EXPLANATION_FOR_CLASSIFICATION_CONFIDENCE__DEFAULT_ERROR,
                    constants.SHORT_EXPLANATION_FOR_CLASSIFICATION_CONFIDENCE__DEFAULT_ERROR,
                    constants.SHORT_EXPLANATION_FOR_CLASSIFICATION_CONFIDENCE__DEFAULT_ERROR,
                    agreement_model_classification=False,
                    non_odd_class_conditional_accuracy=self.model.non_odd_class_conditional_accuracy))
            self.current_reexpression = None
        else:
            partial_reexpression["formatted_output_string"] = formatted_output_string
            # Currently we do not store attached documents (if any) for archive with data additions. However, we do
            # save the file names.
            partial_reexpression[constants.REEXPRESS_QUESTION_KEY] = user_question
            partial_reexpression[constants.REEXPRESS_AI_RESPONSE_KEY] = ai_response
            partial_reexpression[constants.REEXPRESS_ATTACHED_FILE_NAMES] = available_file_names
            self.current_reexpression = partial_reexpression
            self.save_html_visualization()
        return formatted_output_string

    def get_nearest_match_meta_data(self):
        try:
            nearest_support_idx = self.current_reexpression["prediction_meta_data"]["top_k_distances_idx"][0]
            nearest_support_document_id = self.model.train_uuids[nearest_support_idx]
            nearest_match_meta_data = self.support_db.get_document(nearest_support_document_id)
            # also add true labels and predictions
            nearest_match_meta_data["model_train_label"] = self.model.train_labels[nearest_support_idx]
            nearest_match_meta_data["model_train_predicted_label"] = \
                self.model.train_predicted_labels[nearest_support_idx]
            # label_int is also stored in the db for convenience, but the source of truth is model_train_label
            return nearest_match_meta_data
        except:
            return None

    def save_html_visualization(self):
        if self.current_reexpression is not None and self.CREATE_HTML_VISUALIZATION and \
                self.HTML_VISUALIZATION_FILE is not None:
            try:
                nearest_match_meta_data = self.get_nearest_match_meta_data()
                html_content = utils_visualization.create_html_page(self.current_reexpression,
                                                                    nearest_match_meta_data=nearest_match_meta_data)
                html_file_path = Path(self.HTML_VISUALIZATION_FILE)
                if not html_file_path.exists() or not html_file_path.is_file() \
                        or html_file_path.is_symlink():
                    return
                else:
                    with open(str(html_file_path.as_posix()), 'w', encoding='utf-8') as f:
                        f.write(html_content)
            except:
                return

    def update_model_support(self, label: int) -> str:
        message = "There is no existing reexpression in the cache."
        if self.model is None or self.current_reexpression is None:
            return message
        try:
            if label not in [0, 1, data_validator.oodLabel]:
                raise AdaptationError(f"The provided label is not in [0, 1, {data_validator.oodLabel}].", "LABEL_ERROR")
            running_updates_file_path = Path(self.DATA_UPDATE_FILE)
            if not running_updates_file_path.exists() or not running_updates_file_path.is_file() \
                    or running_updates_file_path.is_symlink():
                raise AdaptationError(f"The running_updates.jsonl file does not exist.", "MISSING_UPDATES_FILE")
            if constants.ADMIN_LABEL_MODE:
                document_id = f"added_{str(uuid.uuid4())}"
            else:
                document_id = f"user_added_{str(uuid.uuid4())}"
            prediction_meta_data = self.current_reexpression["prediction_meta_data"]
            exemplar_vector = prediction_meta_data["exemplar_vector"]

            json_for_archive = {}
            reexpression_input = self.current_reexpression["reexpression_input"]
            embedding = [float(x) for x in reexpression_input.squeeze().detach().numpy().tolist()]
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
            json_for_archive[constants.REEXPRESS_MODEL3_CLASSIFICATION] = \
                self.current_reexpression[constants.REEXPRESS_MODEL3_CLASSIFICATION]

            json_for_archive[constants.REEXPRESS_MODEL1_EXPLANATION] = \
                self.current_reexpression[constants.REEXPRESS_MODEL1_EXPLANATION]
            json_for_archive[constants.REEXPRESS_MODEL2_EXPLANATION] = \
                self.current_reexpression[constants.REEXPRESS_MODEL2_EXPLANATION]
            json_for_archive[constants.REEXPRESS_MODEL3_EXPLANATION] = \
                self.current_reexpression[constants.REEXPRESS_MODEL3_EXPLANATION]

            json_for_archive[constants.REEXPRESS_AGREEMENT_MODEL_CLASSIFICATION] = \
                self.current_reexpression[constants.REEXPRESS_AGREEMENT_MODEL_CLASSIFICATION]

            json_for_archive[constants.REEXPRESS_ATTACHED_FILE_NAMES] = \
                self.current_reexpression[constants.REEXPRESS_ATTACHED_FILE_NAMES]
            json_for_archive[constants.REEXPRESS_INFO_KEY] = self.current_reexpression[constants.REEXPRESS_SUBMITTED_TIME_KEY]
            utils_model.save_by_appending_json_lines(str(running_updates_file_path.as_posix()), [json_for_archive])
            # Note: Currently there is no notion of database rollback if subsequent saving of the index fails (a la
            # standard sqlite operations with macOS Core Data). However,
            # in such failures, or when a change to an adaptation is needed, it is always possible to reset to the original
            # database in the repo and then batch add (after any needed changes to the JSON) the file DATA_UPDATE_FILE,
            # and then restart the server. See the documentation.
            self.model.add_to_support(label=label, predicted_label=prediction_meta_data["prediction"],
                                      document_id=document_id, exemplar_vector=exemplar_vector)
            message = "ERROR: Unable to save changes to the training set database. The cache has been cleared."
            utils_model.save_support_set_updates(self.model, self.MODEL_DIR)

            # add to document database
            add_to_support_db_message = ""
            try:
                success = self.support_db.add_document(
                    document_id=document_id,
                    model1_explanation=self.current_reexpression[constants.REEXPRESS_MODEL1_EXPLANATION],
                    model2_explanation=self.current_reexpression[constants.REEXPRESS_MODEL2_EXPLANATION],
                    model3_explanation=self.current_reexpression[constants.REEXPRESS_MODEL3_EXPLANATION],
                    model1_classification_int=int(self.current_reexpression[constants.REEXPRESS_MODEL1_CLASSIFICATION]),
                    model2_classification_int=int(self.current_reexpression[constants.REEXPRESS_MODEL2_CLASSIFICATION]),
                    model3_classification_int=int(self.current_reexpression[constants.REEXPRESS_MODEL3_CLASSIFICATION]),
                    model4_agreement_classification_int=int(self.current_reexpression[constants.REEXPRESS_AGREEMENT_MODEL_CLASSIFICATION]),
                    label_int=label,
                    label_was_updated_int=0,
                    document_source="user_added",
                    info=constants.REEXPRESS_MCP_SERVER_VERSION,
                    user_question=json_for_archive[constants.REEXPRESS_QUESTION_KEY],
                    ai_response=self.current_reexpression[constants.REEXPRESS_AI_RESPONSE_KEY]
                )
                assert success
            except:
                add_to_support_db_message = f" (However, we were unable to update the support database, so the text will not be available for introspection via the HTML visualization. Before adding additional documents, check that the database file exists at {self.reexpress_mcp_server_support_documents_file}.)"

            self.current_reexpression = None

            if label == 0:
                string_label = f"{constants.MCP_SERVER_NOT_VERIFIED_CLASS_LABEL} (Class 0)"
            elif label == 1:
                string_label = f"{constants.MCP_SERVER_VERIFIED_CLASS_LABEL} (Class 1)"
            elif label == data_validator.oodLabel:
                string_label = f"{data_validator.getDefaultLabelName(label=label, abbreviated=False)}"
            else:
                raise AdaptationError(f"The provided label is not in [0, 1, {data_validator.oodLabel}].", "LABEL_ERROR")

            message = f"Successfully added document id {document_id} to the training set database with label: {string_label}. The training set database now contains {self.model.support_index.ntotal} labeled examples.{add_to_support_db_message}"
            return message
        except AdaptationError as e:
            self.current_reexpression = None  # clear the cache to avoid inconsistencies
            return f"ERROR: Unable to save changes to the training set database. {e} The cache has been cleared."
        except:
            self.current_reexpression = None  # clear the cache to avoid inconsistencies
            return message

    def get_reexpress_view(self) -> str:
        if self.current_reexpression is None:
            return "There is no existing reexpression in the cache."  # also return file access information

        prediction_meta_data = self.current_reexpression["prediction_meta_data"]
        # also show files in consideration, if any
        files_in_consideration_message = \
            mcp_utils_test.get_files_in_consideration_message(self.current_reexpression[constants.REEXPRESS_ATTACHED_FILE_NAMES])
        formatted_output_string = f"""
            {constants.predictedFull}: {constants.MCP_SERVER_VERIFIED_CLASS_LABEL if prediction_meta_data["prediction"] == 1 else constants.MCP_SERVER_NOT_VERIFIED_CLASS_LABEL}\n
            Out-of-distribution: {prediction_meta_data["is_ood_lower"]}\n
            {constants.qFull}: {int(prediction_meta_data["original_q"])}\n
            {constants.dFull} Quantile: {torch.min(prediction_meta_data["distance_quantiles"]).item()}\n
            {constants.fFull}: {prediction_meta_data["f"].detach().numpy().tolist()}\n
            Valid index-conditional estimate (at alpha'={prediction_meta_data["non_odd_class_conditional_accuracy"]}, min_valid_rescaled_q={prediction_meta_data["min_valid_qbin_for_class_conditional_accuracy_with_bounded_error"]}, class-wise output thresholds={prediction_meta_data["non_odd_thresholds"]}): {prediction_meta_data["is_valid_index_conditional__lower"]}\n
            p(y | x)_lower: {prediction_meta_data["rescaled_prediction_conditional_distribution__lower"].detach().numpy().tolist()}\n
            Rescaled q_lower: {prediction_meta_data["soft_qbin__lower"][0].item()}\n
            Iterated offset_lower (for class {prediction_meta_data["prediction"]}): {prediction_meta_data["iterated_lower_offset__lower"]}\n
            Effective sample size (by class): {prediction_meta_data["cumulative_effective_sample_sizes"].detach().numpy().tolist()}\n
            {files_in_consideration_message}
            ---------------\n
            {self.current_reexpression["formatted_output_string"]}
        """
        return formatted_output_string
