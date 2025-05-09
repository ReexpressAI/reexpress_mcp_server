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
        # For provenance of any added instances:
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
        previous_query_and_response_to_verify_string = \
            f"<question> {attached_documents}{user_question} </question> <ai_response> {ai_response} </ai_response>"
        # currently identical:
        previous_query_and_response_to_verify_string_reasoning = \
            f"<question> {attached_documents}{user_question} </question> <ai_response> {ai_response} </ai_response>"
        task_configs = [(mcp_utils_llm_api.get_document_attributes,
                         previous_query_and_response_to_verify_string),
                        (mcp_utils_llm_api.get_document_attributes_from_reasoning,
                         previous_query_and_response_to_verify_string_reasoning)]
        try:
            results = await self._run_tasks_with_taskgroup(task_configs)
            log_prob_model_verification_dict, reasoning_model_verification_dict = results
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
            log_prob_model_explanation, reasoning_model_explanation = \
                mcp_utils_llm_api.get_model_explanations(log_prob_model_verification_dict, reasoning_model_verification_dict)
            log_prob_model_embedding, reasoning_model_embedding = \
                mcp_utils_llm_api.llm_api_controller(log_prob_model_explanation=log_prob_model_explanation,
                                                     reasoning_model_explanation=reasoning_model_explanation)
            llm_api_error = log_prob_model_embedding is None or reasoning_model_embedding is None
            if not llm_api_error:
                log_prob_model_verification_dict[constants.REEXPRESS_EMBEDDING_KEY] = log_prob_model_embedding
                reasoning_model_verification_dict[constants.REEXPRESS_EMBEDDING_KEY] = reasoning_model_embedding
                reexpression_input = mcp_utils_data_format.construct_document_attributes_and_embedding(
                    log_prob_model_verification_dict, reasoning_model_verification_dict)
                prediction_meta_data = mcp_utils_test.test(self.main_device, self.model,
                                                           self.global_uncertainty_statistics, reexpression_input)
                if prediction_meta_data is not None:
                    formatted_output_string = mcp_utils_test.format_sdm_estimator_output_for_mcp_tool(
                        prediction_meta_data, log_prob_model_explanation, reasoning_model_explanation,
                        self.model.ood_limit)
                    partial_reexpression["reexpression_input"] = reexpression_input
                    partial_reexpression["prediction_meta_data"] = prediction_meta_data
                    partial_reexpression[constants.REEXPRESS_MODEL1_EXPLANATION] = log_prob_model_explanation
                    partial_reexpression[constants.REEXPRESS_MODEL2_EXPLANATION] = reasoning_model_explanation

        if formatted_output_string == "":
            formatted_output_string = (
                mcp_utils_test.get_formatted_sdm_estimator_output_string(
                    False, 0.01, constants.CALIBRATION_RELIABILITY_LABEL_OOD,
                    constants.SHORT_EXPLANATION_FOR_CLASSIFICATION_CONFIDENCE__DEFAULT_ERROR,
                    constants.SHORT_EXPLANATION_FOR_CLASSIFICATION_CONFIDENCE__DEFAULT_ERROR))
            self.current_reexpression = None
        else:
            partial_reexpression["formatted_output_string"] = formatted_output_string
            # Currently we do not store attached documents (if any) for archive with data additions. However, we do
            # save the file names.
            partial_reexpression[constants.REEXPRESS_QUESTION_KEY] = user_question
            partial_reexpression[constants.REEXPRESS_AI_RESPONSE_KEY] = ai_response
            partial_reexpression[constants.REEXPRESS_ATTACHED_FILE_NAMES] = available_file_names
            self.current_reexpression = partial_reexpression
        return formatted_output_string

    def update_model_support(self, label: int) -> str:
        message = "There is no existing reexpression in the cache."
        if self.model is None or self.current_reexpression is None:
            return message
        try:
            if label not in [0, 1]:
                raise AdaptationError(f"The provided label is not in [0, 1].", "LABEL_ERROR")
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

            json_for_archive[constants.REEXPRESS_MODEL1_EXPLANATION] = \
                self.current_reexpression[constants.REEXPRESS_MODEL1_EXPLANATION]
            json_for_archive[constants.REEXPRESS_MODEL2_EXPLANATION] = \
                self.current_reexpression[constants.REEXPRESS_MODEL2_EXPLANATION]
            json_for_archive[constants.REEXPRESS_ATTACHED_FILE_NAMES] = \
                self.current_reexpression[constants.REEXPRESS_ATTACHED_FILE_NAMES]
            now = datetime.now()
            submitted_time = now.strftime("%Y-%m-%d %H:%M:%S")
            json_for_archive[constants.REEXPRESS_INFO_KEY] = submitted_time
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

            self.current_reexpression = None
            string_label = f"{constants.MCP_SERVER_VERIFIED_CLASS_LABEL} (Class 1)" if label == 1 else \
                f"{constants.MCP_SERVER_NOT_VERIFIED_CLASS_LABEL} (Class 0)"
            message = f"Successfully added document id {document_id} to the training set database with label: {string_label}. The training set database now contains {self.model.support_index.ntotal} labeled examples."
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
        if len(self.current_reexpression[constants.REEXPRESS_ATTACHED_FILE_NAMES]) > 0:
            files_in_consideration_message = f'The verification model had access to: ' \
                                             f'{",".join(self.current_reexpression[constants.REEXPRESS_ATTACHED_FILE_NAMES])}\n\n'
        else:
            files_in_consideration_message = f'The verification model did not have access to any external files.\n\n'
        formatted_output_string = f"""
            {constants.predictedFull}: {constants.MCP_SERVER_VERIFIED_CLASS_LABEL if prediction_meta_data["prediction"] == 1 else constants.MCP_SERVER_NOT_VERIFIED_CLASS_LABEL}\n
            {constants.qFull}: {prediction_meta_data["original_q"]}\n
            {constants.dFull} Quantile: {torch.min(prediction_meta_data["distance_quantiles"]).item()}\n
            {constants.fFull}: {prediction_meta_data["f"].detach().numpy().tolist()}\n
            Valid index-conditional estimate: {prediction_meta_data["is_valid_index_conditional__lower"]}\n
            p(y | x)_lower: {prediction_meta_data["rescaled_prediction_conditional_distribution__lower"].detach().numpy().tolist()}\n
            Rescaled q_lower: {prediction_meta_data["soft_qbin__lower"][0].item()}\n
            Iterated offset_lower (for class {prediction_meta_data["prediction"]}): {prediction_meta_data["iterated_lower_offset__lower"]}\n
            Effective sample size (by class): {prediction_meta_data["cumulative_effective_sample_sizes"].detach().numpy().tolist()}\n
            {files_in_consideration_message}
            ---------------\n
            {self.current_reexpression["formatted_output_string"]}
        """
        return formatted_output_string
