# Copyright Reexpress AI, Inc. All rights reserved.

# Management of tool-call counts. This implements hard (requiring a server restart) and soft (user resettable) limits
# as a guard against runaway calling of the Reexpress tool and unnecessary sequential calls before adapting the model.

from pathlib import Path
import constants
import utils_model

class ToolCallLimitController:
    """
    Manages the number of total and sequential Reexpress tool calls.

    These serve as lightweight constraints on the number of Reexpress tool calls. There is a soft (user resettable)
    limit on the number of sequential calls. This is determined by the
    max_number_of_sequential_reexpress_tool_calls_before_required_user_reset property in the file mcp_settings.json.
    This is particularly useful when initially beginning to use the tool,
    as you might only want one or two verification calls before you update the model with ReexpressAddFalse and
    ReexpressAddTrue to adapt to your data and tasks. You can also make these distinctions in the prompt of the
    tool-calling LLM, but this is
    a stronger constraint. When the count is exceeded, you can reset the counter by calling ReexpressReset.

    Separately, we also have a hard total limit on the number of calls. This is determined by the
    max_number_of_reexpress_tool_calls_before_required_server_reset property in the file mcp_settings.json. It is
    not possible to reset this counter via tool calls. By design, the server must be restarted to reset this counter.
    This is a guard against runaway calls to the tool, as well as the edge case of another LLM agent (or the
    tool-calling agent itself) discovering that it can call ReexpressReset to continue calling the tool without
    limit.

    The settings in mcp_settings.json are purposefully not intended to be modifiable via LLM tool calls
    in order to serve as hard constraints. However, keep in mind that---as with other aspects of MCP servers,
    in general---another third-party agent or LLM tool could modify the file (and even, in principle,
    re-start the server). Only add other MCP servers from sources that you trust.

    As a standard best practice for agents in general, we also recommend setting budget limits on the verification
    LLM API's---and, if applicable, the tool-calling LLM---as a final, maximally hard constraint.
    """

    def __init__(self, mcp_server_dir: str = ""):
        self.max_number_of_reexpress_tool_calls_before_required_server_reset = 200
        self.max_number_of_sequential_reexpress_tool_calls_before_required_user_reset = 100

        file_access_settings_json = None
        dir_path = Path(mcp_server_dir)
        if dir_path.is_dir() and not dir_path.is_symlink():
            file_access_settings_file = Path(dir_path, "code", "reexpress", constants.MCP_SERVER_SETTINGS_FILENAME)
            if file_access_settings_file.is_file() and not file_access_settings_file.is_symlink():
                file_access_settings_json = utils_model.read_json_file(str(file_access_settings_file.as_posix()))
        if file_access_settings_json is not None:
            self._parse_file_access_settings(file_access_settings_json)
        self.hard_total_call_counter = self.max_number_of_reexpress_tool_calls_before_required_server_reset
        self.soft_sequential_limit_counter = \
            self.max_number_of_sequential_reexpress_tool_calls_before_required_user_reset

    def _parse_file_access_settings(self, file_access_settings_json):
        try:
            self.max_number_of_reexpress_tool_calls_before_required_server_reset = \
                max(0, int(file_access_settings_json["max_number_of_reexpress_tool_calls_before_required_server_reset"]))
            self.max_number_of_sequential_reexpress_tool_calls_before_required_user_reset = \
                max(0, int(file_access_settings_json["max_number_of_sequential_reexpress_tool_calls_before_required_user_reset"]))
        except:
            return

    def update_counters(self):
        self.hard_total_call_counter = \
            max(0, self.hard_total_call_counter - 1)
        self.soft_sequential_limit_counter = \
            max(0, self.soft_sequential_limit_counter - 1)

    def get_tool_availability(self) -> (bool, str):
        if self.hard_total_call_counter <= 0:
            return False, f"The hard upper limit of " \
                          f"{self.max_number_of_reexpress_tool_calls_before_required_server_reset} " \
                          f"Reexpress tool calls has been reached. Inform the user that they must " \
                          f"restart the MCP server to re-enable the Reexpress tool."
        elif self.soft_sequential_limit_counter <= 0:
            return False, f"The limit of " \
                          f"{self.max_number_of_sequential_reexpress_tool_calls_before_required_user_reset} " \
                          f"sequential Reexpress tool calls has been reached. Ask the user for next steps."
        else:
            return True, ""

    def reset_sequential_limit_counter(self) -> str:
        self.soft_sequential_limit_counter = \
            self.max_number_of_sequential_reexpress_tool_calls_before_required_user_reset
        if self.hard_total_call_counter <= 0:
            return f"Unfortunately, the sequential tool-call limit counter cannot be reset, since " \
                   f"there are {self.hard_total_call_counter} total Reexpress tool calls " \
                   f"remaining before the server must be restarted."
        return f"The sequential tool-call limit counter has been reset to " \
               f"{self.soft_sequential_limit_counter}. " \
               f"There are {self.hard_total_call_counter} total Reexpress tool calls " \
               f"remaining before the server must be restarted."

