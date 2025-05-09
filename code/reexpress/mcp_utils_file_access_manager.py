# Copyright Reexpress AI, Inc. All rights reserved.

# Management of files to send for verification to the MCP server's LLMs. This is used when the verification requires
# access to the original files being considered by the tool-calling LLM. Note that by default, such file access
# is turned off via the settings in mcp_settings.json.

from pathlib import Path
import codecs

import constants
import utils_model

import os
import time
from datetime import datetime


class FileAccessError(Exception):
    def __init__(self, message="A file access error occurred", error_code=None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

    def __str__(self):
        if self.error_code is not None:
            return f"{self.error_code}: {self.message}"
        return self.message


class ExternalFileController:
    """
    Manages file access and content for data to send to the verification LLMs. The file content is read once on add.

    By default, the ability to access external files is turned off. The allowed_parent_directory must be a valid string
    path in mcp_settings.json. Once that is set, only files and directories that share a common path with
    allowed_parent_directory will be allowed. Also, note that hidden files (starting with .) cannot be read.

    The basic approach is simple but sufficient for typical use cases. The user can add plain text files to a
    running list. When the verification tool is called, the file content will be sent to the verification LLMs. Use
    this when you want to ensure that the verification LLMs have access to the verbatim text of the underlying files,
    rather than depending on the tool-calling LLM to send applicable text to the Reexpress tool. The canonical
    use-case is if you ask an LLM to summarize a document or codebase and you want the verification tool to check
    that summary against the original document or codebase. Without this, the verification would be un-grounded
    relative to the original. On the other hand, for short code-snippets and related cases when you
    just need an initial, first-pass verification, it may be sufficient to just have the tool-calling LLM send
    the relevant portions of the text to the verification tool as part of the arguments to the function
    reexpress(user_question: str, ai_response: str), as per our recommended prompt. However, in those cases,
    keep in mind that the verification is contingent on the tool-calling LLM faithfully representing the grounding
    documents.

    The file mcp_settings.json controls how long files will be accessible (in terms of elapsed time in minutes),
    the max number of lines to consider for each file, the max file size that can be opened,
    and the total number of files to consider. Currently only plain text files with utf-8 encoding (e.g., .txt, .py,
    .swift, .md, .csv, .json, .html, etc.) are
    supported. PDFs, or files in other encodings, must first be converted by another tool. Keep in mind that any
    files sent to the verification LLMs will incur corresponding per-token costs. The default is to keep the total
    amount of text sent relatively modest (e.g., the first 1000 lines of each of at most 3 files),
    but that can be modified by changing the defaults in mcp_settings.json. See the documentation
    for additional details.

    The settings in mcp_settings.json are purposefully not intended to be modifiable via LLM tool calls
    in order to serve as hard constraints. However, keep in mind that---as with other aspects of MCP servers,
    in general---another third-party agent or LLM tool could modify the file (and even, in principle,
    re-start the server). Only add other MCP servers from sources that you trust.

    This has only been tested on macOS Sequoia 15. The behavior on Linux is likely similar, but untested. This is
    currently unsupported on Windows. We plan to release a future version of the MCP server that supports Windows
    and that has been tested on Linux distributions.
    """

    def __init__(self, mcp_server_dir: str = ""):
        self.file_access_enabled = False
        self.allowed_parent_directory = ""
        self.file_access_timeout_in_minutes = 5.0
        self.max_lines_to_consider = 1000
        self.max_file_size_to_open_in_mb = 5.0
        self.max_number_of_files_to_send_to_llms = 3

        file_access_settings_json = None
        dir_path = Path(mcp_server_dir)
        if dir_path.is_dir() and not dir_path.is_symlink():
            file_access_settings_file = Path(dir_path, constants.MCP_SERVER_SETTINGS_FILENAME)
            if file_access_settings_file.is_file() and not file_access_settings_file.is_symlink():
                file_access_settings_json = utils_model.read_json_file(str(file_access_settings_file.as_posix()))
        if file_access_settings_json is not None:
            self._parse_file_access_settings(file_access_settings_json)
        self.file_access_enabled = self.allowed_parent_directory != ""

        self.current_file_content_list = []
        # Optional. Can be set to avoid specifying a full absolute path for each file. This (and the resolved paths
        # of any added files) MUST share a common path with self.allowed_parent_directory, which is blank by default,
        # effectively disabling adding any files.
        self.available_environment_parent_directory = ""

    def _parse_file_access_settings(self, file_access_settings_json):
        try:
            allowed_parent_directory = str(file_access_settings_json["allowed_parent_directory"]).strip()
            dir_path = Path(allowed_parent_directory)
            if dir_path.is_dir() and not dir_path.is_symlink():
                self.allowed_parent_directory = allowed_parent_directory
                self.file_access_timeout_in_minutes = \
                    max(0.0, float(file_access_settings_json["file_access_timeout_in_minutes"]))
                self.max_lines_to_consider = \
                    max(0, int(file_access_settings_json["max_lines_to_consider"]))
                self.max_file_size_to_open_in_mb = \
                    max(0.0, float(file_access_settings_json["max_file_size_to_open_in_mb"]))
                self.max_number_of_files_to_send_to_llms = \
                    max(0, int(file_access_settings_json["max_number_of_files_to_send_to_llms"]))
        except:
            self.allowed_parent_directory = ""

    def _add_to_file_content_list(self, attached_file_content: str, file_source: str, expiration_time: float):
        self.current_file_content_list.append(
            (f'<attached_file source="{file_source}"> {attached_file_content} </attached_file>',
             expiration_time,
             file_source)
        )

    def get_current_external_file_content(self):
        current_time = time.time()
        content_xml = []
        available_file_names = []
        for available_file in self.current_file_content_list:
            expiration_time = available_file[1]
            if current_time < expiration_time:
                content_xml.append(available_file[0])
                available_file_names.append(available_file[2])
        return " ".join(content_xml[-self.max_number_of_files_to_send_to_llms:]), \
            available_file_names[-self.max_number_of_files_to_send_to_llms:]

    def _get_available_file_names(self, with_expiration_string=False):
        current_time = time.time()
        available_file_names = []
        for available_file in self.current_file_content_list:
            expiration_time = available_file[1]
            if current_time < expiration_time:
                if with_expiration_string:
                    available_file_names.append(f"{available_file[2]} "
                                                f"(access expires {datetime.fromtimestamp(expiration_time).strftime('%Y-%m-%d %H:%M:%S')})")
                else:
                    available_file_names.append(available_file[2])
        return available_file_names[-self.max_number_of_files_to_send_to_llms:]

    def add_file(self, filename_with_path: str) -> str:
        if not self.file_access_enabled:
            return f"ERROR: External file access is not enabled."
        try:
            # if filename_with_path is absolute, it will replace self.available_environment_parent_directory, which
            # is the desired behavior --- i.e., can set a path prefix for convenience, but can also override
            file_path = Path(self.available_environment_parent_directory, filename_with_path)
            if not file_path.is_file() or file_path.is_symlink():
                raise FileAccessError(
                    f"The file is not available.", "FILE_ACCESS")
            # Resolve all symlinks, if any (Redundant given the above, but here in case we allow symlinks in the future)
            file_path = file_path.resolve(strict=False)
            allowed_path = Path(self.allowed_parent_directory).resolve(strict=False)
            if not str(file_path).startswith(str(allowed_path)):
                raise FileAccessError(
                    f"The file path has not been granted access. It must be a child of {self.allowed_parent_directory}.",
                    "DIRECTORY_ACCESS")
            if not os.path.commonpath(
                    [self.allowed_parent_directory, str(file_path.as_posix())]) == self.allowed_parent_directory:
                raise FileAccessError(f"The file path has not been granted access. It must be a child of {self.allowed_parent_directory}. Consider modifying {constants.MCP_SERVER_SETTINGS_FILENAME}.", "DIRECTORY_ACCESS")
            if file_path.name.startswith('.'):
                raise FileAccessError("Hidden files starting with . cannot be added.", "HIDDEN_FILE")
            file_size = file_path.stat().st_size
            max_size_bytes = self.max_file_size_to_open_in_mb * 1024 * 1024  # MB to bytes
            if file_size < max_size_bytes:
                file_lines = []
                line_i = 0
                with codecs.open(str(file_path.as_posix()), "r", encoding="utf-8") as f:
                    for line in f:
                        if line_i >= self.max_lines_to_consider:
                            break
                        file_lines.append(line)
                        line_i += 1
                file_content = "".join(file_lines).strip()
                if len(file_content) > 0:
                    expiration_time = time.time() + self.file_access_timeout_in_minutes * 60
                    self._add_to_file_content_list(attached_file_content=file_content,
                                                   file_source=file_path.name,
                                                   expiration_time=expiration_time)
                    if len(file_lines) == self.max_lines_to_consider:
                        length_message = f" the first {len(file_lines)} lines of "
                    else:
                        length_message = " "
                    message = f"Added access to{length_message}{str(file_path.as_posix())}. Accessible until {datetime.fromtimestamp(expiration_time).strftime('%Y-%m-%d %H:%M:%S')}."
                    if len(self.current_file_content_list) > self.max_number_of_files_to_send_to_llms:
                        removed_file = self.current_file_content_list.pop(0)
                        message += f" \nRemoved access to {removed_file[2]}"
                    message += f" \nThe following files will be sent to the Reexpress tool the next time it is called, provided access has not expired: {', '.join(self._get_available_file_names(with_expiration_string=True))}"
                else:
                    message = f"ERROR: File at {file_path} was empty."
            else:
                message = f"ERROR: File size ({file_size} bytes) exceeds {self.max_file_size_to_open_in_mb} MB."
        except FileNotFoundError:
            message = f"ERROR: File not found at {filename_with_path}"
        except FileAccessError as e:
            message = f"ERROR: Unable to open {filename_with_path} given the parent directory {self.available_environment_parent_directory}. {e}"
        except OSError as e:
            message = f"ERROR: Unable to open {filename_with_path} given the parent directory {self.available_environment_parent_directory}"
        return message

    def add_environment_parent_directory(self, parent_directory: str) -> str:
        if self.file_access_enabled:
            try:
                dir_path = Path(parent_directory)
                # currently we keep it simple and don't follow symlinks
                if dir_path.is_dir() and not dir_path.is_symlink():
                    dir_path_string = str(dir_path.as_posix())
                    if os.path.commonpath(
                            [self.allowed_parent_directory, dir_path_string]) == self.allowed_parent_directory:
                        self.available_environment_parent_directory = dir_path_string
                        return f"Added {dir_path_string} as the current parent directory."
                else:
                    self.available_environment_parent_directory = ""
                    return f"ERROR: Unable to add {parent_directory}. " \
                           f"Are you sure it exists and shares a common path with {self.allowed_parent_directory}?"
            except:
                return f"ERROR: Unable to add {parent_directory}"
        else:
            return f"ERROR: External file access is not enabled."

    def remove_all_file_access(self) -> str:
        self.available_environment_parent_directory = ""
        self.current_file_content_list = []
        if self.file_access_enabled:
            return f"The parent file directory and all files have been removed."
        return f"ERROR: External file access is not enabled, so there is no file access to remove."

