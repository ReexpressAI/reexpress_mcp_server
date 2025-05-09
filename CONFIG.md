# Config options

The MCP server has a small number of user-adjustable options defined in [code/reexpress/mcp_settings.json](code/reexpress/mcp_settings.json).[^1] These are read once each time the server is started (e.g., when Claude Desktop is started). If you make changes, you must restart the server (e.g., by restarting Claude Desktop). These control limits on the tool-calls, and as such, are not intended to be modified directly by the MCP server, nor other third-party servers or LLM agents.

```json
{
    "allowed_parent_directory": "",
    "file_access_timeout_in_minutes": 5.0,
    "max_lines_to_consider": 1000,
    "max_file_size_to_open_in_mb": 5.0,
    "max_number_of_files_to_send_to_llms": 3,
    "max_number_of_reexpress_tool_calls_before_required_server_reset": 100,
    "max_number_of_sequential_reexpress_tool_calls_before_required_user_reset": 10
}
```

"allowed_parent_directory": "",
> string

> Blank by default, which means the MCP server cannot see local files, even if you use the ReexpressDirectorySet() tool and ReexpressFileSet() tool. To enable those tools, you must supply an absolute path, for example, "/Users/a/Documents/projects". If provided, only calls to ReexpressDirectorySet() and ReexpressFileSet() will be considered if the full path shares a common path with the string in allowed_parent_directory.

"file_access_timeout_in_minutes": 5.0,
> float

> When you call ReexpressFileSet(), files will only be available to the verification LLMs for this many minutes (5 minutes, by default).

"max_lines_to_consider": 1000,
> int

> When you call ReexpressFileSet(), only this many initial lines will be available to the verification LLMs (the first 1000 lines, by default).

"max_file_size_to_open_in_mb": 5.0,
> float

> When you call ReexpressFileSet(), only files <= this many MB will be available to the verification LLMs (5 MB, by default).

"max_number_of_files_to_send_to_llms": 3,
> int

> When you call ReexpressFileSet(), only the most recent this many files (that have not otherwise expired due to time limits) will be available to the verification LLMs (3 files, by default).

"max_number_of_reexpress_tool_calls_before_required_server_reset": 100,
> int

> The Reexpress tool can only be called this many times before you must restart the server (100 calls, by default). This is intended as a constraint on the total number of calls (and by extension, LLM API calls). As a standard best practice for agents in general, we also recommend setting budget limits on the verification LLM API's---and, if applicable, the tool-calling LLM (if using an API for Claude, for example)---as a final, maximally hard constraint.

"max_number_of_sequential_reexpress_tool_calls_before_required_user_reset": 10
> int

> The Reexpress tool can only be called this many times sequentially before you (or another agent) must call the ReexpressReset tool. This is a soft constraint because the value can be reset, but the limit (10, by default) itself can only be changed with a server restart. Note that Claude and other LLMs may call the ReexpressReset tool themselves without input if you do not require notification for each call to that tool. Calling ReexpressReset does not impact "max_number_of_reexpress_tool_calls_before_required_server_reset".

> [!WARNING]
> These are lightweight constraints, which work well in practice when the Reexpress server is the only MCP server installed. However, keep in mind that---as with other aspects of MCP servers, in general---another third-party agent or LLM tool could modify the file (and even, in principle, re-start the server), abrogating the constraints. Only add other MCP servers from sources that you trust.

[^1]: For convenience, a backup copy of the config file with the original defaults is stored in [documentation/defaults/backup_original_defaults__mcp_settings.json](documentation/defaults/backup_original_defaults__mcp_settings.json). You must rename this to mcp_settings.json and move to the code/reexpress/ directory for it to be read by the server.
