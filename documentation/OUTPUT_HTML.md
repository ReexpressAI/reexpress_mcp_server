# Reexpress MCP Server Output HTML

## Enable

To enable the creation of a static HTML page with each call to the main Reexpress tool, set the following environment variable to '1':

```bash
export REEXPRESS_MCP_SAVE_OUTPUT='1'
```

This should appear in the `llm_api_setup.sh` file as described in [INSTALL.md](/INSTALL.md#5-configure-environment-variables-and-llm-api-keys).

Each time the main Reexpress tool is called, this will save the file `${REEXPRESS_MCP_MODEL_DIR}/visualize/current_reexpression.html`. 

> [!TIP]
> The file `${REEXPRESS_MCP_MODEL_DIR}/visualize/current_reexpression.html` gets overwritten with each tool call. Add a timestamp to `html_file_path` in MCPServerStateController().save_html_visualization() if you want to save all calls.

## Examples

See the description of the `reexpress()` tool in [HOW_TO_USE.md](HOW_TO_USE.md#the-reexpress-tool-reexpressuser_question-str-ai_response-str) for details of the information contained in the HTML. We include additional examples below.

Examples:
- [Verified with confidence `>= 90%`](example_output/html_output_examples/current_reexpression_pos_example.html)
- [Verified but confidence `< 90%`](example_output/html_output_examples/current_reexpression_pos_and_lower_confidence_example.html)
- [NOT Verified with confidence `>= 90%`](example_output/html_output_examples/current_reexpression_neg_example.html)
