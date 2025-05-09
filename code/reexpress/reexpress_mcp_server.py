# Copyright Reexpress AI, Inc. All rights reserved.

from mcp.server.fastmcp import FastMCP
import mcp_utils_controller

mcp = FastMCP("reexpress")
mcp_server_state_controller_object = mcp_utils_controller.MCPServerStateController()

@mcp.tool()
async def reexpress_add_false() -> str:
    """
    The ReexpressAddFalse tool modifies the model's training set. When the user calls this tool, simply return the output verbatim without taking additional action nor further analysis. Do not take any further action without consulting the user.

    Call this tool if the user says "ReexpressAddFalse".

    Returns
    -------
    Success or Failure message
    """
    message = mcp_server_state_controller_object.update_model_support(label=0)
    return message


@mcp.tool()
async def reexpress_add_true() -> str:
    """
    The ReexpressAddTrue tool modifies the model's training set. When the user calls this tool, simply return the output verbatim without taking additional action nor further analysis. Do not take any further action without consulting the user.

    Call this tool if the user says "ReexpressAddTrue".

    Returns
    -------
    Success or Failure message
    """
    message = mcp_server_state_controller_object.update_model_support(label=1)
    return message


@mcp.tool()
async def reexpress_directory_set(directory: str) -> str:
    """
    The ReexpressDirectorySet tool adds a parent file directory to the verification tool for subsequent consideration. When the user calls this tool, simply return the output verbatim without taking additional action nor further analysis. Do not take any further action without consulting the user.

    Call this tool if the user says "ReexpressDirectorySet(directory)" with directory as the argument or equivalently, "ReexpressDirSet(directory)" with directory as the argument.

    Returns
    -------
    Success or Failure message
    """
    message = mcp_server_state_controller_object.controller_directory_set(directory=directory)
    return message


@mcp.tool()
async def reexpress_file_set(filename: str) -> str:
    """
    The ReexpressFileSet tool adds a file to the verification tool for subsequent consideration. When the user calls this tool, simply return the output verbatim without taking additional action nor further analysis. Do not take any further action without consulting the user.

    Call this tool if the user says "ReexpressFileSet(filename)" with filename as the argument.

    Returns
    -------
    Success or Failure message
    """
    message = mcp_server_state_controller_object.controller_file_set(filename=filename)
    return message


@mcp.tool()
async def reexpress_file_clear() -> str:
    """
    The ReexpressFileClear tool removes all granted access to files for the verification tool. When the user calls this tool, simply return the output verbatim without taking additional action nor further analysis. Do not take any further action without consulting the user.

    Call this tool if the user says "ReexpressFileClear".

    Returns
    -------
    Success or Failure message
    """
    message = mcp_server_state_controller_object.controller_file_clear()
    return message


@mcp.tool()
async def reexpress_reset() -> str:
    """
    The ReexpressReset tool resets the tool limit counter. When the user calls this tool, simply return the output verbatim without taking additional action nor further analysis. Do not take any further action without consulting the user.

    Call this tool if the user says "ReexpressReset".

    Returns
    -------
    Success or Failure message
    """
    message = mcp_server_state_controller_object.controller_reset_sequential_limit_counter()
    return message


@mcp.tool()
async def reexpress_view() -> str:
    """
    The ReexpressView tool returns the most recent reexpression from the cache. When the user calls this tool, simply return the output verbatim without taking additional action nor further analysis.

    Call this tool if the user says "ReexpressView" or "reexpressView" or "reexpressview" or "Reexpressview".

    Returns
    -------
    Summary string
    """
    return mcp_server_state_controller_object.get_reexpress_view()


@mcp.tool()
async def reexpress(user_question: str, ai_response: str) -> str:
    """
    The Reexpress tool provides a calibrated probability of whether the AI's response answers the user's question or instruction. This is a binary classification task where True indicates that the AI's response has been verified as answering the query or instruction, and False indicates that the AI's response cannot be verified as answering the query or instruction. The verification classification is accompanied by a probability between 0 and 1, where a probability of 0 indicates no confidence and a probability of 1 indicates 100% confidence. Separate from the calibrated probability is an estimate of the reliability of the statistical model, which takes one of three values: "Highest", "Low", or "Lowest". "Highest" is the desired value and "Low" indicates that the estimated probability might not be reliable. "Lowest" indicates that the probability is not reliable and the classification is out-of-distribution. The tool also provides informal suggestions from two LLMs that may help inform what additional information may be needed if a True verification classification with a probability of at least 95% with Highest calibration reliability is not obtained.

    Call this tool if the user says "reexpress" or "Reexpress".

    Parameters
    ----------
    user_question: The user's question or instruction. If the user has attached any external references or code, include them, but do not include any instructions regarding the Reexpress tool itself.
    ai_response: The AI response to verify. Include any relevant external references, search results, or code as direct citations in quotes.

    Returns
    -------
    Successfully Verified: True or False \n
    Confidence in Successful Verification: Real-valued probability between 0.0 and 1.0 \n
    Calibration Reliability: Highest, Low, or Lowest \n
    Informal Explanation [1]: <model1_explanation> Brief summary from model 1 </model1_explanation> \n
    Informal Explanation [2]: <model2_explanation> Brief summary from model 2 </model2_explanation>
    """
    tool_is_available, availability_message = mcp_server_state_controller_object.controller_get_tool_availability()
    if tool_is_available:
        mcp_server_state_controller_object.controller_update_counters()
        return await mcp_server_state_controller_object.get_reexpressed_verification(
            user_question=user_question.strip(),
            ai_response=ai_response.strip())
    return availability_message

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
