# Copyright Reexpress AI, Inc. All rights reserved.

import torch
import html

import argparse
import constants
import mcp_utils_test
import utils_visualization__constants

def create_html_page(current_reexpression, nearest_match_meta_data=None, nearest_match_meta_data_is_html_escaped=False):
    """
    Creates a static HTML page from model output dictionary.

    Args:
        current_reexpression: Dictionary containing the model output with keys for each field
        nearest_match_meta_data_is_html_escaped: True if the text fields have been processed with html.escape()
    Returns:
        HTML string
    """

    # Extract verification results
    # Need defaults for testing
    prediction_meta_data = current_reexpression.get("prediction_meta_data", {})

    predicted_class = prediction_meta_data.get("prediction", 0)

    successfully_verified = predicted_class == 1
    if successfully_verified:
        successfully_verified_html_class = "positive"
    else:
        successfully_verified_html_class = "negative"
    is_high_reliability_region = prediction_meta_data.get("is_high_reliability_region", False)
    is_ood = prediction_meta_data.get("is_ood", True)
    is_ood_html_class = "positive" if not is_ood else "negative"
    calibration_reliability = \
        mcp_utils_test.get_calibration_reliability_label(is_high_reliability_region, is_ood)

    # Model Level
    try:
        hr_class_conditional_accuracy = prediction_meta_data["hr_class_conditional_accuracy"]
        min_rescaled_similarity_to_determine_high_reliability_region = \
            prediction_meta_data["min_rescaled_similarity_to_determine_high_reliability_region"]
        hr_output_thresholds = \
            prediction_meta_data["hr_output_thresholds"]
        support_index_ntotal = prediction_meta_data["support_index_ntotal"]
    except:
        hr_class_conditional_accuracy = 0.0
        min_rescaled_similarity_to_determine_high_reliability_region = "N/A"
        hr_output_thresholds = "N/A"
        support_index_ntotal = "N/A"

    classification_confidence, classification_confidence_html_class = \
        mcp_utils_test.get_calibration_confidence_label(calibration_reliability=calibration_reliability,
                                                        hr_class_conditional_accuracy=hr_class_conditional_accuracy,
                                                        return_html_class=True)

    model1_name = "gpt-5-2025-08-07"
    model2_name = "gemini-2.5-pro"
    agreement_model_name = "granite-3.3-8b-instruct"

    model1_classification = current_reexpression.get(constants.REEXPRESS_MODEL1_CLASSIFICATION, False)
    model2_classification = current_reexpression.get(constants.REEXPRESS_MODEL2_CLASSIFICATION, False)
    agreement_model_classification = \
        current_reexpression.get(constants.REEXPRESS_AGREEMENT_MODEL_CLASSIFICATION, False)

    if agreement_model_classification:
        agreement_model_classification_string = "Yes"
    else:
        agreement_model_classification_string = "No"

    model1_html_class = "positive" if model1_classification else "negative"
    model2_html_class = "positive" if model2_classification else "negative"
    agreement_model_html_class = "positive" if agreement_model_classification else "negative"

    # We escape HTML as it may be contained within the responses themselves
    model1_explanation = html.escape(current_reexpression.get(constants.REEXPRESS_MODEL1_EXPLANATION, ''))
    model2_explanation = html.escape(current_reexpression.get(constants.REEXPRESS_MODEL2_EXPLANATION, ''))

    model1_summary = html.escape(current_reexpression.get(constants.REEXPRESS_MODEL1_TOPIC_SUMMARY, ''))

    files_in_consideration_message = \
        mcp_utils_test.get_files_in_consideration_message(
            current_reexpression.get(constants.REEXPRESS_ATTACHED_FILE_NAMES, [])).strip()
    submitted_time = current_reexpression.get(constants.REEXPRESS_SUBMITTED_TIME_KEY, 'N/A')

    # Uncertainty
    try:
        sdm_output = \
            prediction_meta_data["sdm_output"].detach().cpu().tolist()
        # TODO: Move this earlier to avoid duplication:
        is_high_reliability_region = prediction_meta_data["is_high_reliability_region"]
        is_high_reliability_region_html_class = "positive" if is_high_reliability_region else "negative"
        rescaled_similarity = prediction_meta_data["rescaled_similarity"]
        cumulative_effective_sample_sizes = \
            prediction_meta_data["cumulative_effective_sample_sizes"].detach().cpu().tolist()

        similarity_q = int(prediction_meta_data["q"])
        distance_d = prediction_meta_data["d"]
        magnitude = prediction_meta_data["f"].detach().cpu().tolist()
        # analysis of the effective sample size:
        distance_d_lower = prediction_meta_data["d_lower"]
        distance_d_upper = prediction_meta_data["d_upper"]
        sdm_output_d_lower = prediction_meta_data["sdm_output_d_lower"].detach().cpu().tolist()
        sdm_output_d_upper = prediction_meta_data["sdm_output_d_upper"].detach().cpu().tolist()
    except:
        sdm_output = "N/A"
        is_high_reliability_region = False
        rescaled_similarity = "N/A"
        is_high_reliability_region_html_class = "negative"
        cumulative_effective_sample_sizes = "N/A"
        similarity_q = "N/A"
        distance_d = "N/A"
        magnitude = "N/A"
        # analysis of the effective sample size:
        distance_d_lower = "N/A"
        distance_d_upper = "N/A"
        sdm_output_d_lower = "N/A"
        sdm_output_d_upper = "N/A"

    user_question = html.escape(current_reexpression.get(constants.REEXPRESS_QUESTION_KEY, ''))
    ai_response = html.escape(current_reexpression.get(constants.REEXPRESS_AI_RESPONSE_KEY, ''))

    # Nearest Match
    try:
        assert nearest_match_meta_data is not None
        nearest_match_html_string = nearest_match_html(nearest_match_meta_data,
                                                       model1_name,
                                                       model2_name,
                                                       agreement_model_name,
                                                       content_is_html_escaped=nearest_match_meta_data_is_html_escaped)
    except:
        nearest_match_html_string = """<div class="section" style="margin-left: 40px;"> The nearest match is not available. </div>"""


    html_content_string = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reexpress MCP Server Output</title>
    {utils_visualization__constants.css_style}
</head>
<body>
    <div class="container">
        <div class="header">
            Reexpress MCP Server Output <span class="mcp-server-version">(v{constants.REEXPRESS_MCP_SERVER_VERSION})</span>
        </div>

        <div class="section">
            <div class="section-title">Verification Results</div>

            <div class="field-box" style="margin-bottom: 20px;">
                <div class="field-label">Successfully Verified (Prediction)</div>
                <div class="field-value">
                    <div class="field-value"><span class="tag tag-{successfully_verified_html_class}">{successfully_verified}</span></div>
                </div>
            </div>

            <div class="field-box" style="margin-bottom: 20px;">
                <div class="field-label">Confidence</div>
                <div class="field-value"><span class="tag tag-{classification_confidence_html_class}">{classification_confidence}</span></div>
            </div>

            <div class="explanation-box-{model1_html_class}">
                <div class="explanation-title-{model1_html_class}">Model 1 Summary <span class="model-name">({model1_name})</span></div>
                <div>{model1_summary}</div>
            </div>
            
            <div class="explanation-box-{model1_html_class}">
                <div class="explanation-title-{model1_html_class}">Model 1 Explanation <span class="model-name">({model1_name})</span></div>
                <div>{model1_explanation}</div>
            </div>

            <div class="explanation-box-{model2_html_class}">
                <div class="explanation-title-{model2_html_class}">Model 2 Explanation <span class="model-name">({model2_name})</span></div>
                <div>{model2_explanation}</div>
            </div>

            <div class="explanation-box-{agreement_model_html_class}">
                <div class="explanation-title-{agreement_model_html_class}">Model 3 Agreement <span class="model-name">({agreement_model_name})</span></div>
                <div>{constants.AGREEMENT_MODEL_USER_FACING_PROMPT}</div>
                <div><span class="tag tag-{agreement_model_html_class}">{agreement_model_classification_string}</span></div>
            </div>
        </div>
        
        <div class="separator"></div>

        <div class="section">
            <div class="section-title">Additional Information</div>
            <div class="field-grid">
                <div class="field-box">
                    <div class="field-label">File Access</div>
                    <div class="field-value">{files_in_consideration_message}</div>
                </div>
                
                <div class="field-box">
                    <div class="field-label">Date</div>
                    <div class="field-value">{submitted_time}</div>
                </div>
                
            </div>
        </div>

        <div class="section">
            <div class="section-title">Uncertainty (instance-level) Details</div>
            <div class="field-box" style="margin-bottom: 20px;">
                <div class="field-label">p(y | x)</div>
                <div class="field-value">{sdm_output}</div>
            </div>
            <div class="field-grid">
                <div class="field-box">
                    <div class="field-label">{constants.CALIBRATION_HIGH_RELIABILITY_REGION_LABEL_FULL}</div>
                    <div class="field-value">
                        <span class="tag tag-{is_high_reliability_region_html_class}">{is_high_reliability_region}</span>
                    </div>
                </div>

                <div class="field-box">
                    <div class="field-label">Out-of-Distribution</div>
                    <div class="field-value">
                        <span class="tag tag-{is_ood_html_class}">{is_ood}</span>
                    </div>
                </div>
                <div class="field-box">
                    <div class="field-label">Rescaled Similarity (q')</div>
                    <div class="field-value">{rescaled_similarity}</div>
                </div>
            </div>
            <div class="field-grid">
                <div class="field-box">
                    <div class="field-label">
                        {constants.qFull}
                    </div>
                    <div class="field-value">{similarity_q}</div>
                </div>

                <div class="field-box">
                    <div class="field-label">
                        {constants.dQuantileFull}
                    </div>
                    <div class="field-value">{distance_d}</div>
                </div>

                <div class="field-box">
                    <div class="field-label">
                        {constants.fFull}
                    </div>
                    <div class="field-value">{magnitude}</div>
                </div>
            </div>
        </div>

        <div class="section" style="margin-left: 40px;">
            <div class="section-title">Analysis of the Effective Sample Size</div>
            <div class="field-box" style="margin-bottom: 20px;">
                <div class="field-label">p(y | x)_lower</div>
                <div class="field-value">{sdm_output_d_lower}</div>
            </div>
            <div class="field-box" style="margin-bottom: 20px;">
                <div class="field-label">p(y | x)_upper</div>
                <div class="field-value">{sdm_output_d_upper}</div>
            </div>
            <div class="field-grid">
                <div class="field-box">
                    <div class="field-label">Effective Sample Size (by class)</div>
                    <div class="field-value">{cumulative_effective_sample_sizes}</div>
                </div>
            </div>
            <div class="field-grid">
                <div class="field-box">
                    <div class="field-label">
                        {constants.dQuantileLowerFull}
                    </div>
                    <div class="field-value">{distance_d_lower}</div>
                </div>

                <div class="field-box">
                    <div class="field-label">
                        {constants.dQuantileUpperFull}
                    </div>
                    <div class="field-value">{distance_d_upper}</div>
                </div>
            </div>
        </div>
        <div class="section">
            <div class="section-title">SDM Estimator (Model-level) Details</div>
            <div class="field-grid">
            
                <div class="field-box">
                    <div class="field-label">
                        α
                    </div>
                    <div class="field-value">{hr_class_conditional_accuracy}</div>
                </div>

                <div class="field-box">
                    <div class="field-label">
                        Minimum Rescaled Similarity (q'_min)
                    </div>
                    <div class="field-value">{min_rescaled_similarity_to_determine_high_reliability_region}</div>
                </div>

                <div class="field-box">
                    <div class="field-label">
                        Class-wise Output Thresholds (ψ)
                    </div>
                    <div class="field-value">{hr_output_thresholds}</div>
                </div>
                
                <div class="field-box">
                    <div class="field-label">
                        Support/training size
                    </div>
                    <div class="field-value">{support_index_ntotal}</div>
                </div>
            </div>
        </div>
        
        
        <div class="section">
            <div class="section-title">Prompt</div>
            <div class="prompt-box">{user_question}</div>
        </div>

        <div class="section">
            <div class="section-title">AI Response</div>
            <div class="document-box">{ai_response}</div>
        </div>
        
        <div class="separator"></div>
        
        {nearest_match_html_string}

        <div class="separator"></div>

        <div class="section">
            <div class="section-title">Legend</div>
            <div class="legend-content">
                <p>An ensemble of models 1, 2, and 3 (including the hidden states of model 3) is taken as the input to the SDM estimator that determines the verification classification.</p>
                <p>The classification is in the {constants.CALIBRATION_HIGH_RELIABILITY_REGION_LABEL_FULL_NON_TITLE} when the rescaled Similarity (q') is at least the minimum rescaled Similarity (q'_min) AND the predictive uncertainty, p(y | x), for the predicted class is at least the corresponding class-wise output threshold (ψ) for the predicted class.</p>
                <p>The estimates in the section 'Analysis of the Effective Sample Size' are based on the DKW inequality applied to the distance quantiles.</p>
                <div class="legend-items">
                    <div class="legend-item">
                        <span class="legend-label">Class 0:</span>
                        <span class="legend-value">{constants.MCP_SERVER_NOT_VERIFIED_CLASS_LABEL}</span>
                    </div>
                    <div class="legend-item">
                        <span class="legend-label">Class 1:</span>
                        <span class="legend-value">{constants.MCP_SERVER_VERIFIED_CLASS_LABEL}</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

    return html_content_string


def escape_html(content, content_is_html_escaped=False):
    if content_is_html_escaped:
        return content
    else:
        return html.escape(content)


def nearest_match_html(nearest_match_meta_data,
                       model1_name,
                       model2_name,
                       agreement_model_name,
                       content_is_html_escaped=False):
    # Use content_is_html_escaped=True when you know the fields have already been escaped with html.escape() (e.g.,
    # when retrieving from a database)
    predicted_class = nearest_match_meta_data.get("model_train_predicted_label", -1)
    true_label = nearest_match_meta_data.get("model_train_label", -1)

    successfully_verified = predicted_class == 1
    if successfully_verified:
        successfully_verified_html_class = "positive"
    else:
        successfully_verified_html_class = "negative"

    if true_label == 1:
        true_class_string_label = constants.MCP_SERVER_VERIFIED_CLASS_LABEL
        true_class_html_class = "positive"
    else:
        true_class_string_label = constants.MCP_SERVER_NOT_VERIFIED_CLASS_LABEL
        true_class_html_class = "negative"

    model1_classification = nearest_match_meta_data.get("model1_classification_int", -1) == 1
    model2_classification = nearest_match_meta_data.get("model2_classification_int", -1) == 1
    agreement_model_classification = \
        nearest_match_meta_data.get("agreement_model_classification_int", -1) == 1

    if agreement_model_classification:
        agreement_model_classification_string = "Yes"
    else:
        agreement_model_classification_string = "No"

    model1_html_class = "positive" if model1_classification else "negative"
    model2_html_class = "positive" if model2_classification else "negative"
    agreement_model_html_class = "positive" if agreement_model_classification else "negative"

    # Escape HTML, as applicable
    model1_explanation = escape_html(nearest_match_meta_data.get(constants.REEXPRESS_MODEL1_EXPLANATION, ''),
                                     content_is_html_escaped=content_is_html_escaped)
    model2_explanation = escape_html(nearest_match_meta_data.get(constants.REEXPRESS_MODEL2_EXPLANATION, ''),
                                     content_is_html_escaped=content_is_html_escaped)

    model1_summary = escape_html(nearest_match_meta_data.get(constants.REEXPRESS_MODEL1_TOPIC_SUMMARY, ''),
                                 content_is_html_escaped=content_is_html_escaped)

    user_question = escape_html(nearest_match_meta_data.get("user_question", ''),
                                content_is_html_escaped=content_is_html_escaped)
    ai_response = escape_html(nearest_match_meta_data.get(constants.REEXPRESS_AI_RESPONSE_KEY, ''),
                              content_is_html_escaped=content_is_html_escaped)

    document_id = escape_html(nearest_match_meta_data.get("document_id", ""),
                              content_is_html_escaped=content_is_html_escaped)
    document_source = escape_html(nearest_match_meta_data.get("document_source", ""),
                                  content_is_html_escaped=content_is_html_escaped)

    nearest_match_html_string = f"""
        <div class="nearest-match-box">
            <div class="section" style="margin-left: 40px;">
                <div class="section-title">Nearest Match in Training</div>
                
                <div class="field-grid">        
                    <div class="field-box" style="margin-bottom: 20px;">
                        <div class="field-label">Successfully Verified (Prediction)</div>
                        <div class="field-value">
                            <div class="field-value"><span class="tag tag-{successfully_verified_html_class}">{successfully_verified}</span></div>
                        </div>
                    </div>
        
                    <div class="field-box" style="margin-bottom: 20px;">
                        <div class="field-label">True Label</div>
                        <div class="field-value"><span class="tag tag-{true_class_html_class}">{true_class_string_label}</span></div>
                    </div>
                </div>
                
                <div class="explanation-box-{model1_html_class}">
                    <div class="explanation-title-{model1_html_class}">Model 1 Summary <span class="model-name">({model1_name})</span></div>
                    <div>{model1_summary}</div>
                </div>
                <div class="explanation-box-{model1_html_class}">
                    <div class="explanation-title-{model1_html_class}">Model 1 Explanation <span class="model-name">({model1_name})</span></div>
                    <div>{model1_explanation}</div>
                </div>
    
                <div class="explanation-box-{model2_html_class}">
                    <div class="explanation-title-{model2_html_class}">Model 2 Explanation <span class="model-name">({model2_name})</span></div>
                    <div>{model2_explanation}</div>
                </div>
                    
                <div class="explanation-box-{agreement_model_html_class}">
                    <div class="explanation-title-{agreement_model_html_class}">Model 3 Agreement <span class="model-name">({agreement_model_name})</span></div>
                    <div>{constants.AGREEMENT_MODEL_USER_FACING_PROMPT}</div>
                    <div><span class="tag tag-{agreement_model_html_class}">{agreement_model_classification_string}</span></div>
                </div>
                
                <div class="section">
                    <div class="section-title">Prompt</div>
                    <div class="prompt-box">{user_question}</div>
                </div>
    
                <div class="section">
                    <div class="section-title">AI Response</div>
                    <div class="document-box">{ai_response}</div>
                </div>
                <div class="field-grid">        
                    <div class="field-box" style="margin-bottom: 20px;">
                        <div class="field-label">Document ID</div>
                        <div class="field-value">{document_id}</div>
                    </div>
                    <div class="field-box" style="margin-bottom: 20px;">
                        <div class="field-label">Document Source</div>
                        <div class="field-value">{document_source}</div>
                    </div>                
                </div>
            </div>
        </div>
    """
    return nearest_match_html_string


def save_html_file(current_reexpression, nearest_match_meta_data=None, filename='reexpress_mcp_server_output.html'):
    """
    Saves the generated HTML to a file.

    Args:
        current_reexpression: Dictionary containing the neural model output
        filename: Name of the output HTML file
    """
    html_content = create_html_page(current_reexpression, nearest_match_meta_data=nearest_match_meta_data)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"HTML file saved as: {filename}")


def main():
    parser = argparse.ArgumentParser(description="-----[VISUALIZE]-----")
    parser.add_argument("--output_file", default="", help="")
    options = parser.parse_args()

    # Generate and save the HTML file
    save_html_file({}, {}, filename=options.output_file)


if __name__ == "__main__":
    main()
