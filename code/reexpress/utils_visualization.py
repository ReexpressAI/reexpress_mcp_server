# Copyright Reexpress AI, Inc. All rights reserved.

import torch
import html

import argparse
import constants
import mcp_utils_test
import utils_visualization__constants

def create_html_page(current_reexpression, nearest_match_meta_data=None):
    """
    Creates a static HTML page from model output dictionary.

    Args:
        current_reexpression: Dictionary containing the model output with keys for each field

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
    is_valid_index_conditional__lower = prediction_meta_data.get("is_valid_index_conditional__lower", False)
    is_ood_lower = prediction_meta_data.get("is_ood_lower", True)
    is_ood_lower_html_class = "positive" if not is_ood_lower else "negative"
    calibration_reliability = \
        mcp_utils_test.get_calibration_reliability_label(is_valid_index_conditional__lower, is_ood_lower)

    # Model Level
    try:
        non_odd_class_conditional_accuracy = prediction_meta_data["non_odd_class_conditional_accuracy"]
        min_valid_qbin_for_class_conditional_accuracy_with_bounded_error = \
            prediction_meta_data["min_valid_qbin_for_class_conditional_accuracy_with_bounded_error"]
        non_odd_thresholds = \
            prediction_meta_data["non_odd_thresholds"]
        support_index_ntotal = prediction_meta_data["support_index_ntotal"]
    except:
        non_odd_class_conditional_accuracy = 0.0
        min_valid_qbin_for_class_conditional_accuracy_with_bounded_error = "N/A"
        non_odd_thresholds = "N/A"
        support_index_ntotal = "N/A"

    classification_confidence, classification_confidence_html_class = \
        mcp_utils_test.get_calibration_confidence_label(calibration_reliability=calibration_reliability,
                                                        non_odd_class_conditional_accuracy=non_odd_class_conditional_accuracy,
                                                        return_html_class=True)

    model1_name = "gpt-4.1-2025-04-14"
    model2_name = "o4-mini-2025-04-16-high"
    model3_name = "gemini-2.5-pro"
    model4_name = "granite-3.3-8b-instruct"

    model1_classification = current_reexpression.get(constants.REEXPRESS_MODEL1_CLASSIFICATION, False)
    model2_classification = current_reexpression.get(constants.REEXPRESS_MODEL2_CLASSIFICATION, False)
    model3_classification = current_reexpression.get(constants.REEXPRESS_MODEL3_CLASSIFICATION, False)
    model4_agreement_classification = \
        current_reexpression.get(constants.REEXPRESS_AGREEMENT_MODEL_CLASSIFICATION, False)

    if model4_agreement_classification:
        agreement_model_classification_string = "Yes"
    else:
        agreement_model_classification_string = "No"

    model1_html_class = "positive" if model1_classification else "negative"
    model2_html_class = "positive" if model2_classification else "negative"
    model3_html_class = "positive" if model3_classification else "negative"
    model4_html_class = "positive" if model4_agreement_classification else "negative"

    # We escape HTML as it may be contained within the responses themselves
    model1_explanation = html.escape(current_reexpression.get(constants.REEXPRESS_MODEL1_EXPLANATION, ''))
    model2_explanation = html.escape(current_reexpression.get(constants.REEXPRESS_MODEL2_EXPLANATION, ''))
    model3_explanation = html.escape(current_reexpression.get(constants.REEXPRESS_MODEL3_EXPLANATION, ''))

    files_in_consideration_message = \
        mcp_utils_test.get_files_in_consideration_message(
            current_reexpression.get(constants.REEXPRESS_ATTACHED_FILE_NAMES, [])).strip()
    submitted_time = current_reexpression.get(constants.REEXPRESS_SUBMITTED_TIME_KEY, 'N/A')

    # Uncertainty
    try:
        rescaled_prediction_conditional_distribution__lower = \
            prediction_meta_data["rescaled_prediction_conditional_distribution__lower"].detach().numpy().tolist()
        is_valid_index_conditional__lower = prediction_meta_data["is_valid_index_conditional__lower"]
        is_valid_index_conditional__lower_html_class = "positive" if is_valid_index_conditional__lower else "negative"
        soft_qbin__lower = prediction_meta_data["soft_qbin__lower"][0].item()
        iterated_lower_offset__lower = prediction_meta_data["iterated_lower_offset__lower"]
        cumulative_effective_sample_sizes = \
            prediction_meta_data["cumulative_effective_sample_sizes"].detach().numpy().tolist()

        similarity_q = int(prediction_meta_data["original_q"])
        distance_d = torch.min(prediction_meta_data["distance_quantiles"]).item()
        magnitude = prediction_meta_data["f"].detach().numpy().tolist()

    except:
        rescaled_prediction_conditional_distribution__lower = "N/A"
        soft_qbin__lower = "N/A"
        iterated_lower_offset__lower = "N/A"
        is_valid_index_conditional__lower_html_class = "negative"
        cumulative_effective_sample_sizes = "N/A"
        similarity_q = "N/A"
        distance_d = "N/A"
        magnitude = "N/A"

    user_question = html.escape(current_reexpression.get(constants.REEXPRESS_QUESTION_KEY, ''))
    ai_response = html.escape(current_reexpression.get(constants.REEXPRESS_AI_RESPONSE_KEY, ''))

    # Nearest Match
    try:
        assert nearest_match_meta_data is not None
        nearest_match_html_string = nearest_match_html(nearest_match_meta_data,
                                                       model1_name,
                                                       model2_name,
                                                       model3_name,
                                                       model4_name)
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
            Reexpress MCP Server Output
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
                <div class="explanation-title-{model1_html_class}">Model 1 Explanation <span class="model-name">({model1_name})</span></div>
                <div>{model1_explanation}</div>
            </div>

            <div class="explanation-box-{model2_html_class}">
                <div class="explanation-title-{model2_html_class}">Model 2 Explanation <span class="model-name">({model2_name})</span></div>
                <div>{model2_explanation}</div>
            </div>

            <div class="explanation-box-{model3_html_class}">
                <div class="explanation-title-{model3_html_class}">Model 3 Explanation <span class="model-name">({model3_name})</span></div>
                <div>{model3_explanation}</div>
            </div>
            <div class="explanation-box-{model4_html_class}">
                <div class="explanation-title-{model4_html_class}">Model 4 Agreement <span class="model-name">({model4_name})</span></div>
                <div>{constants.AGREEMENT_MODEL_USER_FACING_PROMPT}</div>
                <div><span class="tag tag-{model4_html_class}">{agreement_model_classification_string}</span></div>
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
                <div class="field-label">p(y | x)_lower</div>
                <div class="field-value">{rescaled_prediction_conditional_distribution__lower}</div>
            </div>
            <div class="field-grid">
                <div class="field-box">
                    <div class="field-label">Valid Index-Conditional Estimate</div>
                    <div class="field-value">
                        <span class="tag tag-{is_valid_index_conditional__lower_html_class}">{is_valid_index_conditional__lower}</span>
                    </div>
                </div>

                <div class="field-box">
                    <div class="field-label">Out-of-Distribution</div>
                    <div class="field-value">
                        <span class="tag tag-{is_ood_lower_html_class}">{is_ood_lower}</span>
                    </div>
                </div>
                <div class="field-box">
                    <div class="field-label">Rescaled q_lower, <span style="font-family: 'Times New Roman', serif;">
                            (<span class="math-qtilde">q</span><sub style="font-size: 0.7em;">lower</sub>)
                        </span></div>
                    <div class="field-value">{soft_qbin__lower}</div>
                </div>
                <div class="field-box">
                    <div class="field-label">Iterated offset_lower (for class {predicted_class}), 
                        <span class="math-operator-m">
                            m
                            <span class="math-superscript-hat-y">ŷ</span>
                            <span class="math-subscript-floor">
                                ⌊<span class="qtilde-small">q</span>⌋
                            </span>
                        </span>
                    </div>
                    <div class="field-value">{iterated_lower_offset__lower}</div>
                </div>
                <div class="field-box">
                    <div class="field-label">Effective Sample Size (by class)</div>
                    <div class="field-value">{cumulative_effective_sample_sizes}</div>
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
                        {constants.dFull} Quantile
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

        <div class="section">
            <div class="section-title">SDM Estimator (Model-level) Details</div>
            <div class="field-grid">
            
                <div class="field-box">
                    <div class="field-label">
                        α'
                    </div>
                    <div class="field-value">{non_odd_class_conditional_accuracy}</div>
                </div>

                <div class="field-box">
                    <div class="field-label">
                        Min valid rescaled q
                        <span style="font-family: 'Times New Roman', serif;">
                            (<span class="math-qtilde">q</span><sup style="font-size: 0.7em;">γ</sup><sub style="font-size: 0.7em;">min</sub>)
                        </span>
                    </div>
                    <div class="field-value">{min_valid_qbin_for_class_conditional_accuracy_with_bounded_error}</div>
                </div>

                <div class="field-box">
                    <div class="field-label">
                        Class-wise output thresholds (ψ)
                    </div>
                    <div class="field-value">{non_odd_thresholds}</div>
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
                <p>An ensemble of models 1, 2, 3, and 4 (including the hidden states of model 4) is taken as the input to the SDM estimator that determines the verification classification.</p>
                
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


def nearest_match_html(nearest_match_meta_data,
                       model1_name,
                       model2_name,
                       model3_name,
                       model4_name):

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
    model3_classification = nearest_match_meta_data.get("model3_classification_int", -1) == 1
    model4_agreement_classification = \
        nearest_match_meta_data.get("model4_agreement_classification_int", -1) == 1

    if model4_agreement_classification:
        agreement_model_classification_string = "Yes"
    else:
        agreement_model_classification_string = "No"

    model1_html_class = "positive" if model1_classification else "negative"
    model2_html_class = "positive" if model2_classification else "negative"
    model3_html_class = "positive" if model3_classification else "negative"
    model4_html_class = "positive" if model4_agreement_classification else "negative"

    # We escape HTML as it may be contained within the responses themselves
    model1_explanation = html.escape(nearest_match_meta_data.get(constants.REEXPRESS_MODEL1_EXPLANATION, ''))
    model2_explanation = html.escape(nearest_match_meta_data.get(constants.REEXPRESS_MODEL2_EXPLANATION, ''))
    model3_explanation = html.escape(nearest_match_meta_data.get(constants.REEXPRESS_MODEL3_EXPLANATION, ''))

    user_question = html.escape(nearest_match_meta_data.get("user_question", ''))
    ai_response = html.escape(nearest_match_meta_data.get(constants.REEXPRESS_AI_RESPONSE_KEY, ''))

    document_id = nearest_match_meta_data.get("document_id", "")
    document_source = nearest_match_meta_data.get("document_source", "")

    if document_source != "openthoughts.o4mini_high":
        # The support documents from version 1.0.0 used a medium thinking budget
        model2_name = "o4-mini-2025-04-16-medium"

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
                    <div class="explanation-title-{model1_html_class}">Model 1 Explanation <span class="model-name">({model1_name})</span></div>
                    <div>{model1_explanation}</div>
                </div>
    
                <div class="explanation-box-{model2_html_class}">
                    <div class="explanation-title-{model2_html_class}">Model 2 Explanation <span class="model-name">({model2_name})</span></div>
                    <div>{model2_explanation}</div>
                </div>
    
                <div class="explanation-box-{model3_html_class}">
                    <div class="explanation-title-{model3_html_class}">Model 3 Explanation <span class="model-name">({model3_name})</span></div>
                    <div>{model3_explanation}</div>
                </div>
                
                <div class="explanation-box-{model4_html_class}">
                    <div class="explanation-title-{model4_html_class}">Model 4 Agreement <span class="model-name">({model4_name})</span></div>
                    <div>{constants.AGREEMENT_MODEL_USER_FACING_PROMPT}</div>
                    <div><span class="tag tag-{model4_html_class}">{agreement_model_classification_string}</span></div>
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
