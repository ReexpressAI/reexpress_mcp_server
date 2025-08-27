# Training and Calibration Data

The training and calibration data is a subset of the [OpenVerification1](https://huggingface.co/datasets/ReexpressAI/OpenVerification1) dataset available on HuggingFace. The text of the training data appearing in the support set is available in reexpress_mcp_server_db/reexpress_mcp_server_support_documents.db in the model directory. These correspond to the document_id's stored in the class property self.train_uuids of the SimilarityDistanceMagnitudeCalibrator() model (see code/reexpress/sdm_model.py), and the corresponding calibration set document_id's are stored in the class property self.calibration_uuids. The model also stores the corresponding ground-truth labels and predictions for the final training/calibration split. (The SDM estimator is itself trained by iteratively shuffling the data, so the particular final split of the data into training and calibration sets is model-dependent.)

For reference, if you want to pull up the row in OpenVerification1 for a particular document (e.g., when using the interactive graphs), the following can be used:

```python
from datasets import load_dataset
dataset = load_dataset("ReexpressAI/OpenVerification1")
def retrieve_row_by_id(document_id: str):
    for split_name in ["eval", "validation", "train"]:
        filtered_dataset = dataset[split_name].filter(lambda x: x['id'] == document_id)
        if filtered_dataset.num_rows == 1:
            print(filtered_dataset[0])
            return filtered_dataset
``` 
