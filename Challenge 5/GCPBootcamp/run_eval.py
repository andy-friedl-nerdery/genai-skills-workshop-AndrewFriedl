from vertexai.evaluation import EvalTask, PairwiseMetric # <<< Requires vertexai SDK
import pandas as pd
from google.cloud import aiplatform
import os

# --- Configuration ---
# Uses the environment variable set outside of the script
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
OUTPUT_BUCKET = "gs://qwiklabs-gcp-03-b295c10c44aa-eval-bucket"  # Must exist in your project

if not PROJECT_ID:
    raise EnvironmentError("GOOGLE_CLOUD_PROJECT environment variable is not set.")

# Initialize the Vertex AI SDK
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# --- 1. Define Input Data (Pre-generated Outputs) ---
evaluation_inputs = [
    {
        "input": "The city council approved a 5% property tax increase effective Jan 1st.",
        "output_a": "Detailed announcement about the 5% property tax increase.",
        "output_b": "Concise social media post: 5% property tax hike approved."
    },
    {
        "input": "Due to a water main break, the intersection of Main and 5th street is closed indefinitely. Use detours.",
        "output_a": "Formal notice: Main & 5th closed due to water main break.",
        "output_b": "Social media alert: Main & 5th closed. Detours in effect."
    }
]

dataset_df = pd.DataFrame(evaluation_inputs)

# --- 2. Define Metric (Automated Pairwise Comparison) ---
pairwise_metric = PairwiseMetric(
    metric="pairwise",
    # The scoring model uses this template to judge the outputs
    metric_prompt_template="Which output is more authoritative and concise? 0=worst, 1=best."
)

# --- 3. Create Evaluation Task ---
eval_task = EvalTask(
    dataset=dataset_df,
    metrics=[pairwise_metric],
    output_uri_prefix=OUTPUT_BUCKET
)

# --- 4. Run the Evaluation ---
# The evaluate method uses a powerful foundation model to score the outputs based on the metric_prompt_template.
results = eval_task.evaluate()

print("✅ Evaluation complete.")
print(f"View raw results in GCS: {OUTPUT_BUCKET}")

# --- 5. Access and Analyze Results ---
print("\n📊 Summary Metrics:")
print(results.summary_metrics)

print("\n📋 Detailed Metrics Table:")
metrics_df = results.metrics_table
print(metrics_df[['input', 'output_a', 'output_b', 'pairwise/pairwise_choice', 'pairwise/explanation']])

# Method to save to CSV for later analysis
metrics_df.to_csv("evaluation_results.csv", index=False)
print("\n💾 Results saved to evaluation_results.csv")