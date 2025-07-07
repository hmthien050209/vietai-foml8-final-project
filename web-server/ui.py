import gradio as gr
import datetime
import http.client
import json

SERVER_URL = "localhost"
SERVER_PORT = 8000

def get_model_info_display():
    conn = http.client.HTTPConnection(SERVER_URL, SERVER_PORT)
    conn.request("GET", "/model_info")
    res = conn.getresponse()
    model_info = json.loads(res.read())
    
    f1 = model_info["f1"]
    precision = model_info["precision"]
    recall = model_info["recall"]
    # Convert Unix timestamp to human-readable format
    dt_object = datetime.datetime.fromtimestamp(model_info["last_updated"])
    last_updated_str = dt_object.strftime("%d/%m/%Y %H:%M:%S")

    return (
        f"F1 score: {f1:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"Last updated: {last_updated_str}"
    )

def train_model():
    conn = http.client.HTTPConnection(SERVER_URL, SERVER_PORT)
    conn.request("POST", "/train")
    res = conn.getresponse()
    model_info = json.loads(res.read())
    
    model_info["last_updated"] = datetime.datetime.now().timestamp()
    return "Model trained!", get_model_info_display()

def predict_heart_disease(
        age: int,
        dataset: str,
        sex: str,
        cp: str,
        trestbps: float,
        chol: float,
        fbs: bool,
        restecg: str,
        thalch: float,
        exang: bool,
        oldpeak: float,
        slope: str
) -> str:
    conn = http.client.HTTPConnection(SERVER_URL, SERVER_PORT)
    payload = {
        "age": age,
        "dataset": dataset,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalch": thalch,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,   
    }

    headers = { 'Content-Type': "application/json" }

    conn.request("POST", "/predict", json.dumps(payload), headers)

    res = conn.getresponse()
    data = json.loads(res.read())
    
    return data['result']


with gr.Blocks(title="Heart Disease Prediction") as app:
    gr.Markdown("# Heart Disease Prediction")

    with gr.Tab("Train"):
        gr.Markdown("## Model Information")
        
        model_info_output = gr.Textbox(
            label="Model Info",
            value=get_model_info_display(),
            interactive=False,
            lines=4
        )
        train_button = gr.Button("Train Model")
        train_status_output = gr.Textbox(
            label="Training Status",
            interactive=False
        )

        train_button.click(
            fn=train_model,
            inputs=[],
            outputs=[train_status_output, model_info_output]
        )

    with gr.Tab("Predict"):
        gr.Markdown("## Predict Heart Disease")
        with gr.Row():
            age_input = gr.Slider(
                minimum=1,
                maximum=100,
                step=1,
                label="Age (years)",
                value=50
            )
            dataset_input = gr.Textbox(
                label="Origin (e.g., Cleveland, Hungarian)",
                placeholder="Enter dataset origin",
                value="Cleveland"
            )
        with gr.Row():
            sex_input = gr.Radio(
                choices=["Male", "Female"],
                label="Sex",
                value="Male"
            )
            cp_input = gr.Dropdown(
                choices=["typical angina", "atypical angina", "non-anginal", "asymptomatic"],
                label="Chest pain type",
                value="typical angina"
            )
        with gr.Row():
            trestbps_input = gr.Number(
                label="Resting blood pressure (mmHg)",
                value=120.0
            )
            chol_input = gr.Number(
                label="Serum cholesterol (mg/dl)",
                value=200.0
            )
        with gr.Row():
            fbs_input = gr.Dropdown(
                label="Fasting blood sugar (> 120mg/dl)?",
                choices=[True, False],
                value=False
            )
            restecg_input = gr.Dropdown(
                choices=["normal", "st-t abnormality", "lv hypertrophy"],
                label="Resting ECG results",
                value="normal"
            )
        with gr.Row():
            thalch_input = gr.Number(
                label="Max heart rate achieved",
                value=150.0
            )
            exang_input = gr.Dropdown(
                label="Exercise-induced angina",
                choices=[True, False],   
                value=False
            )
        with gr.Row():
            oldpeak_input = gr.Number(
                label="ST depression induced by exercise relative to rest",
                value=1.0
            )
            slope_input = gr.Dropdown(
                choices=["upsloping", "flat", "downsloping"],
                label="Slope of the peak exercise ST segment",
                value="flat"
            )

        predict_button = gr.Button("Predict Heart Disease")
        output_prediction = gr.Textbox(
            label="Has heart disease?",
            interactive=False
        )

        predict_button.click(
            fn=predict_heart_disease,
            inputs=[
                age_input,
                dataset_input,
                sex_input,
                cp_input,
                trestbps_input,
                chol_input,
                fbs_input,
                restecg_input,
                thalch_input,
                exang_input,
                oldpeak_input,
                slope_input
            ],
            outputs=output_prediction
        )

# To run the Gradio app, use:
if __name__ == "__main__":
    app.launch()
