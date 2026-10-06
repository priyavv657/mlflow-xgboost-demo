import gradio as gr
import joblib

# Load the trained model
model = joblib.load("model.pkl")


# Function that makes a prediction
def predict(feature1, feature2):
    prediction = model.predict([[feature1, feature2]])
    return int(prediction[0])


# Create the Gradio interface
app = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Feature 1"),
        gr.Number(label="Feature 2")
    ],
    outputs=gr.Number(label="Prediction"),
    title="XGBoost Prediction",
    description="Enter Feature 1 and Feature 2 to get a prediction."
)


import os

app.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
