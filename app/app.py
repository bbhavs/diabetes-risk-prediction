import gradio as gr
import joblib
import pandas as pd

# Load trained pipeline
model = joblib.load("xgboost_diabetes_model.pkl")

# Print what features the model expects
print("Model expects these features:")
try:
    print(model.named_steps["preprocess"].feature_names_in_)
except:
    print("Could not access feature_names_in_")

def predict_diabetes(
    age,
    waist,
    diastolic_bp,
    systolic_bp,
    crp,
    hemoglobin,
    vigorous_activity,
    moderate_activity,
    alcohol,
    hypertension,
    father_diabetes,
    mother_diabetes,
    idrs
):

    input_data = pd.DataFrame([{
        "dm005": age,
        "bm076": waist,
        "bm017": diastolic_bp,
        "bm018": systolic_bp,
        "crp": crp,
        "hb": hemoglobin,
        "hb211": vigorous_activity,
        "hb213": moderate_activity,
        "hb101": alcohol,
        "fm303": hypertension,
        "fm304s1": father_diabetes,
        "fm304s2": mother_diabetes,
        "IDRS": idrs
    }])

    print("\nInput Data:")
    print(input_data)

    # Get full probability output
    probabilities = model.predict_proba(input_data)

    print("Raw Model Output:")
    print(probabilities)

    prob = probabilities[0][1]

    if prob < 0.30:
        risk_level = "Low Risk"
    elif prob < 0.60:
        risk_level = "Moderate Risk"
    else:
        risk_level = "High Risk"

    return f"Predicted Probability of Diabetes: {prob:.4f}\nRisk Category: {risk_level}"


interface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="Waist Circumference (cm)"),
        gr.Number(label="Diastolic BP"),
        gr.Number(label="Systolic BP"),
        gr.Number(label="CRP (mg/L)"),
        gr.Number(label="Hemoglobin (g/dL)"),
        gr.Number(label="Vigorous Activity (0=No, 1=Yes)"),
        gr.Number(label="Moderate Activity (0=No, 1=Yes)"),
        gr.Number(label="Alcohol Consumption (0=No, 1=Yes)"),
        gr.Number(label="Hypertension (0=No, 1=Yes)"),
        gr.Number(label="Father Has Diabetes (0=No, 1=Yes)"),
        gr.Number(label="Mother Has Diabetes (0=No, 1=Yes)"),
        gr.Number(label="IDRS Score")
    ],
    outputs="text",
    title="Diabetes Risk Prediction Tool (Prototype)",
    description="XGBoost-based diabetes risk prediction model developed using LASI dataset."
)

interface.launch()
