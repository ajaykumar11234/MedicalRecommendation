from flask import Flask, request, render_template, jsonify  # Import jsonify
import numpy as np
import pandas as pd
import pickle
import os

# Initialize Flask app
app = Flask(__name__)

# Define the base dataset path
base_path = r"C:\Users\ajayk\OneDrive\Desktop\MedicalML\datasets"

# Function to check if a file exists
def load_csv(filename):
    path = os.path.join(base_path, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        print(f"⚠️ Warning: {filename} not found!")
        return None

# Load datasets
sym_des = load_csv("symtoms_df.csv")
precautions = load_csv("precautions_df.csv")
workout = load_csv("workout_df.csv")
description = load_csv("description.csv")
medications = load_csv("medications.csv")
diets = load_csv("diets.csv")

# Load trained model
model_path = os.path.join("models", "svc.pkl")
if os.path.exists(model_path):
    svc = pickle.load(open(model_path, "rb"))
else:
    print("⚠️ Error: Model file svc.pkl not found!")
    svc = None  # Prevent crashing if model isn't available

# Helper function to fetch details for a disease
def helper(dis):
    # Check if disease exists in description
    desc = description[description['Disease'] == dis]['Description']
    desc = desc.iloc[0] if not desc.empty else "No description available."

    # Get precautions
    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = pre.values.flatten().tolist() if not pre.empty else ["No precautions available."]

    # Get medications
    med = medications[medications['Disease'] == dis]['Medication']
    med = med.tolist() if not med.empty else ["No medications available."]

    # Get diet recommendations
    die = diets[diets['Disease'] == dis]['Diet']
    die = die.tolist() if not die.empty else ["No diet recommendations available."]

    # Get workout recommendations
    if {'disease', 'workout'}.issubset(workout.columns):
        wrkout = workout[workout['disease'] == dis]['workout']
        wrkout = wrkout.tolist() if not wrkout.empty else ["No workout recommendations available."]
    else:
        print("⚠️ Warning: Column names in workout_df.csv might be incorrect!")
        wrkout = ["No workout recommendations available."]

    return desc, pre, med, die, wrkout



# Symptoms dictionary (unchanged)
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9}
# Truncated for brevity...

diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis'}
# Truncated for brevity...

# Function to predict disease
def get_predicted_value(patient_symptoms):
    if svc is None:
        return "Model not loaded. Prediction unavailable."

    input_vector = np.zeros(132) 

    for item in patient_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
        else:
            print(f"⚠️ Warning: Symptom '{item}' not recognized.")
    return diseases_list.get(svc.predict([input_vector])[0], "Unknown disease")

# Flask routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def home():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms', '').strip()

        if not symptoms:
            message = "⚠️ Please enter symptoms!"
            return render_template('index.html', message=message)

        # Process user symptoms
        user_symptoms = [s.strip() for s in symptoms.split(',')]
        predicted_disease = get_predicted_value(user_symptoms)

        # Get details of the predicted disease
        dis_des, my_precautions, medications, my_diet, workout = helper(predicted_disease)

        return render_template('index.html', predicted_disease=predicted_disease, dis_des=dis_des,
                               my_precautions=my_precautions, medications=medications, my_diet=my_diet, workout=workout)
    return render_template('index.html')

# Additional routes
@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/developer')
def developer():
    return render_template("developer.html")

@app.route('/blog')
def blog():
    return render_template("blog.html")

if __name__ == '__main__':
    app.run(debug=True)
