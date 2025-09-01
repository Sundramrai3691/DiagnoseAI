import os
import json
import requests

import torch
import pickle
import nltk
import random
import warnings
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, session, make_response
from fpdf import FPDF
from io import BytesIO
from xhtml2pdf import pisa
from rapidfuzz import process
from dotenv import load_dotenv
from textblob import TextBlob
from symspellpy import SymSpell, Verbosity
from importlib.metadata import version as get_version  # ✅ Replacing pkg_resources

from nnet import NeuralNet
from nltk_utils import bag_of_words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# 🧠 NLTK Data Check
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

warnings.filterwarnings("ignore", category=FutureWarning)

# 🔐 Load environment
load_dotenv()

app = Flask(__name__)
app.secret_key = "super-secret-key"
device = torch.device('cpu')
random.seed(datetime.now().timestamp())

# 🛠️ SymSpell Setup
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = os.path.join(os.path.dirname(__file__), "symspellpy", "frequency_dictionary_en_82_765.txt")
if not os.path.exists(dictionary_path):  # fallback
    from symspellpy import __file__ as symspell_path
    dictionary_path = os.path.join(os.path.dirname(symspell_path), "frequency_dictionary_en_82_765.txt")

sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# 🔍 Logging Setup
if not os.path.exists("logs"):
    os.makedirs("logs")
logging.basicConfig(
    filename="logs/unrecognized_symptoms.txt",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# 📦 Load Model
model_data = torch.load("models/data.pth")
input_size = model_data['input_size']
hidden_size = model_data['hidden_size']
output_size = model_data['output_size']
all_words = model_data['all_words']
tags = model_data['tags']
model_state = model_data['model_state']

nlp_model = NeuralNet(input_size, hidden_size, output_size).to(device)
nlp_model.load_state_dict(model_state)
nlp_model.eval()

# 🧬 Load Data
diseases_description = pd.read_csv("data/symptom_Description.csv")
diseases_description['Disease'] = diseases_description['Disease'].str.lower().str.strip()

disease_precaution = pd.read_csv("data/symptom_precaution.csv")
disease_precaution['Disease'] = disease_precaution['Disease'].str.lower().str.strip()

symptom_severity = pd.read_csv("data/Symptom-severity.csv")
symptom_severity = symptom_severity.apply(
    lambda col: col.map(lambda s: s.lower().strip().replace(" ", "") if isinstance(s, str) else s)
)

with open('data/list_of_symptoms.pickle', 'rb') as f:
    symptoms_list = pickle.load(f)

with open('models/fitted_model.pickle2', 'rb') as f:
    prediction_model = pickle.load(f)

# 🧑 User session state
user_profile = {}
user_symptoms = {}

# 🔧 Correction Logic
def correct_input_with_llm(user_input):
    suggestions = sym_spell.lookup_compound(user_input, max_edit_distance=2)
    corrected = suggestions[0].term if suggestions else user_input
    corrected = str(TextBlob(corrected).correct())
    if corrected != user_input:
        print(f"🔧 Corrected: '{user_input}' → '{corrected}'")
    return corrected

# 🔍 Fuzzy Matching
def fuzzy_match_symptom(phrase, threshold=85):
    match, score, _ = process.extractOne(phrase, symptoms_list)
    return match if score >= threshold else None

# 🔎 Classify Symptom
def get_symptom(sentence):
    tokens = nltk.word_tokenize(sentence)
    X = bag_of_words(tokens, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)
    output = nlp_model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    prob = torch.softmax(output, dim=1)[0][predicted.item()].item()
    return tag, prob

# 📄 Report Generator
def generate_pdf_report(user_profile, disease, description, precautions):
    if not os.path.exists("static/reports"):
        os.makedirs("static/reports")
    filename = f"static/reports/report_{user_profile.get('name', 'user')}.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="DiagnoseAI - Medical Report", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Name: {user_profile.get('name')} | Age: {user_profile.get('age')} | Gender: {user_profile.get('gender')}", ln=True)
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt=f"Predicted Disease: {disease.title()}\n\nDescription:\n{description}\n\nPrecautions:\n{precautions}")
    pdf.output(filename)
    return filename

# 🔗 Routes
@app.route("/")
def index():
    user_symptoms.clear()
    user_profile.clear()
    return render_template("index.html", data=symptoms_list)

@app.route("/start", methods=["POST"])
def set_user_profile():
    data = request.get_json()
    user_profile['name'] = data.get("name", "User")
    user_profile['age'] = data.get("age", "N/A")
    user_profile['gender'] = data.get("gender", "N/A")
    return jsonify({"msg": f"Welcome {user_profile['name']}! You can now start entering your symptoms."})

@app.route("/symptom", methods=["POST"])
def predict_symptom():
    raw_input = request.json.get("sentence", "").strip().lower()

    if raw_input in {"done", "done.", "done!"}:
        if not user_symptoms:
            return jsonify("⚠️ Please enter at least one symptom before typing 'done'.")

        x_test = [1 if s in user_symptoms else 0 for s in symptoms_list]
        disease = prediction_model.predict(np.asarray(x_test).reshape(1, -1))[0].strip().lower()

        try:
            description = diseases_description.loc[diseases_description['Disease'] == disease, 'Description'].iloc[0]
            precaution_row = disease_precaution[disease_precaution['Disease'] == disease]
            precautions = ", ".join([
                precaution_row['Precaution_1'].iloc[0],
                precaution_row['Precaution_2'].iloc[0],
                precaution_row['Precaution_3'].iloc[0],
                precaution_row['Precaution_4'].iloc[0]
            ])

            result = []
            for s, conf in user_symptoms.items():
                severity = symptom_severity.loc[symptom_severity['Symptom'] == s.replace(" ", ""), 'weight'].iloc[0]
                result.append({
                    "symptom": s,
                    "confidence": conf,
                    "severity": int(severity),
                    "description": description,
                    "precautions": precautions.split(", "),
                    "disease": disease.title()
                })

            session['report_data'] = {
                "name": user_profile.get('name', 'User'),
                "age": int(user_profile.get('age', 0)),
                "gender": user_profile.get('gender', 'N/A'),
                "disease": disease,
                "description": description,
                "precautions": precautions.split(", "),
                "date": datetime.now().strftime("%d %B %Y"),
            }

            generate_pdf_report(user_profile, disease, description, precautions)
            user_symptoms.clear()
            return jsonify(result)

        except Exception as e:
            return jsonify(f"An error occurred while generating the report. ({str(e)})")

    tokens = word_tokenize(raw_input)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if w.isalpha() and w not in stop_words]

    responses = []
    for phrase in filtered_tokens:
        corrected = correct_input_with_llm(phrase)
        if corrected != phrase:
            responses.append(f"🤖 Did you mean '<b>{corrected}</b>'? Please confirm or retype.")
            continue

        symptom, prob = get_symptom(corrected)
        if prob > 0.5:
            user_symptoms[symptom] = prob
            responses.append(f"✅ Symptom noted: '<b>{symptom}</b>' ({prob * 100:.1f}% confidence)")
        else:
            fuzzy = fuzzy_match_symptom(phrase)
            if fuzzy:
                user_symptoms[fuzzy] = 0.6
                responses.append(f"🤖 Typo detected. Added '<b>{fuzzy}</b>' (fuzzy match)")
            else:
                responses.append(f"❌ Couldn’t recognize: '<b>{phrase}</b>'")
                logging.info(f"Unrecognized: {phrase}")

    return jsonify("<br>".join(responses))

@app.route("/download_report", methods=["GET"])
def download_report():
    filename = f"static/reports/report_{user_profile.get('name', 'user')}.pdf"
    if os.path.exists(filename):
        return send_file(filename, as_attachment=True)
    return "Report not found", 404

@app.route("/download_pdf")
def download_pdf():
    pdf_html = render_template("report.html", data=session.get("report_data", {}))

    # ✅ Replace CSS variables with actual hex values
    pdf_html = pdf_html.replace("var(--border)", "#000000")      # black
    pdf_html = pdf_html.replace("var(--text)", "#333333")        # dark gray
    pdf_html = pdf_html.replace("var(--background)", "#ffffff")  # white

    pdf = BytesIO()
    pisa_status = pisa.CreatePDF(pdf_html, dest=pdf)
    if pisa_status.err:
        return "PDF generation failed", 500

    pdf.seek(0)
    response = make_response(pdf.read())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=DiagnoseAI_report.pdf'
    return response


# 🤖 Ask DiagnoseAI Anything Route
# 🤖 Ask DiagnoseAI Anything Route
@app.route("/ask_ai", methods=["POST"])
def ask_ai():
    data = request.get_json()
    question = data.get("question") or data.get("message")

    if not question or not question.strip():
        return jsonify({"reply": "❌ Please enter a valid health-related query."})

    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    if not HUGGINGFACE_API_KEY:
        return jsonify({"reply": "❌ Hugging Face API key not found in server environment."})

    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}"
    }
    payload = {
        "inputs": question.strip()
    }

    try:
        hf_response = requests.post(
            "https://api-inference.huggingface.co/models/google/flan-t5-small",
            headers=headers,
            json=payload,
            timeout=15
        )

        if hf_response.status_code == 200:
            output = hf_response.json()
            reply = ""
            if isinstance(output, list) and len(output) > 0:
                reply = output[0].get("generated_text", "").strip()
            elif isinstance(output, dict) and "generated_text" in output:
                reply = output["generated_text"].strip()
            else:
                reply = str(output)
            return jsonify({"reply": reply})
        else:
            return jsonify({"reply": f"⚠️ Hugging Face API error ({hf_response.status_code})"})

    except Exception as e:
        return jsonify({"reply": f"⚠️ Error contacting AI model: {str(e)}"})


# 🚀 Run Server
if __name__ == "__main__":
    print(f"🚀 DiagnoseAI DiagnoseAI running at: http://127.0.0.1:5000")
    app.run(debug=True, host="127.0.0.1", port=5000, use_reloader=False)
