# DiagnoseAI

DiagnoseAI is a Flask-based web application that provides intelligent health diagnostics using NLP and machine learning models.

## Features

- Symptom analysis and diagnosis suggestion
- Medical chatbot for general health queries
- PDF report generation
- Spell correction and semantic similarity for improved query handling

## Tech Stack

- Python (Flask)
- Hugging Face Transformers
- PyTorch (CPU)
- Pandas, NumPy, Scikit-learn
- Gunicorn (for deployment)

## Installation

1. Clone the repository:
   
git clone https://github.com/Sundramrai3691/DiagnoseAI.git
cd DiagnoseAI

2. Install dependencies:
   
3. Set environment variables:
- Create a `.env` file and add your keys:
  ```
  HUGGINGFACE_API_KEY=your_api_key_here
  ```

4. Run the application:
gunicorn app:app
