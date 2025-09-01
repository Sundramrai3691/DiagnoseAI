# DiagnoseAI

DiagnoseAI is a modern Flask-based web application that provides intelligent health diagnostics using advanced NLP and machine learning models. The application features a clean, minimalist interface with an AI assistant named "Meddy" for personalized health assessments.

## ✨ Features

- **Smart Symptom Analysis**: Advanced ML models analyze symptoms for accurate diagnosis suggestions
- **Interactive AI Chat**: Conversational interface with Meddy for general health queries
- **PDF Report Generation**: Comprehensive medical reports for healthcare professionals
- **Modern UI**: Clean, responsive design with accessibility features
- **Voice Input**: Speech recognition for hands-free symptom input
- **Spell Correction**: Intelligent correction and semantic similarity for improved query handling

## 🎨 UI Design (2025 Update)

The application now features a modern, minimalist design with:
- **Design System**: CSS custom properties for consistent theming
- **Typography**: Inter font family with proper font weights
- **Responsive Layout**: Mobile-first design that works on all devices
- **Accessibility**: WCAG AA compliant with proper ARIA labels and keyboard navigation
- **Dark Mode**: Toggle between light and dark themes
- **Micro-interactions**: Subtle animations and loading states

## 🛠️ Tech Stack

- **Backend**: Python (Flask)
- **AI/ML**: Hugging Face Transformers, PyTorch (CPU)
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Frontend**: Modern vanilla JavaScript, CSS Grid/Flexbox
- **Deployment**: Gunicorn (production ready)

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- pip package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Sundramrai3691/DiagnoseAI.git
   cd DiagnoseAI
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file in the root directory:
   ```env
   HUGGINGFACE_API_KEY=your_api_key_here
   ```

4. **Run the application:**
   
   **Development:**
   ```bash
   python app.py
   ```
   
   **Production:**
   ```bash
   gunicorn app:app
   ```

5. **Access the application:**
   Open your browser and navigate to `http://127.0.0.1:5000`

## 📱 Usage

1. **Start Assessment**: Enter your personal information (name, age, gender)
2. **Describe Symptoms**: Type or speak your symptoms (e.g., "fever, headache, fatigue")
3. **Get Diagnosis**: Receive AI-powered analysis with confidence scores and severity levels
4. **View Recommendations**: See personalized precautions and health advice
5. **Download Report**: Generate PDF report to share with healthcare professionals
6. **Ask Questions**: Use the AI chat for general health queries

## 🏗️ Project Structure

```
DiagnoseAI/
├── app.py                 # Main Flask application
├── nnet.py               # Neural network model
├── nltk_utils.py         # NLP utilities
├── requirements.txt      # Python dependencies
├── data/                 # Medical datasets
│   ├── symptom_Description.csv
│   ├── symptom_precaution.csv
│   └── Symptom-severity.csv
├── models/               # Trained ML models
├── static/               # Frontend assets
│   ├── css/styles.css   # Modern CSS with design tokens
│   ├── js/main.js       # Enhanced JavaScript
│   └── img/logo.svg     # Vector logo
├── templates/            # HTML templates
│   ├── index.html       # Main interface
│   └── report.html      # PDF report template
└── logs/                # Application logs
```

## 🎯 Key Features Explained

### Symptom Analysis Pipeline
1. **Input Processing**: Natural language processing with spell correction
2. **Feature Extraction**: Bag-of-words model with stemming
3. **Classification**: Neural network predicts symptom categories
4. **Disease Prediction**: Machine learning model suggests likely conditions
5. **Confidence Scoring**: Provides reliability metrics for each prediction

### AI Chat Assistant
- Powered by Hugging Face's FLAN-T5 model
- Handles general health queries and lifestyle advice
- Contextual responses based on medical knowledge

### Report Generation
- Comprehensive PDF reports with patient information
- Diagnosis summary with confidence scores
- Recommended precautions and next steps
- Professional formatting for healthcare providers

## 🔧 Configuration

### Environment Variables
- `HUGGINGFACE_API_KEY`: Required for AI chat functionality
- `FLASK_ENV`: Set to `development` for debug mode

### Model Files
The application requires pre-trained models in the `models/` directory:
- `data.pth`: Neural network weights and vocabulary
- `fitted_model.pickle2`: Disease prediction model

## 🚀 Deployment

### Heroku
```bash
git push heroku main
```

### Render
The included `render.yaml` configures automatic deployment.

### Docker
```bash
docker build -t diagnoseai .
docker run -p 5000:5000 diagnoseai
```

## 🧪 Testing

### Manual Verification Checklist
- [ ] App starts with `python app.py` or `gunicorn app:app`
- [ ] User registration and symptom submission works
- [ ] Diagnosis results display correctly with new UI
- [ ] PDF report generation functions properly
- [ ] AI chat responds to queries
- [ ] Responsive design works on mobile (≤360px) and desktop (≥1200px)
- [ ] Dark mode toggle functions correctly
- [ ] Voice input works in supported browsers
- [ ] Keyboard navigation is accessible
- [ ] No console errors or broken static assets

### Browser Compatibility
- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Mobile browsers (iOS Safari, Chrome Mobile)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Sundram Rai**
- GitHub: [@Sundramrai3691](https://github.com/Sundramrai3691)

## 🙏 Acknowledgments

- Hugging Face for transformer models
- Medical datasets from various open sources
- Flask community for excellent documentation
- Contributors and testers

---

**⚠️ Medical Disclaimer**: This application is for educational and informational purposes only. It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers with questions about medical conditions.