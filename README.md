# Thorax AI – Intelligent Clinical Decision Support System (ICDSS)

Thorax AI is a Streamlit-based clinical decision support system for detecting thoracic pathologies from chest X-ray images using deep learning.

## 🚀 Features

- Upload X-ray images (JPG, PNG, DICOM)
- Multi-label pathology prediction (14 classes)
- Grad-CAM explainability heatmaps
- AI-assisted clinical report generation
- PDF report export
- Multi-patient session tracking

## 🧠 Model

- Architecture: CNN-based (e.g. DenseNet / ResNet / EfficientNet)
- Input size: 224 × 224
- Output: 14 thoracic pathologies (multi-label)

## 📂 Project Structure
FINAL PROJECT/
│
├── icdss_main_app.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│ └── model_stage3_targeted.h5
└── .streamlit/
└── secrets.toml # NOT included in repo

## 🔐 Secrets Setup

Create `.streamlit/secrets.toml` locally:

```toml
ANTHROPIC_API_KEY = "your_api_key_here"