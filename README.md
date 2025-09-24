# reply-classification


# Reply Classification API
This project provides a **FastAPI service** that classifies email replies into:
- **positive** (interested in meeting/demo)
- **negative** (not interested / rejection)
- **neutral** (non-committal or irrelevant)
It uses a Logistic Regression model trained on TF-IDF features for fast and reliable predictions.
## 🚀 Setup Instructions
### 1. Clone this repository
```bash
git clone <https://github.com/Neutrino09/reply-classification.git>
cd reply_classifier_api

pip install -r requirements.txt

uvicorn app:app --reload

The server will start at:
http://127.0.0.1:8000
Example request

POST /predict
Input:
{
  "text": "Looking forward to the demo!"
}


Output:
{
  "label": "positive",
  "confidence": 0.97
}

