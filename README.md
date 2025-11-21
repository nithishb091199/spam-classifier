# Spam Email Classifier

**A production-ready spam classifier** built with Python, scikit-learn, and Flask — includes TF-IDF feature extraction, a Multinomial Naive Bayes model, a demo UI, unit tests, and CI.

![Demo screenshot](sandbox:/mnt/data/fa550e33-6440-4ecd-ab6d-5b621d53ae77.png)

## Live demo (local)

Start the project locally (Windows PowerShell):

```powershell
# Activate venv
. .venv/Scripts/Activate.ps1
# Install deps (if not already)
pip install -r requirements.txt
# Run the app
python app/app.py
# Visit in a browser:
# http://127.0.0.1:5000/
```

## Project structure

```
spam-classifier/
├─ README.md
├─ requirements.txt
├─ data/
│  ├─ raw/ (raw datasets)
│  └─ processed/ (cleaned csv & joblib model)
├─ notebooks/
├─ src/
│  ├─ data_preprocess.py
│  └─ train.py
├─ app/
│  ├─ app.py
│  └─ static/index.html
├─ tests/
│  └─ test_inference.py
└─ .github/workflows/python-app.yml
```

## What this project demonstrates

* End-to-end ML workflow: EDA, preprocessing, model training, evaluation, and inference
* Text feature engineering with TF-IDF
* Model serving via Flask (REST API) and a lightweight demo frontend
* Unit tests with pytest and CI with GitHub Actions
* Docker-ready for easy deployment

## Quick API examples

Health check:

```bash
curl http://127.0.0.1:5000/health
```

Predict (curl):

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Free entry to win a prize"}'
```

Response example:

```json
{
  "prediction": "spam",
  "label": 1,
  "confidence": 0.99
}
```

## Model & results

* Model: `MultinomialNB` with `TfidfVectorizer` (unigrams + bigrams)
* Typical test accuracy: **95%+**
* Saved pipeline: `data/processed/spam_model.joblib`

## How to reproduce training

```powershell
. .venv/Scripts/Activate.ps1
python src/data_preprocess.py
python src/train.py
```

## Run tests (pytest)

```powershell
. .venv/Scripts/Activate.ps1
pytest -q
```

## Docker (build & run)

```bash
# Build
docker build -t spam-classifier .
# Run
docker run -p 5000:5000 spam-classifier
```

## Deployment tips

* For simple hosting use Render or Heroku (Procfile included)
     https://spam-classifier-v2f7.onrender.com/ 

  <img width="1422" height="582" alt="image" src="https://github.com/user-attachments/assets/221110ab-d822-4a2b-bf94-e3d9a1b80110" />

* For production, use a WSGI server (gunicorn or waitress) behind a reverse proxy

## License & contact

nithishb091199@gmail.com.
