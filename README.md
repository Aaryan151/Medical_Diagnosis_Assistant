# 🏥 MediPredict – Multi-Disease Prediction System

AI-powered web app to predict 4 diseases using trained Random Forest models.

## 📁 Project Structure

```
disease_predictor/
│
├── app.py                    ← Main Streamlit application
├── requirements.txt          ← Python dependencies
├── README.md
│
└── models/
    ├── diabetes_model.pkl
    ├── heart_disease_model.pkl
    ├── lung_cancer_model.pkl
    └── thyroid_model.pkl
```

---

## 🚀 How to Run Locally

### Step 1 – Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 – Run the app
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## ☁️ Deploy on Streamlit Cloud (Free Hosting)

### Step 1 – Push to GitHub
1. Create a new GitHub repository (e.g. `medipredict`)
2. Push all files:
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/medipredict.git
git push -u origin main
```

> ⚠️ `.pkl` files may be large — if >100MB, use [Git LFS](https://git-lfs.github.com/)

### Step 2 – Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Connect your GitHub account
4. Select your repo → branch `main` → main file `app.py`
5. Click **Deploy**

Your app will be live at:
`https://<your-app-name>.streamlit.app`

---

## 🔬 Diseases Covered

| Disease       | Target Column | Model Accuracy |
|--------------|---------------|----------------|
| Heart Disease | `target`      | ~95%           |
| Diabetes      | `Outcome`     | ~80%           |
| Lung Cancer   | `LUNG_CANCER` | ~99%           |
| Thyroid       | `binaryClass` | ~99%           |

---

## ⚠️ Disclaimer

This tool is for **educational purposes only**.  
Do NOT use it as a substitute for professional medical advice.
