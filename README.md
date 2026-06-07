# 🚀 Resume Screening System using Machine Learning

## 📌 Overview
This project is my **first hands-on Machine Learning & AI project**.  
It automatically analyzes resumes and predicts the appropriate job category using **Python, Scikit-Learn, and Natural Language Processing (NLP)** techniques.

## 🛠️ Tech Stack
- Python  
- Pandas  
- Regular Expressions (Regex)  
- TF-IDF Vectorization  
- Label Encoding  
- Scikit-Learn  

## 🤖 Models Explored
- Multinomial Naive Bayes  
- Logistic Regression  
- Linear SVM (LinearSVC)  
- Voting Classifier Ensemble  

## 🔄 Workflow
1. Loaded and analyzed dataset  
2. Cleaned resume text using Regex  
3. Converted text into numerical features using TF-IDF  
4. Encoded job categories using LabelEncoder  
5. Split data into training and testing sets  
6. Trained multiple ML models  
7. Evaluated performance using Accuracy Score and Cross Validation  

## 📊 Results
- Naive Bayes → ~97%  
- Logistic Regression → ~99%  
- Linear SVM → ~99%  
- Voting Classifier → ~99%  

## 💡 Key Learnings
- Text preprocessing techniques  
- Feature extraction using TF-IDF  
- Classification algorithms  
- Model evaluation & cross-validation  
- Ensemble learning with Voting Classifier  
- NLP for resume parsing  
- AI-driven automation  

## 📂 Dataset
The dataset used is `UpdatedResumeDataSet.csv`.  
(⚠️ Note: Please ensure you have the dataset locally or download it from a trusted source such as Kaggle.)

## 🚀 How to Run
```bash
# Clone the repository
git clone https://github.com/<your-username>/Resume-Screening-ML.git

# Navigate into the project folder
cd Resume-Screening-ML

# Install dependencies
pip install -r requirements.txt

# Run the script
python resume_screening.py
