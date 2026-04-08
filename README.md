#  Credit Risk Prediction API using Machine Learning

##  Overview
This project predicts whether a loan applicant is likely to default or repay using Machine Learning models.
It helps financial organizations to make data-driven lending decisions and reduce financial risk.

---

##  Objectives
- Predict customer creditworthiness  
- Reduce loan default risk  
- Enable data-driven decision making  

---

##  Dataset Credits

The dataset used in this project is the **"Give Me Some Credit"** dataset from Kaggle.

Link: https://www.kaggle.com/c/GiveMeSomeCredit

---

##  Tech Stack
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib / Seaborn
- Tensarflow
- keras
- MLP  

---

##  Models Used
- Logistic Regression  
- Decision Tree  
- Random Forest  
- XGBoost
- Artificial Neural Networks(ANN)  

---

##  Model Performance
| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | 0.79     |
| Random Forest       | 0.85     |
| XGBoost             | 0.84     |
| ANN                 | 0.82     |
---

##  End-to-End ML Pipeline

### Data Pipeline
1. Data Ingestion  
2. Data Cleaning  
3. Feature Engineering  
4. Exploratory Data Analysis (EDA)  
5. Model Training  
6. Model Evaluation  
7. Prediction Pipeline  

---

##  Pipeline Architecture

Data Source → Ingestion → Processing → Training → Evaluation → Deployment → Prediction API → Monitoring

---

##  Deployment (AWS EC2)

The model was deployed on an AWS EC2 instance and exposed via a REST API for real-time predictions.

###  Deployment Steps
- Launched EC2 instance (Linux)  
- Configured environment and dependencies  
- Deployed ML model using Flask API  
- Opened required ports using Security Groups  
- Accessed model via public IP  

---

###  Current Status
- EC2 instance has been **terminated to avoid unnecessary cloud charges**  
- Deployment can be re-enabled anytime using the same setup  

---

###  Technologies Used in Deployment
- AWS EC2  
- Ubuntu 
- Flask API  
- Git  

---

##  MLOps Approach

- Designed an end-to-end ML pipeline  
- Structured code for scalability and deployment  
- Followed best practices for modular development  
- Deployment aligned with MLOps principles (CI/CD ready)  

---

##  Results
- Random Forest achieved highest accuracy  
- Identified key factors affecting loan default  
- Built a scalable ML pipeline  

---

##  Future Improvements
- Add CI/CD pipeline using GitHub Actions  
- Deploy using Docker + Cloud (AWS / Render)  
- Implement model monitoring and retraining  

  



