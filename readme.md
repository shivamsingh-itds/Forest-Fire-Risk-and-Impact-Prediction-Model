# Forest Fire Risk and Impact Prediction using Machine Learning

## 📌 Project Overview
Forest fires pose serious environmental and economic threats. Early detection of fire occurrence and estimation of potential damage are crucial for effective disaster management.  
This project applies machine learning techniques to address forest fire prediction through **two complementary tasks**:

1. **Classification** – Predicting whether a forest fire will occur (primary objective)
2. **Regression** – Estimating the burned area when a fire occurs (supporting analysis)

The project is designed as an end-to-end ML workflow with clear problem framing, exploratory data analysis, feature engineering, model development, and evaluation.

---

## 📄 Research Paper Reference
This work is inspired by and compared against the research paper:

**Cortez, P. & Morais, A. (2007)**  
*A Data Mining Approach to Predict Forest Fires using Meteorological Data*

The referenced study primarily focused on predicting burned area using regression models and highlighted the difficulty of achieving high accuracy due to the highly skewed nature of fire size data.

📌 In this project:
- The research paper is included in the `research/` directory
- Our approach extends the study by:
  - Framing **fire occurrence as a classification problem**
  - Applying **log-normal regression** for burned area prediction
  - Comparing results with the research findings

---

## 📊 Dataset Description
The dataset contains meteorological and environmental features such as:
- Temperature
- Humidity
- Wind
- Rain
- Additional weather indices

The target variable `area` represents the total burned forest area.

---
## 🧠 Methodology

### 1️⃣ Classification (Primary Task)
**Objective:**  
Predict whether a forest fire will occur.

**Target Variable:**  
A binary target `Fire` was created:
- `Fire = 1` if burned area > 0  
- `Fire = 0` otherwise

**Models Used:**
- Logistic Regression (baseline)
- Random Forest (final model)

📌 **Final Model:**  
Random Forest Classifier was selected based on superior performance and robustness.

**Evaluation Metrics:**
- Precision
- Recall
- F1-score
- ROC-AUC

Recall was emphasized due to the higher cost of missing a fire event compared to false alarms.

---

### 2️⃣ Regression (Supporting Analysis)
**Objective:**  
Estimate the burned area when a fire occurs.

**Key Challenge:**  
The burned area variable is highly right-skewed with extreme outliers.

**Solution:**  
A **log-normal regression approach** was applied:
- Regression performed only on records where `area > 0`
- Target transformed using `log1p(area)`
- Predictions inverse-transformed for evaluation

**Models Used:**
- Random Forest Regressor
- Gradient Boosting Regressor

This approach significantly improved model stability compared to direct regression.

---

## 📈 Key Insights
- Forest fire **occurrence** is more predictable than exact burned area
- Temperature and wind are strong indicators of fire risk
- Burned area prediction remains inherently difficult due to environmental uncertainty
- Log-normal transformation is essential for meaningful regression results

---

## ✅ Conclusion
This project demonstrates a practical and research-aligned approach to forest fire prediction by combining classification and regression techniques.  
Compared to the reference research paper, the classification-based framing provides stronger real-world applicability for early warning systems, while log-normal regression improves burned area estimation stability.

---

## 🚀 Future Improvements
- Incorporate satellite or spatial data
- Time-series modeling for seasonal fire trends
- Cost-sensitive learning for classification

---

## 🛠️ Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

---
## 👤 Author

**Shivam Singh**
Aspiring Data Scientist | Machine Learning Enthusiast

🔗 GitHub: [https://github.com/shivamsingh-itds]
🔗 LinkedIn: [www.linkedin.com/in/shivamsinghds]

---

⭐ If you find this project helpful, feel free to star the repository!