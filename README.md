# 🍽️ AI-Based Smart Food Waste Reduction System

An AI-powered system to predict food demand, identify waste risk, and recommend actions to reduce food waste in supply chains.

---

## 🚀 Project Overview

Food waste is a major global issue, especially in developing countries. This project uses Machine Learning to:

- Predict product demand 📊
- Classify waste risk ⚠️
- Forecast future demand 📈
- Suggest smart actions 🤖

---

## 🧠 Models Used

### 1. Demand Prediction
- Model: Gradient Boosting Regressor
- R² Score: ~0.84
- MAE: ~10.8

### 2. Risk Classification
- Model: Random Forest / Classifier
- Accuracy: 97.5%

### 3. Time-Series Forecasting
- Model: Gradient Boosting with Lag Features
- Forecast: Next 7 days demand

---

## 📊 Features Used

- Product
- Quantity (Stock)
- Price
- Expiry Days
- Temperature
- Day of Week
- Weekend Indicator

---

## ⚙️ System Workflow
Input Data → Preprocessing →
Demand Prediction → Risk Classification →
Forecasting → Decision Engine → Output

## 💡 Example Output

⚠️ Available Products: Milk, Bread, Apple, Banana, Tomato, Chicken, Rice, Eggs

Enter product: Apple
Enter stock: 200
Enter price: 30
Enter expiry days: 6
Enter temperature: 23
Enter day of week (0-6): 6
Is it a weekend? (1/0): 0

================ SMART FOOD WASTE SYSTEM ================

🛒 Product: Apple
📦 Stock Available: 200 units
📅 Expiry in: 6 days

📊 DEMAND INSIGHT:
👉 Expected sales: ~148 units
⚠️ Extra stock: ~52 units may remain unsold

🚨 RISK STATUS:
🟢 Low Risk → Everything looks safe

💡 RECOMMENDED ACTION:
→ Continue normal selling

📈 NEXT 7 DAYS SALES FORECAST:
Day 1: ~67 units
Day 2: ~48 units
Day 3: ~74 units
Day 4: ~68 units
Day 5: ~69 units
Day 6: ~52 units
Day 7: ~73 units

=========================================================

---

## 📂 Dataset

Synthetic dataset (1000 rows) simulating:
- Inventory
- Temperature impact
- Demand patterns

---

## ▶️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt