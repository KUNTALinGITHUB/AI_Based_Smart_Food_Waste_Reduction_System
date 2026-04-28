import joblib
import pandas as pd
import random
from datetime import datetime, timedelta
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Product configuration
products = {
    "Milk": {"base_price": 50, "expiry": (1, 5), "temp": (2, 6), "demand_factor": 0.8},
    "Bread": {"base_price": 30, "expiry": (2, 6), "temp": (18, 25), "demand_factor": 0.7},
    "Apple": {"base_price": 120, "expiry": (3, 7), "temp": (10, 20), "demand_factor": 0.6},
    "Banana": {"base_price": 60, "expiry": (2, 5), "temp": (12, 25), "demand_factor": 0.9},
    "Tomato": {"base_price": 40, "expiry": (2, 6), "temp": (15, 25), "demand_factor": 0.75},
    "Chicken": {"base_price": 200, "expiry": (1, 3), "temp": (0, 4), "demand_factor": 0.65},
    "Rice": {"base_price": 70, "expiry": (30, 90), "temp": (20, 35), "demand_factor": 0.5},
    "Eggs": {"base_price": 80, "expiry": (5, 10), "temp": (4, 10), "demand_factor": 0.85}
}

rows = []
start_date = datetime(2026, 1, 1)

for i in range(1000):
    product = random.choice(list(products.keys()))
    info = products[product]

    date = start_date + timedelta(days=random.randint(0, 180))
    weekday = date.weekday()
    is_weekend = 1 if weekday >= 5 else 0
    weekend_factor = 1.2 if is_weekend else 1.0

    quantity = random.randint(50, 200)
    price = info["base_price"] + random.randint(-15, 15)
    temperature = round(random.uniform(info["temp"][0], info["temp"][1]), 2)

    expiry_min, expiry_max = info["expiry"]
    temp_effect = max(0, (temperature - np.mean(info["temp"])) * 0.5)
    expiry_days = max(1, int(random.randint(expiry_min, expiry_max) - temp_effect))

    demand_base = quantity * info["demand_factor"] * weekend_factor
    price_sensitivity = max(0.5, 1 - (price - info["base_price"]) / 100)
    predicted_demand = int(demand_base * price_sensitivity * random.uniform(0.7, 1.1))

    # Decision logic (rule-based layer)
    if expiry_days <= 2 and quantity > predicted_demand:
        risk_level = "HIGH"
        action = "Apply 25% discount + Send to NGO"
    elif expiry_days <= 3:
        risk_level = "MEDIUM"
        action = "Apply 10-15% discount"
    else:
        risk_level = "LOW"
        action = "Normal selling"

    rows.append([
        date.strftime("%Y-%m-%d"),
        product,
        quantity,
        price,
        expiry_days,
        temperature,
        weekday,
        is_weekend,
        predicted_demand,
        risk_level,
        action
    ])

# ---------------- RAW DATASET ----------------
df_raw = pd.DataFrame(rows, columns=[
    "date", "product", "quantity", "price", "expiry_days",
    "temperature", "day_of_week", "is_weekend",
    "predicted_demand", "risk_level", "recommended_action"
])

df_raw.to_csv("food_waste_raw_1000.csv", index=False)

# ---------------- ML DATASET ----------------
df_ml = df_raw.copy()

# Encode product
le = LabelEncoder()
df_ml["product_encoded"] = le.fit_transform(df_ml["product"])

# Drop text column for ML
df_ml = df_ml.drop(columns=["product", "date"])

# Normalize numerical features
scaler = MinMaxScaler()
num_cols = ["quantity", "price", "expiry_days", "temperature"]

df_ml[num_cols] = scaler.fit_transform(df_ml[num_cols])

df_ml.to_csv("food_waste_ml_1000.csv", index=False)

# After creating encoder & scaler
joblib.dump(le, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Encoder & Scaler saved!")

print("✅ RAW dataset saved: food_waste_raw_1000.csv")
print("✅ ML dataset saved: food_waste_ml_1000.csv")
print(df_ml.head())