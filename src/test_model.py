import joblib
import pandas as pd

# ================= LOAD MODELS =================
try:
    demand_model = joblib.load("demand_model.pkl")
    risk_model = joblib.load("risk_model.pkl")

    le = joblib.load("label_encoder.pkl")
    risk_encoder = joblib.load("risk_encoder.pkl")
    scaler = joblib.load("scaler.pkl")
except:
    print("❌ Error loading models. Check file paths.")
    exit()

# ================= INPUT =================
print("\n⚠️ Available Products: Milk, Bread, Apple, Banana, Tomato, Chicken, Rice, Eggs\n")

product = input("Enter product: ").strip()

def safe_int(prompt):
    try:
        return int(input(prompt))
    except:
        print("❌ Invalid number")
        exit()

def safe_float(prompt):
    try:
        return float(input(prompt))
    except:
        print("❌ Invalid number")
        exit()

quantity = safe_int("Enter stock: ")
price = safe_int("Enter price: ")
expiry_days = safe_int("Enter expiry days: ")
temperature = safe_float("Enter temperature: ")
day_of_week = safe_int("Enter day of week (0-6): ")
is_weekend = safe_int("Is it a weekend? (1/0): ")

# ================= VALIDATION =================
if day_of_week < 0 or day_of_week > 6:
    print("❌ Day must be between 0 and 6")
    exit()

if is_weekend not in [0, 1]:
    print("❌ Weekend must be 0 or 1")
    exit()

# ================= PREPROCESSING =================
try:
    product_encoded = le.transform([product])[0]
except:
    print("❌ Invalid product name")
    exit()

input_df = pd.DataFrame([{
    "quantity": quantity,
    "price": price,
    "expiry_days": expiry_days,
    "temperature": temperature
}])

scaled_values = scaler.transform(input_df)

scaled_df = pd.DataFrame(scaled_values, columns=[
    "quantity", "price", "expiry_days", "temperature"
])

final_input = pd.DataFrame([{
    "quantity": scaled_df.loc[0, "quantity"],
    "price": scaled_df.loc[0, "price"],
    "expiry_days": scaled_df.loc[0, "expiry_days"],
    "temperature": scaled_df.loc[0, "temperature"],
    "day_of_week": day_of_week,
    "is_weekend": is_weekend,
    "product_encoded": product_encoded
}])

# ================= DEMAND PREDICTION =================
predicted_demand = int(demand_model.predict(final_input)[0])

# ================= RISK PREDICTION =================
risk_pred = risk_model.predict(final_input)[0]
risk = risk_encoder.inverse_transform([risk_pred])[0]

# ================= SMART CORRECTION =================
surplus = quantity - predicted_demand

# More realistic rule tuning
if surplus > 50 and expiry_days <= 5:
    risk = "HIGH"
elif surplus > 20 and expiry_days <= 7:
    risk = "MEDIUM"

# ================= ACTION =================
if risk == "HIGH":
    action = "🔥 Offer 25% discount + Send excess stock to NGO"
elif risk == "MEDIUM":
    action = "⚡ Offer 10–15% discount to increase sales"
else:
    action = "✅ Continue normal selling"

# ================= FORECASTING =================
future_predictions = []

try:
    forecast_model = joblib.load(f"forecasting_model/gbr_{product}.pkl")
    last_values = joblib.load(f"forecasting_model/last_values_{product}.pkl")
    last_date = joblib.load(f"forecasting_model/last_date_{product}.pkl")
    feature_cols = joblib.load(f"forecasting_model/features_{product}.pkl")

    lag = 7

    for i in range(7):
        next_date = last_date + pd.Timedelta(days=1)

        day_of_week_f = next_date.weekday()
        is_weekend_f = 1 if day_of_week_f >= 5 else 0

        input_data = last_values[-lag:] + [day_of_week_f, is_weekend_f]
        input_df_forecast = pd.DataFrame([input_data], columns=feature_cols)

        pred = forecast_model.predict(input_df_forecast)[0]
        future_predictions.append(int(pred))

        last_values.append(pred)
        last_date = next_date

except:
    future_predictions = ["Not available"] * 7

# ================= USER-FRIENDLY OUTPUT =================
print("\n========================================================")
print("🤖 AI SMART FOOD WASTE REDUCTION SYSTEM")
print("========================================================\n")

print(f"🛒 Product: {product}")
print(f"📦 Stock Available: {quantity} units")
print(f"📅 Expiry in: {expiry_days} days")

print("\n📊 DEMAND ANALYSIS")
print(f"👉 Predicted Demand: ~{predicted_demand} units")

if surplus > 0:
    print(f"⚠️ Surplus Stock: ~{surplus} units (risk of waste)")
else:
    print("✅ Stock matches demand (no waste expected)")

print("\n🚨 RISK LEVEL")
if risk == "HIGH":
    print("🔴 HIGH → Immediate action required")
elif risk == "MEDIUM":
    print("🟡 MEDIUM → Monitor and act soon")
else:
    print("🟢 LOW → Situation under control")

print("\n💡 RECOMMENDED ACTION")
print(action)

print("\n📈 7-DAY DEMAND FORECAST")
for i, val in enumerate(future_predictions, 1):
    print(f"Day {i}: ~{val} units")

print("\n========================================================\n")