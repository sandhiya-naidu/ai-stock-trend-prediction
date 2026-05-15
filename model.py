import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

stock = "TCS.NS"

# Download data
data = yf.download(stock, period="2y", group_by="column", progress=False)

#  FIX: Flatten columns (THIS IS THE KEY)
data.columns = data.columns.get_level_values(0)

# Create Tomorrow Close
data['Tomorrow_Close'] = data['Close'].shift(-1)

# Drop last row (NaN tomorrow)
data = data.dropna()

# Create Trend (NOW THIS WILL WORK)
data['Trend'] = (data['Tomorrow_Close'] > data['Close']).astype(int)

# Features
features = ['Open', 'High', 'Low', 'Close', 'Volume']
X = data[features]
y = data['Trend']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f" Model trained with accuracy: {acc*100:.2f}%")

# Save model
with open("stock_model.pkl", "wb") as f:
    pickle.dump(model, f)

print(" stock_model.pkl saved successfully!")
