# train.py
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import joblib

# 1. Fetch the real airline passenger dataset via URL
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
df = pd.read_csv(url)

# 2. Format columns to meet strict Time-Series requirements
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
df.index.freq = 'MS'  # 'MS' stands for Month Start frequency

# Rename target column for clean access
df = df.rename(columns={'Passengers': 'passengers'})

# 3. Fit ARIMA model 
# We use order=(1, 1, 1) as our initial baseline configuration
model = ARIMA(df['passengers'], order=(1, 1, 1))
model_fitted = model.fit()

# 4. Save the model artifact to disk
joblib.dump(model_fitted, "arima_model.pkl")
print("Successfully trained ARIMA on Air Passengers dataset!")
print(df.tail())