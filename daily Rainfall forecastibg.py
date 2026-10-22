import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. Load Dataset & Convert to Daily Time Series
df = pd.read_csv('Indian Rainfall Dataset District-wise Daily Measurements.csv', sep=';')
dist_df = df[df['district'] == 'Ernakulam'].copy()

days_cols = ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th',
             '11th', '12th', '13th', '14th', '15th', '16th', '17th', '18th', '19th', '20th',
             '21st', '22nd', '23rd', '24th', '25th', '26th', '27th', '28th', '29th', '30th', '31st']

records = []
for idx, row in dist_df.iterrows():
    month = int(row['month'])
    for i, col in enumerate(days_cols, start=1):
        try:
            date_str = f"2022-{month:02d}-{i:02d}"
            pd.to_datetime(date_str)
            records.append({'date': date_str, 'rainfall': row[col]})
        except ValueError:
            pass

ts_df = pd.DataFrame(records)
ts_df['date'] = pd.to_datetime(ts_df['date'])
ts_df = ts_df.sort_values('date').reset_index(drop=True)

# 2. Normalize & Create 7-day Sequences
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(ts_df[['rainfall']])

def create_sequences(data, seq_length=7):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

SEQ_LENGTH = 7
X, y = create_sequences(scaled_data, SEQ_LENGTH)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Chronological Split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 3. Build & Train LSTM Model
tf.random.set_seed(42)
model = Sequential([
    LSTM(50, activation='relu', input_shape=(SEQ_LENGTH, 1)),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

# 4. Evaluate
y_pred_scaled = model.predict(X_test)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_inv = scaler.inverse_transform(y_pred_scaled)

print(f"Test MAE: {mean_absolute_error(y_test_inv, y_pred_inv):.4f} mm")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test_inv, y_pred_inv)):.4f} mm")