# Importing libraries.
import pandas as pd

# Loading the dataset.
data = pd.read_csv("temperature_data (1).csv")

# Separating features and target based on 'profile_id'.
x = data.drop(columns=['motor_speed', 'profile_id'])
y = data['motor_speed']
unique_profiles = data['profile_id'].unique()
from sklearn.model_selection import train_test_split
train_profiles, test_profiles = train_test_split(unique_profiles, test_size=0.2, random_state=42)
x_train = x[data['profile_id'].isin(train_profiles)]
y_train = y[data['profile_id'].isin(train_profiles)]

# Scaling the features.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)

# Building the MLP model.
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, input_dim=x_train_scaled.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(1) ])
    
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(x_train_scaled, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=1)

# Pickle module is used to save Python objects (e.g., Machine Learning models, Scalers, etc.) into separate files for later use.
import pickle

# Saving the MLP model as "mlp_model.keras" using Keras in the .keras format. # Saving it in .h5 format had given a legacy format warning.
model.save("mlp_model.keras") 

# Saving the Scaler Object
# Python's with statement automatically closes the file right after the block. Without the with statement, after opening the file with the open() function, we would've needed to manually close it with the close() function.
 with open("scaler.pkl", "wb") as f: # Create a new file "scaler.pkl" and open it in a write-binary mode.
    pickle.dump(scaler, f) # Save the Scaler Object into this file.

# After this, 2 files will be created:
# "scaler.pkl" which will be used to standardize the new input data before making predictions.
# "mlp_model.keras" which contains the trained model so we don't have to retrain the model each time we need to make predictions.