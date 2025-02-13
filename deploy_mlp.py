# Importing libraries.
import pandas as pd
import streamlit as st

# Title for the Streamlit app.
st.title('Machine Learning Model: Multi-layer Perceptron')

# Title for the Sidebar.
st.sidebar.header('User Input Parameters')

# Creating a function which will request values from the user.
def user_input_features():
	ambient = st.sidebar.number_input('Ambient Temperature')
	coolant = st.sidebar.number_input('Coolant Temperature')
	u_d = st.sidebar.number_input('Voltage d-component')
	u_q = st.sidebar.number_input('Voltage q-component')
	torque = st.sidebar.number_input('Torque induced by Current')
	i_d = st.sidebar.number_input('Current d-component')
	i_q = st.sidebar.number_input('Current q-component')
	pm = st.sidebar.number_input('Permanent Magnet Surface Temperature')
	stator_yoke = st.sidebar.number_input('Stator Yoke Temperature')
	stator_tooth = st.sidebar.number_input('Stator Tooth Temperature')
	stator_winding = st.sidebar.number_input('Stator Winding Temperature')
	
	cols = {'Ambient': ambient,
			'Coolant': coolant,
			'u_d': u_d,
			'u_q': u_q,
			'Torque': torque,
			'i_d': i_d,
			'i_q': i_q,
			'pm': pm,
			'stator_yoke': stator_yoke,
			'stator_tooth': stator_tooth,
			'stator_winding': stator_winding}
			
    # Creating a DataFrame to feed to the model.
	features = pd.DataFrame(cols, index = [0]) # Since I'm  passing a dictionary with scalar values to pd.DataFrame, need to specify the index (0), to tell pandas to create a single row.
    
	return features
	
df = user_input_features() # User-provided Inputs will be stored in 'df'.

st.subheader('User Input Parameters')
st.write(df) # Running the function.

# For uniformity purposes, need to scale 'df' since the corresponding Trained data is already scaled.
# Loading the trained Scaler object ("scaler.pkl") to use it on the user-provided inputs.
import pickle
with open("scaler.pkl", "rb") as f: # Load the Scaler Object in a read-binary mode.
    scaler = pickle.load(f) # Save it in variable 'scaler'.

df.columns = df.columns.str.lower() # Had received an error regarding column name mismatch with the trained data. Lowering the case to make the case match.
from sklearn.preprocessing import StandardScaler
df_scaled = scaler.transform(df)

# Loading the MLP model
import tensorflow as tf
from tensorflow.keras.models import load_model
model = load_model("mlp_model.keras", compile=False) # Save it in variable 'model'.

# Using the model to make predictions based on user-provided inputs (stored in 'df_scaled').
y_pred_nn = model.predict(df_scaled)

# Display the result under 'Predicted Motor Speed' formatted to 3 decimal places.
st.subheader('Predicted Motor Speed')
st.write(f"{float(y_pred_nn[0]):.3f}")