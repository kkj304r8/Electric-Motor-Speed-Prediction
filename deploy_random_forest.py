# Importing libraries.
import pandas as pd
import streamlit as st

# Title for the Streamlit app.
st.title('Machine Learning Model: Random Forest')

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

# Building the Random Forest model.
from sklearn.ensemble import RandomForestRegressor
forest_model = RandomForestRegressor(n_estimators=20, max_depth=10, min_samples_split=10, n_jobs=-1, random_state=42)
forest_model.fit(x_train_scaled, y_train)

# Using the model to make predictions based on user-provided inputs (stored in 'df').
y_pred_forest = forest_model.predict(df)

# Display the result under 'Predicted Motor Speed' formatted to 3 decimal places.
st.subheader('Predicted Motor Speed')
st.write(f"{float(y_pred_forest[0]):.3f}")