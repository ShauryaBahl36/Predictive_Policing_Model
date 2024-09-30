import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error
import folium
from streamlit_folium import folium_static
import os

# Directory to save uploaded files
UPLOAD_DIR = r'C:\Users\Acer\python\Predictive Policing\input'
# Directory to save output files
OUTPUT_DIR = r'C:\Users\Acer\python\Predictive Policing\output'

# Ensure the upload and output directories exist
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Title of the app
st.title('Predictive Policing Application')

# File upload module for historical crime data
uploaded_crime_file = st.file_uploader("Choose a file for crime data (2001-2011)", type=["csv"])

# File upload module for latitude and longitude data
uploaded_latlon_file = st.file_uploader("Choose a file for latitude and longitude data", type=["csv"])

if uploaded_crime_file is not None and uploaded_latlon_file is not None:
    # Validate file types
    if uploaded_crime_file.name.endswith('.csv') and uploaded_latlon_file.name.endswith('.csv'):
        # Save the uploaded files
        crime_file_path = os.path.join(UPLOAD_DIR, uploaded_crime_file.name)
        with open(crime_file_path, "wb") as f:
            f.write(uploaded_crime_file.getbuffer())
        
        latlon_file_path = os.path.join(UPLOAD_DIR, uploaded_latlon_file.name)
        with open(latlon_file_path, "wb") as f:
            f.write(uploaded_latlon_file.getbuffer())
        
        st.success(f"Files '{uploaded_crime_file.name}' and '{uploaded_latlon_file.name}' uploaded successfully!")
        
        # Load the datasets
        crime_data = pd.read_csv(crime_file_path)
        latlon_data = pd.read_csv(latlon_file_path)
        st.write(crime_data.head())
        st.write(latlon_data.head())
        
        # Execute button module
        if st.button('Execute'):
            # Preprocess the crime data
            pivot_table = crime_data.pivot(index='DISTRICT', columns='YEAR', values='TOTAL IPC CRIMES')
            pivot_table.fillna(0, inplace=True)
            
            # Function to create lagged features
            def create_lagged_features(data, lags):
                lagged_data = {}
                for lag in lags:
                    lagged_data[f'lag_{lag}'] = data.shift(lag)
                return pd.DataFrame(lagged_data)

            # Split the data into training and testing sets
            train_years = list(range(2001, 2009))
            test_years = [2009, 2010, 2011]
            
            X_train = np.array(train_years).reshape(-1, 1)
            X_test = np.array(test_years).reshape(-1, 1)
            
            districts = pivot_table.index.values
            y = pivot_table.values
            
            predictions = {}
            actuals = {}
            for i, district in enumerate(districts):
                district_crimes = pd.Series(y[i], index=train_years + test_years)
                lagged_features = create_lagged_features(district_crimes, lags=[1, 2])
                district_crimes = district_crimes.to_frame(name='crimes')
                
                full_data = pd.concat([district_crimes, lagged_features], axis=1).dropna()
                
                X_full = full_data.drop(columns=['crimes']).values
                y_full = full_data['crimes'].values
                
                # Train-test split
                X_train_full = X_full[:len(train_years)-2]
                y_train_full = y_full[:len(train_years)-2]
                X_test_full = X_full[len(train_years)-2:]
                y_test_full = y_full[len(train_years)-2:]
                
                # Train the model
                model = XGBRegressor(n_estimators=100, random_state=42)
                model.fit(X_train_full, y_train_full)
                
                # Predict the test set
                y_pred = model.predict(X_test_full)
                
                # Round predictions to nearest integer and convert to int
                y_pred = np.round(y_pred).astype(int)
                
                predictions[district] = y_pred
                actuals[district] = y_test_full
            
            # Calculate MAPE for each district
            mape_values = {year: [] for year in test_years}
            for district in districts:
                for i, year in enumerate(test_years):
                    mape_values[year].append(mean_absolute_percentage_error([actuals[district][i]], [predictions[district][i]]))
            
            # Create a DataFrame for the predictions and actuals
            predictions_df = pd.DataFrame({
                'DISTRICT': districts,
                'PREDICTED_2009': [pred[0] for pred in predictions.values()],
                'PREDICTED_2010': [pred[1] for pred in predictions.values()],
                'PREDICTED_2011': [pred[2] for pred in predictions.values()],
                'ACTUAL_2009': [act[0] for act in actuals.values()],
                'ACTUAL_2010': [act[1] for act in actuals.values()],
                'ACTUAL_2011': [act[2] for act in actuals.values()],
                'MAPE_2009': mape_values[2009],
                'MAPE_2010': mape_values[2010],
                'MAPE_2011': mape_values[2011]
            })
            
            # Merge predictions with latitude and longitude data
            merged_df = predictions_df.merge(latlon_data, on='DISTRICT')
            
            # Display the merged DataFrame
            st.write(merged_df)
            
            # Save the predictions to a CSV file
            predictions_csv_path = os.path.join(OUTPUT_DIR, 'predicted_crimes_2009_2011.csv')
            merged_df.to_csv(predictions_csv_path, index=False)
            st.success(f"Predicted crimes saved to '{predictions_csv_path}'")
            
            # Display MAPE results
            st.write("Mean Absolute Percentage Error (MAPE) by District and Year:")
            st.write(merged_df[['DISTRICT', 'MAPE_2009', 'MAPE_2010', 'MAPE_2011']])
            
            # Create a folium map
            map_bihar = folium.Map(location=[25.0961, 85.3131], zoom_start=7)
            
            # Add district markers to the map
            for _, row in merged_df.iterrows():
                folium.CircleMarker(
                    location=[row['LATITUDE'], row['LONGITUDE']],
                    radius=row['PREDICTED_2011'] / 100,  # Adjust the radius as needed
                    popup=f"{row['DISTRICT']}: {row['PREDICTED_2011']} predicted crimes",
                    color='crimson',
                    fill=True,
                    fill_color='crimson'
                ).add_to(map_bihar)
            
            # Save the map to an HTML file
            map_bihar_html = os.path.join(OUTPUT_DIR, 'crime_hotspots_map.html')
            map_bihar.save(map_bihar_html)
            st.success(f"Map saved to '{map_bihar_html}'")
            
            # Display the map
            folium_static(map_bihar)
    else:
        st.error("File types must be CSV.")
else:
    st.info("Please upload both the crime data file and the latitude and longitude data file.")
