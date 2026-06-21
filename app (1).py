
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model, scaler, and outlier bounds
model = joblib.load('best_gb_model.pkl')
scaler = joblib.load('scaler.pkl')
outlier_bounds = joblib.load('outlier_bounds.pkl')

def cap_outliers_iqr_segment(df_segment, column, q1, q3, iqr):
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df_segment[column] = np.where(df_segment[column] < lower_bound, lower_bound, df_segment[column])
    df_segment[column] = np.where(df_segment[column] > upper_bound, upper_bound, df_segment[column])
    return df_segment

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        # Convert data types to numeric, handling potential errors
        player_data = {
            'Player_Level': float(data.get('player_level')),
            'Average_Session_Length_Mins': float(data.get('average_session_length_mins')),
            'Login_Frequency_Last_14_Days': float(data.get('login_frequency_last_14_days')),
            'Days_Since_Last_Login': float(data.get('days_since_last_login')),
            'In_App_Purchases_USD': float(data.get('in_app_purchases_usd')),
            'Failed_Level_Attempts': float(data.get('failed_level_attempts')),
            'Support_Tickets_Raised': float(data.get('support_tickets_raised'))
        }

        player_df = pd.DataFrame([player_data])

        # Scale the new data
        scaled_player_data = scaler.transform(player_df)
        scaled_player_df = pd.DataFrame(scaled_player_data, columns=player_df.columns)

        # Cap outliers using the learned bounds
        for col in scaled_player_df.columns:
            if col in outlier_bounds:
                bounds = outlier_bounds[col]
                scaled_player_df = cap_outliers_iqr_segment(scaled_player_df, col, bounds['Q1'], bounds['Q3'], bounds['IQR'])

        # Make prediction
        prediction = model.predict(scaled_player_df)[0]
        prediction_proba = model.predict_proba(scaled_player_df)[0][1]

        return jsonify({
            'prediction': int(prediction),
            'probability': float(prediction_proba),
            'message': 'Player is likely to churn' if prediction == 1 else 'Player is likely to not churn'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
