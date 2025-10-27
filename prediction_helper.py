import pandas as pd
from joblib import load

# Load models and scalers (all joblib-based)
model_rest = load(r"artifacts\model_rest.joblib")
model_young = load(r"artifacts\model_young.joblib")
scaler_rest = load(r"artifacts\scaler_rest.joblib")
scaler_young = load(r"artifacts\scaler_young.joblib")

def calculate_normalized_risk(medical_history):
    """
    Calculates a normalized risk score from medical_history string.
    Normalized between 0 (min) and 1 (max).
    """
    # Risk score mapping inside the function
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }

    # Define min and max possible total scores
    min_score = 0
    max_score = 14  # maximum possible combined score

    if not medical_history:
        return 0

    # Split multiple diseases and normalize
    diseases = [d.strip().lower() for d in medical_history.split("&")]

    # Calculate total risk score
    total_risk_score = sum(risk_scores.get(d, 0) for d in diseases)

    # Normalize
    normalized_risk_score = (total_risk_score - min_score) / (max_score - min_score)
    return normalized_risk_score

def preprocess(input_data):
    expected_columns =  ['age',
    'number_of_dependants',
    'income_lakhs',
    'insurance_plan',
    'genetical_risk',
    'normalized_risk_score',
    'gender_Male',
    'region_Northwest',
    'region_Southeast',
    'region_Southwest',
    'marital_status_Unmarried',
    'bmi_category_Obesity',
    'bmi_category_Overweight',
    'bmi_category_Underweight',
    'smoking_status_Occasional',
    'smoking_status_Regular',
    'employment_status_Salaried',
    'employment_status_Self-Employed']

    insurance_plan_encoding = {'Bronze':1,'Silver':2,'Gold':3}
    df = pd.DataFrame(0, columns=expected_columns,index=[0])
    bmi = input_data['bmi_category']

    for key, value in input_data.items():
        # Numeric columns
        if key == 'age':
            df['age'] = value
        elif key == 'number_of_dependants':
            df['number_of_dependants'] = value
        elif key == 'income_lakhs':
            df['income_lakhs'] = value
        elif key == 'genetical_risk':
            df['genetical_risk'] = value
        elif key == 'insurance_plan':
            if value == 'Bronze':
                df['insurance_plan'] = 1
            elif value == 'Silver':
                df['insurance_plan'] = 2
            elif value == 'Gold':
                df['insurance_plan'] = 3

        # Gender
        elif key == 'gender':
            if value == 'Male':
                df['gender_Male'] = 1

        # Region
        elif key == 'region':
            if value == 'Northwest':
                df['region_Northwest'] = 1
            elif value == 'Southeast':
                df['region_Southeast'] = 1
            elif value == 'Southwest':
                df['region_Southwest'] = 1

        # Marital status
        elif key == 'marital_status':
            if value == 'Unmarried':
                df['marital_status_Unmarried'] = 1

        # BMI category
        elif key == 'bmi_category':
            if value == 'Underweight':
                df['bmi_category_Underweight'] = 1
            elif value == 'Overweight':
                df['bmi_category_Overweight'] = 1
            elif value == 'Obesity':
                df['bmi_category_Obesity'] = 1

        # Smoking status
        elif key == 'smoking_status':
            if value == 'Occasional':
                df['smoking_status_Occasional'] = 1
            elif value == 'Regular':
                df['smoking_status_Regular'] = 1

        # Employment status
        elif key == 'employment_status':
            if value == 'Salaried':
                df['employment_status_Salaried'] = 1
            elif value == 'Self-Employed':
                df['employment_status_Self-Employed'] = 1

    df['normalized_risk_score'] = calculate_normalized_risk(input_data['medical_history'])
    df = handle_scaling(input_data['age'],df)
    return df

def handle_scaling(age, df):
    if age<=25:
        scaler_object = scaler_young
    else:
        scaler_object = scaler_rest

    cols_to_scale = scaler_object['cols_to_scale']
    scaler = scaler_object['scaler']

    df['income_level'] = None
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    df.drop('income_level',axis='columns',inplace=True)
    return df


def predict(input_data):
    input_df = preprocess(input_data)

    if input_data['age'] <= 25:
        prediction = model_young.predict(input_df)
    else:
        prediction = model_rest.predict(input_df)
    return int(prediction)