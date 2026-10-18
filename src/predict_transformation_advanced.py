"""
Advanced Transformation Date Prediction Model

This model predicts when a MGUS patient will transform to Multiple Myeloma by:
1. Tracking biomarker progression patterns over time
2. Using patient-level train/test split (69 train, 20 test)
3. Focusing on 8 key biomarkers in order of importance
4. Considering the rate of change and trajectory of each biomarker
5. Testing multiple regression algorithms

Key Innovation: We model the VELOCITY and ACCELERATION of biomarker changes,
not just their absolute values, as disease progression shows characteristic 
patterns of increasing marker abnormality over time.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

# Key biomarkers in order of importance
KEY_BIOMARKERS = ['BAND', 'SKLRAT', 'CREA', 'ADCAL', 'HB', 'IGA', 'IGM', 'B2MG']

# Baseline model class for naive predictions
class NaiveModel:
    """Simple baseline model that always predicts a constant value"""
    def __init__(self, prediction_value):
        self.prediction_value = prediction_value
    
    def predict(self, X):
        if hasattr(X, '__len__'):
            return np.full(len(X), self.prediction_value)
        else:
            return np.array([self.prediction_value])

def load_cancer_data():
    """Load cancer patients data"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(script_dir), 'data', 'cancer_patients_89.csv')
    
    df = pd.read_csv(data_path)
    # Parse dates with mixed format (handles DD/MM/YY and DD/MM/YYYY)
    df['DateTestTaken'] = pd.to_datetime(df['DateTestTaken'], format='mixed', dayfirst=True)
    df['TransformationDate'] = pd.to_datetime(df['TransformationDate'], format='mixed', dayfirst=True)
    
    print(f"\nLoaded data: {len(df)} records from {df['UniqueID'].nunique()} patients")
    return df

def calculate_biomarker_progression_features(patient_data, transformation_date):
    """
    Calculate progression features for a patient's biomarkers over time.
    
    JUSTIFICATION FOR APPROACH:
    - Disease progression is characterized by CHANGING biomarker values, not static ones
    - We calculate: trend (slope), acceleration (curvature), volatility (variance)
    - Recent values are weighted more heavily as they're closer to transformation
    - Time-to-transformation is crucial: how fast are markers deteriorating?
    
    Returns features that capture:
    1. Most recent values (closest to transformation)
    2. Linear trend (rate of change)
    3. Acceleration (is change accelerating?)
    4. Volatility (how unstable are the markers?)
    5. Time-weighted progression scores
    """
    features = {}
    
    # Calculate days until transformation for each test
    patient_data['DaysToTransformation'] = (
        transformation_date - patient_data['DateTestTaken']
    ).dt.total_seconds() / (24 * 3600)
    
    # Only use tests BEFORE transformation
    pre_transformation = patient_data[patient_data['DaysToTransformation'] > 0].copy()
    
    if len(pre_transformation) == 0:
        return None
    
    # Sort by date (oldest to newest)
    pre_transformation = pre_transformation.sort_values('DateTestTaken')
    
    for biomarker in KEY_BIOMARKERS:
        marker_data = pre_transformation[pre_transformation['TestCode'] == biomarker].copy()
        
        if len(marker_data) == 0:
            # No data for this marker - use neutral values
            features[f'{biomarker}_recent'] = np.nan
            features[f'{biomarker}_trend'] = 0
            features[f'{biomarker}_acceleration'] = 0
            features[f'{biomarker}_volatility'] = 0
            features[f'{biomarker}_count'] = 0
            continue
        
        marker_data = marker_data.sort_values('DateTestTaken')
        # Convert Result to numeric, coerce errors to NaN
        marker_data['Result'] = pd.to_numeric(marker_data['Result'], errors='coerce')
        # Drop rows with NaN results
        marker_data = marker_data.dropna(subset=['Result'])
        
        if len(marker_data) == 0:
            # No valid numeric data
            features[f'{biomarker}_recent'] = np.nan
            features[f'{biomarker}_trend'] = 0
            features[f'{biomarker}_acceleration'] = 0
            features[f'{biomarker}_volatility'] = 0
            features[f'{biomarker}_count'] = 0
            continue
            
        values = marker_data['Result'].values
        dates = marker_data['DateTestTaken'].values
        days_to_trans = marker_data['DaysToTransformation'].values
        
        # Feature 1: Most recent value (closest to transformation)
        features[f'{biomarker}_recent'] = values[-1]
        
        # Feature 2: Count of measurements
        features[f'{biomarker}_count'] = len(values)
        
        # Feature 3: Linear trend (rate of change over time)
        if len(values) >= 2:
            # Use days from first test as x-axis
            days_from_start = (dates - dates[0]).astype('timedelta64[D]').astype(float)
            # Fit linear regression: slope tells us rate of change
            try:
                trend = np.polyfit(days_from_start, values, 1)[0]
            except (np.linalg.LinAlgError, ValueError, Warning):
                trend = 0
            features[f'{biomarker}_trend'] = trend
        else:
            features[f'{biomarker}_trend'] = 0
        
        # Feature 4: Acceleration (second derivative - is trend accelerating?)
        if len(values) >= 3:
            days_from_start = (dates - dates[0]).astype('timedelta64[D]').astype(float)
            # Fit quadratic: second coefficient is acceleration
            try:
                poly_coeffs = np.polyfit(days_from_start, values, 2)
                features[f'{biomarker}_acceleration'] = poly_coeffs[0]
            except (np.linalg.LinAlgError, ValueError, Warning):
                features[f'{biomarker}_acceleration'] = 0
        else:
            features[f'{biomarker}_acceleration'] = 0
        
        # Feature 5: Volatility (standard deviation - how unstable is the marker?)
        if len(values) >= 2:
            features[f'{biomarker}_volatility'] = np.std(values)
        else:
            features[f'{biomarker}_volatility'] = 0
        
        # Feature 6: Weighted recent trend (last 6 months have more weight)
        recent_mask = days_to_trans <= 180  # Last 6 months
        if recent_mask.sum() >= 2:
            recent_values = values[recent_mask]
            recent_dates = dates[recent_mask]
            recent_days = (recent_dates - recent_dates[0]).astype('timedelta64[D]').astype(float)
            try:
                recent_trend = np.polyfit(recent_days, recent_values, 1)[0]
            except (np.linalg.LinAlgError, ValueError):
                recent_trend = features[f'{biomarker}_trend']
            features[f'{biomarker}_recent_trend'] = recent_trend
        else:
            features[f'{biomarker}_recent_trend'] = features[f'{biomarker}_trend']
    
    # Meta-features: overall disease progression indicators
    features['total_measurements'] = len(pre_transformation)
    features['monitoring_duration_days'] = (
        pre_transformation['DateTestTaken'].max() - 
        pre_transformation['DateTestTaken'].min()
    ).total_seconds() / (24 * 3600)
    
    # Average days to transformation (how close are we monitoring to the event?)
    features['avg_days_to_transformation'] = pre_transformation['DaysToTransformation'].mean()
    features['min_days_to_transformation'] = pre_transformation['DaysToTransformation'].min()
    
    return features

def prepare_training_data(df):
    """
    Prepare PROSPECTIVE training data using 6-month lookback window
    
    KEY STRATEGY FOR PROSPECTIVE PREDICTION:
    - Use tests from 6-12 months BEFORE transformation for training
    - This simulates real-world scenario: predict future transformation from current tests
    - Models learn patterns from earlier disease stage, not end-stage
    - Target: predict months remaining until transformation from that earlier point
    
    Example: Patient transforms on Jan 1, 2024
    - Use tests from July 1 - Dec 31, 2023 (last 6 months before transformation)
    - Ignore tests from Jan-Jun 2023 (too early, may not be relevant)
    - Target: 6 months (time from Dec 31, 2023 to transformation)
    """
    print("\n" + "="*80)
    print("PREPARING PROSPECTIVE TRAINING DATA (6-MONTH LOOKBACK)")
    print("="*80)
    
    # Get unique patients
    patients = df['UniqueID'].unique()
    print(f"\nTotal patients: {len(patients)}")
    
    # Build feature matrix
    X_data = []
    y_data = []
    patient_ids = []
    skipped_patients = 0
    
    LOOKBACK_MONTHS = 6  # Use data from 6 months before transformation
    LOOKBACK_DAYS = LOOKBACK_MONTHS * 30.44
    
    for patient_id in patients:
        patient_data = df[df['UniqueID'] == patient_id].copy()
        transformation_date = patient_data['TransformationDate'].iloc[0]
        
        # Calculate days to transformation for each test
        patient_data['DaysToTransformation'] = (
            transformation_date - patient_data['DateTestTaken']
        ).dt.total_seconds() / (24 * 3600)
        
        # Only use tests that are 6+ months BEFORE transformation
        # This simulates making a prediction 6 months in advance
        early_tests = patient_data[patient_data['DaysToTransformation'] >= LOOKBACK_DAYS].copy()
        
        if len(early_tests) == 0:
            skipped_patients += 1
            continue
        
        # Use the MOST RECENT test from this early period as our "prediction point"
        # This represents the latest information we'd have 6 months before transformation
        prediction_point_date = early_tests['DateTestTaken'].max()
        
        # Get all data up to this prediction point
        data_up_to_prediction = patient_data[patient_data['DateTestTaken'] <= prediction_point_date].copy()
        
        # Calculate how many days from prediction point to transformation
        days_until_transformation = (transformation_date - prediction_point_date).total_seconds() / (24 * 3600)
        
        # Calculate progression features using data up to prediction point
        features = calculate_biomarker_progression_features(data_up_to_prediction, transformation_date)
        
        if features is None:
            skipped_patients += 1
            continue
        
        X_data.append(features)
        y_data.append(days_until_transformation)
        patient_ids.append(patient_id)
    
    print(f"Usable patients: {len(X_data)}")
    print(f"Skipped patients (insufficient early data): {skipped_patients}")
    
    # Convert to DataFrame
    X = pd.DataFrame(X_data)
    y = pd.Series(y_data, name='DaysUntilTransformation')
    patient_ids = pd.Series(patient_ids, name='UniqueID')
    
    print(f"\nFeatures created: {X.shape[1]} features for {X.shape[0]} patients")
    print(f"\nTarget variable (Days Until Transformation):")
    print(f"  Mean: {y.mean():.1f} days ({y.mean()/30:.1f} months)")
    print(f"  Median: {y.median():.1f} days ({y.median()/30:.1f} months)")
    print(f"  Range: {y.min():.1f} to {y.max():.1f} days")
    
    # Show feature importance order
    print(f"\nFeature categories created:")
    print(f"  - Recent values: Most recent biomarker measurement")
    print(f"  - Trends: Rate of change over time (slope)")
    print(f"  - Acceleration: Is the trend accelerating? (curvature)")
    print(f"  - Volatility: Stability of measurements (std dev)")
    print(f"  - Recent trends: Weighted trend for last 6 months")
    print(f"  - Meta-features: Overall monitoring patterns")
    
    return X, y, patient_ids

def patient_level_split(X, y, patient_ids, test_size=20, random_state=42):
    """
    Split data at patient level (69 train, 20 test)
    """
    print("\n" + "="*80)
    print("PATIENT-LEVEL TRAIN/TEST SPLIT")
    print("="*80)
    
    # Split by patient
    unique_patients = patient_ids.unique()
    train_patients, test_patients = train_test_split(
        unique_patients, 
        test_size=test_size, 
        random_state=random_state
    )
    
    train_mask = patient_ids.isin(train_patients)
    test_mask = patient_ids.isin(test_patients)
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    print(f"\nTraining set: {len(train_patients)} patients")
    print(f"Test set: {len(test_patients)} patients")
    print(f"\nTraining target distribution:")
    print(f"  Mean: {y_train.mean():.1f} days ({y_train.mean()/30:.1f} months)")
    print(f"  Median: {y_train.median():.1f} days")
    print(f"\nTest target distribution:")
    print(f"  Mean: {y_test.mean():.1f} days ({y_test.mean()/30:.1f} months)")
    print(f"  Median: {y_test.median():.1f} days")
    
    return X_train, X_test, y_train, y_test, test_patients

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train multiple regression models and compare performance
    """
    print("\n" + "="*80)
    print("TRAINING MULTIPLE REGRESSION MODELS")
    print("="*80)
    
    # Handle missing values
    X_train_filled = X_train.fillna(X_train.median())
    X_test_filled = X_test.fillna(X_train.median())  # Use train median for test
    
    # Scale features (important for some algorithms)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_filled)
    X_test_scaled = scaler.transform(X_test_filled)
    
    # Define models to test
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Elastic Net': ElasticNet(alpha=1.0, l1_ratio=0.5),
        'Support Vector Regression': SVR(kernel='rbf', C=100, gamma='scale'),
    }
    
    results = []
    trained_models = {}
    
    # Add baseline models: Naïve and Seasonal-Naïve
    # Naïve: Predict the mean of training set
    print(f"\n{'='*60}")
    print(f"Training: Naïve (Mean Baseline)")
    print(f"{'='*60}")
    
    naive_prediction = y_train.mean()
    y_train_naive = np.full(len(y_train), naive_prediction)
    y_test_naive = np.full(len(y_test), naive_prediction)
    
    train_mae = mean_absolute_error(y_train, y_train_naive)
    test_mae = mean_absolute_error(y_test, y_test_naive)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_naive))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_naive))
    train_r2 = r2_score(y_train, y_train_naive)
    test_r2 = r2_score(y_test, y_test_naive)
    
    print(f"\nNaïve prediction: {naive_prediction:.1f} days ({naive_prediction/30:.1f} months)")
    print(f"\nTraining Performance:")
    print(f"  MAE: {train_mae:.1f} days ({train_mae/30:.1f} months)")
    print(f"  RMSE: {train_rmse:.1f} days ({train_rmse/30:.1f} months)")
    print(f"  R²: {train_r2:.4f}")
    
    print(f"\nTest Performance:")
    print(f"  MAE: {test_mae:.1f} days ({test_mae/30:.1f} months)")
    print(f"  RMSE: {test_rmse:.1f} days ({test_rmse/30:.1f} months)")
    print(f"  R²: {test_r2:.4f}")
    
    results.append({
        'Model': 'Naïve (Mean)',
        'Train_MAE_days': train_mae,
        'Train_MAE_months': train_mae/30,
        'Test_MAE_days': test_mae,
        'Test_MAE_months': test_mae/30,
        'Train_RMSE_days': train_rmse,
        'Test_RMSE_days': test_rmse,
        'Train_R2': train_r2,
        'Test_R2': test_r2,
        'Predictions': y_test_naive
    })
    
    # Seasonal-Naïve: Predict the median of training set
    print(f"\n{'='*60}")
    print(f"Training: Seasonal-Naïve (Median Baseline)")
    print(f"{'='*60}")
    
    seasonal_naive_prediction = y_train.median()
    y_train_seasonal = np.full(len(y_train), seasonal_naive_prediction)
    y_test_seasonal = np.full(len(y_test), seasonal_naive_prediction)
    
    train_mae = mean_absolute_error(y_train, y_train_seasonal)
    test_mae = mean_absolute_error(y_test, y_test_seasonal)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_seasonal))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_seasonal))
    train_r2 = r2_score(y_train, y_train_seasonal)
    test_r2 = r2_score(y_test, y_test_seasonal)
    
    print(f"\nSeasonal-Naïve prediction: {seasonal_naive_prediction:.1f} days ({seasonal_naive_prediction/30:.1f} months)")
    print(f"\nTraining Performance:")
    print(f"  MAE: {train_mae:.1f} days ({train_mae/30:.1f} months)")
    print(f"  RMSE: {train_rmse:.1f} days ({train_rmse/30:.1f} months)")
    print(f"  R²: {train_r2:.4f}")
    
    print(f"\nTest Performance:")
    print(f"  MAE: {test_mae:.1f} days ({test_mae/30:.1f} months)")
    print(f"  RMSE: {test_rmse:.1f} days ({test_rmse/30:.1f} months)")
    print(f"  R²: {test_r2:.4f}")
    
    results.append({
        'Model': 'Seasonal-Naïve (Median)',
        'Train_MAE_days': train_mae,
        'Train_MAE_months': train_mae/30,
        'Test_MAE_days': test_mae,
        'Test_MAE_months': test_mae/30,
        'Train_RMSE_days': train_rmse,
        'Test_RMSE_days': test_rmse,
        'Train_R2': train_r2,
        'Test_R2': test_r2,
        'Predictions': y_test_seasonal
    })
    
    # Store baseline models
    trained_models['Naïve (Mean)'] = {
        'model': NaiveModel(naive_prediction),
        'scaler': None,
        'use_scaled': False
    }
    
    trained_models['Seasonal-Naïve (Median)'] = {
        'model': NaiveModel(seasonal_naive_prediction),
        'scaler': None,
        'use_scaled': False
    }
    
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"{'='*60}")
        
        # Some models need scaled data
        if name in ['Ridge Regression', 'Lasso Regression', 'Elastic Net', 'Support Vector Regression']:
            X_tr = X_train_scaled
            X_te = X_test_scaled
        else:
            X_tr = X_train_filled
            X_te = X_test_filled
        
        # Train
        model.fit(X_tr, y_train)
        
        # Predict
        y_train_pred = model.predict(X_tr)
        y_test_pred = model.predict(X_te)
        
        # Evaluate
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        print(f"\nTraining Performance:")
        print(f"  MAE: {train_mae:.1f} days ({train_mae/30:.1f} months)")
        print(f"  RMSE: {train_rmse:.1f} days ({train_rmse/30:.1f} months)")
        print(f"  R²: {train_r2:.4f}")
        
        print(f"\nTest Performance:")
        print(f"  MAE: {test_mae:.1f} days ({test_mae/30:.1f} months)")
        print(f"  RMSE: {test_rmse:.1f} days ({test_rmse/30:.1f} months)")
        print(f"  R²: {test_r2:.4f}")
        
        results.append({
            'Model': name,
            'Train_MAE_days': train_mae,
            'Train_MAE_months': train_mae/30,
            'Test_MAE_days': test_mae,
            'Test_MAE_months': test_mae/30,
            'Train_RMSE_days': train_rmse,
            'Test_RMSE_days': test_rmse,
            'Train_R2': train_r2,
            'Test_R2': test_r2,
            'Predictions': y_test_pred
        })
        
        trained_models[name] = {
            'model': model,
            'scaler': scaler if name in ['Ridge Regression', 'Lasso Regression', 'Elastic Net', 'Support Vector Regression'] else None,
            'use_scaled': name in ['Ridge Regression', 'Lasso Regression', 'Elastic Net', 'Support Vector Regression']
        }
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Test_MAE_days')
    
    return results_df, trained_models, X_train_filled, scaler

def show_detailed_predictions(y_test, results_df, test_patients, df):
    """
    Show detailed predictions for each test patient
    Shows prediction from 6-month lookback point (realistic prospective scenario)
    """
    print("\n" + "="*80)
    print("DETAILED PREDICTIONS FOR TEST PATIENTS")
    print("="*80)
    
    # Get best model predictions
    best_model_name = results_df.iloc[0]['Model']
    best_predictions = results_df.iloc[0]['Predictions']
    
    print(f"\nUsing best model: {best_model_name}")
    print(f"Test MAE: {results_df.iloc[0]['Test_MAE_months']:.2f} months")
    
    print(f"\n📊 PROSPECTIVE PREDICTION SCENARIO (6-month lookback):")
    print(f"For each patient, we used tests from 6+ months BEFORE transformation")
    print(f"This simulates real-world prediction: Can we predict future transformation?")
    
    # Get actual transformation dates for test patients
    test_patient_info = []
    LOOKBACK_DAYS = 6 * 30.44
    
    for i, patient_id in enumerate(test_patients):
        patient_data = df[df['UniqueID'] == patient_id].copy()
        actual_transformation_date = patient_data['TransformationDate'].iloc[0]
        
        # Calculate days to transformation for each test
        patient_data['DaysToTransformation'] = (
            actual_transformation_date - patient_data['DateTestTaken']
        ).dt.total_seconds() / (24 * 3600)
        
        # Find the prediction point (last test 6+ months before transformation)
        early_tests = patient_data[patient_data['DaysToTransformation'] >= LOOKBACK_DAYS]
        if len(early_tests) > 0:
            prediction_point_date = early_tests['DateTestTaken'].max()
            actual_days_from_prediction = (actual_transformation_date - prediction_point_date).total_seconds() / (24 * 3600)
        else:
            # Fallback to earliest test
            prediction_point_date = patient_data['DateTestTaken'].min()
            actual_days_from_prediction = (actual_transformation_date - prediction_point_date).total_seconds() / (24 * 3600)
        
        predicted_days = best_predictions[i]
        error_days = abs(actual_days_from_prediction - predicted_days)
        
        test_patient_info.append({
            'PatientID': patient_id,
            'Prediction_Point': prediction_point_date.strftime('%Y-%m-%d'),
            'Actual_Transformation': actual_transformation_date.strftime('%Y-%m-%d'),
            'Actual_Days': actual_days_from_prediction,
            'Actual_Months': actual_days_from_prediction / 30.44,
            'Predicted_Days': predicted_days,
            'Predicted_Months': predicted_days / 30.44,
            'Error_Days': error_days,
            'Error_Months': error_days / 30.44,
        })
    
    predictions_df = pd.DataFrame(test_patient_info)
    predictions_df = predictions_df.sort_values('Actual_Months')
    
    print("\n" + "-"*130)
    print(f"{'Patient':<15} {'Prediction Point':<17} {'Actual Transform':<18} {'Actual(mo)':<12} {'Predicted(mo)':<15} {'Error(mo)':<12}")
    print("-"*130)
    
    for _, row in predictions_df.iterrows():
        print(f"{str(row['PatientID']):<15} {row['Prediction_Point']:<17} {row['Actual_Transformation']:<18} "
              f"{row['Actual_Months']:>11.1f} {row['Predicted_Months']:>14.1f} {row['Error_Months']:>11.1f}")
    
    print("-"*130)
    print(f"\nSummary Statistics:")
    print(f"  Mean Error: {predictions_df['Error_Months'].mean():.2f} months")
    print(f"  Median Error: {predictions_df['Error_Months'].median():.2f} months")
    print(f"  Best Prediction: {predictions_df['Error_Months'].min():.2f} months error")
    print(f"  Worst Prediction: {predictions_df['Error_Months'].max():.2f} months error")
    print(f"\n  Mean Actual Time to Transformation: {predictions_df['Actual_Months'].mean():.1f} months")
    print(f"  Mean Predicted Time to Transformation: {predictions_df['Predicted_Months'].mean():.1f} months")
    
    return predictions_df

def explain_model_approach():
    """
    Print detailed justification of the modeling approach
    """
    print("\n" + "="*80)
    print("MODEL APPROACH JUSTIFICATION")
    print("="*80)
    
    print("""
CORE PHILOSOPHY:
MGUS to Myeloma transformation is a PROGRESSIVE disease. Static biomarker values
alone don't tell the full story - we need to understand HOW FAST and IN WHAT DIRECTION
the biomarkers are changing.

FEATURE ENGINEERING STRATEGY:

1. RECENT VALUES (Most Recent Measurement)
   - Captures current disease state
   - Closest measurement to transformation is most informative
   
2. TREND (Linear Rate of Change)
   - Calculated as slope over time: are markers getting worse?
   - Positive/negative trends indicate progression direction
   - BAND increasing over time signals worsening paraprotein
   - HB decreasing signals developing anemia
   
3. ACCELERATION (Second Derivative)
   - Is the trend itself accelerating?
   - Captures non-linear progression patterns
   - Disease often accelerates as it approaches transformation
   
4. VOLATILITY (Standard Deviation)
   - How unstable are the measurements?
   - High volatility may indicate disease instability
   - Stable values suggest slower progression
   
5. RECENT TREND (Last 6 Months Weighted)
   - Recent changes are more predictive than old patterns
   - Gives higher weight to measurements closer to transformation
   
6. META-FEATURES
   - Monitoring duration: longer history = more data
   - Measurement frequency: more tests = better trend estimates
   - Time to transformation: how close are we monitoring?

BIOMARKER IMPORTANCE ORDER:
1. BAND (Paraprotein) - Most critical myeloma marker
2. SKLRAT (Serum Free Light Chain Ratio) - Diagnostic criterion
3. CREA (Creatinine) - Kidney function (CRAB criteria)
4. ADCAL (Adjusted Calcium) - Hypercalcemia (CRAB criteria)
5. HB (Hemoglobin) - Anemia (CRAB criteria)
6. IGA, IGM - Immunoglobulin suppression
7. B2MG (Beta-2 Microglobulin) - Prognosis marker

PREDICTION TARGET:
We predict "Days from earliest test to transformation" rather than absolute dates.
This makes the model generalizable: given a new blood sample, we calculate progression
features and predict how many days until transformation.

MODEL SELECTION:
We test 6 different algorithms:
- Tree-based: Random Forest, Gradient Boosting (handle non-linearity well)
- Linear: Ridge, Lasso, ElasticNet (interpretable, regularized)
- SVM: Support Vector Regression (good for complex patterns)

CLINICAL UTILITY:
Once trained, the model can:
1. Take ANY blood test sample with the 8 key biomarkers
2. Calculate progression features (if historical data exists)
3. Predict time until transformation
4. Help clinicians identify high-risk patients for closer monitoring
""")

def main():
    """Main execution"""
    print("\n" + "="*80)
    print("ADVANCED TRANSFORMATION DATE PREDICTION")
    print("Training: 69 patients | Testing: 20 patients")
    print("="*80)
    
    # Explain approach
    explain_model_approach()
    
    # Load data
    df = load_cancer_data()
    
    # Prepare features
    X, y, patient_ids = prepare_training_data(df)
    
    # Split into train/test (69/20)
    X_train, X_test, y_train, y_test, test_patients = patient_level_split(
        X, y, patient_ids, test_size=20
    )
    
    # Train models
    results_df, trained_models, X_train_filled, scaler = train_and_evaluate_models(
        X_train, X_test, y_train, y_test
    )
    
    # Show comparison
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    print("\n" + results_df[['Model', 'Test_MAE_months', 'Test_RMSE_days', 'Test_R2']].to_string(index=False))
    
    # Show detailed predictions
    predictions_df = show_detailed_predictions(y_test, results_df, test_patients, df)
    
    # Save results
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    results_df_save = results_df.drop('Predictions', axis=1)
    results_df_save.to_csv(os.path.join(output_dir, 'transformation_prediction_models.csv'), index=False)
    predictions_df.to_csv(os.path.join(output_dir, 'transformation_predictions_detailed.csv'), index=False)
    
    # Save trained models
    print(f"\n💾 Saving trained models...")
    for model_name, model_info in trained_models.items():
        # Create safe filename
        safe_name = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        model_path = os.path.join(output_dir, f'model_advanced_{safe_name}.pkl')
        
        # Save model with scaler if applicable
        model_data = {
            'model': model_info['model'],
            'scaler': model_info.get('scaler'),
            'use_scaled': model_info.get('use_scaled', False),
            'feature_names': X_train_filled.columns.tolist(),
            'model_name': model_name
        }
        joblib.dump(model_data, model_path)
        print(f"   ✓ {model_name} -> {safe_name}.pkl")
    
    print(f"\n✅ Results saved to output folder")
    print(f"   - transformation_prediction_models.csv")
    print(f"   - transformation_predictions_detailed.csv")
    print(f"   - {len(trained_models)} model files (.pkl)")
    
    # Show best model
    print("\n" + "="*80)
    print("🏆 BEST MODEL")
    print("="*80)
    best = results_df.iloc[0]
    print(f"\nModel: {best['Model']}")
    print(f"Test MAE: {best['Test_MAE_months']:.2f} months ({best['Test_MAE_days']:.0f} days)")
    print(f"Test RMSE: {best['Test_RMSE_days']/30:.2f} months")
    print(f"Test R²: {best['Test_R2']:.4f}")
    
    return results_df, predictions_df, trained_models

if __name__ == "__main__":
    results_df, predictions_df, trained_models = main()
