import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import PowerConsumptionForecaster
def diagnose_model_predictions(forecaster):
    print("\n" + "="*70)
    print("  MODEL DIAGNOSIS - Finding Why Predictions Are Low")
    print("="*70)
    print()
    print("📊 STEP 1: Training Data Analysis")
    print("-" * 70)
    train_stats = {
        'mean': forecaster.train_df['Global_active_power'].mean(),
        'min': forecaster.train_df['Global_active_power'].min(),
        'max': forecaster.train_df['Global_active_power'].max(),
        'std': forecaster.train_df['Global_active_power'].std(),
        '25%': forecaster.train_df['Global_active_power'].quantile(0.25),
        '50%': forecaster.train_df['Global_active_power'].quantile(0.50),
        '75%': forecaster.train_df['Global_active_power'].quantile(0.75),
    }
    print("Training Data Statistics (Global_active_power):")
    for key, value in train_stats.items():
        print(f"  {key:>6s}: {value:>8.3f} kW")
    print()
    print("🔮 STEP 2: Test Predictions Analysis")
    print("-" * 70)
    X_test = forecaster.test_df[forecaster.feature_cols]
    y_test = forecaster.test_df[forecaster.target_col]
    predictions = forecaster.model.predict(X_test, 
                                          num_iteration=forecaster.model.best_iteration)
    pred_stats = {
        'mean': np.mean(predictions),
        'min': np.min(predictions),
        'max': np.max(predictions),
        'std': np.std(predictions),
        '25%': np.percentile(predictions, 25),
        '50%': np.percentile(predictions, 50),
        '75%': np.percentile(predictions, 75),
    }
    print("Prediction Statistics:")
    for key, value in pred_stats.items():
        print(f"  {key:>6s}: {value:>8.3f} kW")
    print()
    print("⚖️  STEP 3: Actual vs Predicted Comparison")
    print("-" * 70)
    comparison = {
        'Actual Mean': y_test.mean(),
        'Predicted Mean': np.mean(predictions),
        'Difference': abs(y_test.mean() - np.mean(predictions)),
        'Ratio': np.mean(predictions) / y_test.mean() if y_test.mean() != 0 else 0
    }
    for key, value in comparison.items():
        print(f"  {key:>20s}: {value:>8.3f}")
    print()
    print("🔍 STEP 4: Diagnosis")
    print("-" * 70)
    issues = []
    if comparison['Ratio'] < 0.5:
        issues.append("❌ CRITICAL: Predictions are less than 50% of actual values!")
        issues.append("   Possible cause: Model not properly trained or wrong features")
    if pred_stats['std'] < 0.1:
        issues.append("❌ CRITICAL: Predictions have very low variance!")
        issues.append("   Possible cause: Model returning constant average value")
    if pred_stats['max'] < 1.0 and train_stats['max'] > 5.0:
        issues.append("❌ CRITICAL: Predictions are in wrong scale!")
        issues.append("   Possible cause: Data normalization issue or unit mismatch")
    if not issues:
        print("✅ No major issues detected!")
    else:
        print("Issues Found:")
        for issue in issues:
            print(f"  {issue}")
    print()
    print("📋 STEP 5: Sample Predictions (First 10)")
    print("-" * 70)
    print(f"{'Index':<8} {'Actual':>10} {'Predicted':>10} {'Difference':>12} {'Error %':>10}")
    print("-" * 70)
    for i in range(min(10, len(predictions))):
        actual = y_test.iloc[i]
        pred = predictions[i]
        diff = actual - pred
        error_pct = (diff / actual * 100) if actual != 0 else 0
        print(f"{i:<8} {actual:>10.3f} {pred:>10.3f} {diff:>12.3f} {error_pct:>9.1f}%")
    print()
    return {
        'train_stats': train_stats,
        'pred_stats': pred_stats,
        'comparison': comparison,
        'issues': issues
    }
def fix_low_predictions(filepath):
    print("\n" + "="*70)
    print("  ATTEMPTING TO FIX LOW PREDICTIONS")
    print("="*70)
    print()
    print("1. Loading data...")
    df = pd.read_csv(
        filepath,
        sep=';',
        parse_dates={'datetime': ['Date', 'Time']},
        low_memory=False,
        na_values=['?']
    )
    df.set_index('datetime', inplace=True)
    df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
    print(f"   Original data shape: {df.shape}")
    print(f"   Original mean: {df['Global_active_power'].mean():.3f} kW")
    print(f"   Original range: {df['Global_active_power'].min():.3f} - {df['Global_active_power'].max():.3f} kW")
    print()
    print("2. Testing different resampling methods...")
    df_mean = df[['Global_active_power']].resample('H').mean()
    print(f"   Resample with mean(): {df_mean['Global_active_power'].mean():.3f} kW")
    df_sum = df[['Global_active_power']].resample('H').sum()
    df_sum['Global_active_power'] = df_sum['Global_active_power'] / 60  
    print(f"   Resample with sum/60: {df_sum['Global_active_power'].mean():.3f} kW")
    df_first = df[['Global_active_power']].resample('H').first()
    print(f"   Resample with first(): {df_first['Global_active_power'].mean():.3f} kW")
    print()
    print("3. Checking for data quality issues...")
    nan_count = df_mean['Global_active_power'].isna().sum()
    zero_count = (df_mean['Global_active_power'] == 0).sum()
    print(f"   NaN values: {nan_count}")
    print(f"   Zero values: {zero_count}")
    if nan_count > 0 or zero_count > len(df_mean) * 0.1:
        print("   ⚠️  Warning: Many NaN or zero values detected!")
    print()
    print("4. Creating diagnostic visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].hist(df['Global_active_power'].dropna(), bins=50, 
                    edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Original Data Distribution (Per Minute)')
    axes[0, 0].set_xlabel('Power (kW)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(df['Global_active_power'].mean(), color='r', 
                      linestyle='--', linewidth=2, label=f'Mean: {df["Global_active_power"].mean():.2f}')
    axes[0, 0].legend()
    axes[0, 1].hist(df_mean['Global_active_power'].dropna(), bins=50, 
                    color='green', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Resampled Data Distribution (Hourly)')
    axes[0, 1].set_xlabel('Power (kW)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(df_mean['Global_active_power'].mean(), color='r', 
                      linestyle='--', linewidth=2, label=f'Mean: {df_mean["Global_active_power"].mean():.2f}')
    axes[0, 1].legend()
    sample_orig = df['Global_active_power'][:1440]  
    sample_resamp = df_mean['Global_active_power'][:24]
    axes[1, 0].plot(sample_orig.index, sample_orig.values, 
                   linewidth=1, alpha=0.7, label='Original (1-min)')
    axes[1, 0].set_title('Original Data (First 24 Hours)')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Power (kW)')
    axes[1, 0].legend()
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 1].plot(sample_resamp.index, sample_resamp.values, 
                   marker='o', linewidth=2, markersize=6, label='Resampled (hourly)')
    axes[1, 1].set_title('Resampled Data (First 24 Hours)')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Power (kW)')
    axes[1, 1].legend()
    axes[1, 1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig('./results/data_diagnosis.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved diagnostic plots to ./results/data_diagnosis.png")
    print()
    print("="*70)
    print("  RECOMMENDATIONS")
    print("="*70)
    print()
    if df_mean['Global_active_power'].mean() < 0.5:
        print("❌ CRITICAL ISSUE DETECTED!")
        print()
        print("Your resampled data has very low values (< 0.5 kW average).")
        print("This is NOT normal for household consumption!")
        print()
        print("Possible solutions:")
        print("  1. Check if data file is correct")
        print("  2. Verify the separator (should be ';' not ',')")
        print("  3. Check if Global_active_power column has correct values")
        print("  4. Make sure data is not already in Wh (should be in kW)")
        print()
    else:
        print("✅ Data looks reasonable!")
        print()
        print("If model still predicts low values, try:")
        print("  1. Increase num_boost_round in training")
        print("  2. Adjust learning_rate")
        print("  3. Add more features")
        print("  4. Check feature engineering")
        print()
    return df_mean
def test_single_prediction(forecaster, test_features):
    print("\n" + "="*70)
    print("  SINGLE PREDICTION TEST")
    print("="*70)
    print()
    print("Input Features:")
    for feature, value in test_features.items():
        print(f"  {feature:>25s}: {value}")
    print()
    test_df = pd.DataFrame([test_features])
    test_df = test_df[forecaster.feature_cols]  
    prediction = forecaster.model.predict(test_df, 
                                         num_iteration=forecaster.model.best_iteration)
    print("Prediction:")
    print(f"  {'Predicted Power':>25s}: {prediction[0]:.3f} kW")
    print()
    similar_hour = forecaster.train_df[forecaster.train_df['hour'] == test_features.get('hour', 0)]
    if len(similar_hour) > 0:
        print("Comparison with Training Data (Same Hour):")
        print(f"  {'Training Average':>25s}: {similar_hour['Global_active_power'].mean():.3f} kW")
        print(f"  {'Training Min':>25s}: {similar_hour['Global_active_power'].min():.3f} kW")
        print(f"  {'Training Max':>25s}: {similar_hour['Global_active_power'].max():.3f} kW")
        print()
    return prediction[0]
if __name__ == "__main__":
    filepath = 'resources/household_power_cleaned.csv'
    print("\n" + "="*70)
    print("  MODEL DEBUGGING TOOL")
    print("="*70)
    df_fixed = fix_low_predictions(filepath)
    print("\n" + "="*70)
    print("  RETRAINING MODEL WITH FIXED DATA")
    print("="*70)
    print()
    forecaster = PowerConsumptionForecaster(filepath)
    forecaster.df = df_fixed
    forecaster.prepare_data()
    forecaster.train_model()
    diagnosis = diagnose_model_predictions(forecaster)
    test_features = {
        'hour': 22,
        'dayofweek': 6,  
        'month': 12,
        'quarter': 4,
        'year': 2006,
        'dayofyear': 350,
        'weekofyear': 50,
        'lag_1h': 2.5,
        'lag_2h': 2.4,
        'lag_3h': 2.6,
        'lag_24h': 2.8,
        'lag_168h': 2.7,
        'rolling_mean_3h': 2.5,
        'rolling_mean_24h': 2.3,
        'rolling_std_24h': 0.5
    }
    prediction = test_single_prediction(forecaster, test_features)
    print("\n" + "="*70)
    print("  DEBUGGING COMPLETE")
    print("="*70)
    print()