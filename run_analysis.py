from model import PowerConsumptionForecaster
if __name__ == "__main__":
    filepath = 'data/household_power_cleaned.csv'
    print("\n" + "="*60)
    print("  HOUSEHOLD POWER CONSUMPTION FORECASTING WITH LIGHTGBM")
    print("="*60)
    print()
    forecaster = PowerConsumptionForecaster(filepath)
    metrics, predictions, feature_importance = forecaster.run_complete_analysis(
        save_path='./results'
    )
    print("\n📊 Final Model Performance:")
    print(f"  - RMSE: {metrics['rmse']:.4f} kW")
    print(f"  - MAE: {metrics['mae']:.4f} kW")
    print(f"  - R² Score: {metrics['r2']:.4f}")
    print(f"  - MAPE: {metrics['mape']:.2f}%")
    print()
    print("🎉 All done! Your model is ready to use!")
    print()