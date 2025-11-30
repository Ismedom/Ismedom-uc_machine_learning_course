"""
test_cost_prediction.py - Test script to demonstrate the new cost prediction feature
"""
from model import PowerConsumptionForecaster
from cost_analyzer import PowerCostAnalyzer
import pandas as pd
def test_cost_prediction():
    """Test the new cost prediction feature using the trained model."""
    print("🔌 Testing Cost Prediction Feature")
    print("=" * 50)
    print("\n1. Loading forecasting model...")
    filepath = 'resources/household_power_cleaned.csv'
    try:
        forecaster = PowerConsumptionForecaster(filepath)
        forecaster.load_data()
        forecaster.explore_data(save_path='./results')
        forecaster.prepare_data()
        forecaster.train_model()
        predictions, metrics, feature_importance = forecaster.evaluate_model()
        forecaster.visualize_results(predictions, save_path='./results')
        print("✓ Model loaded and trained successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None
    print("\n2. Predicting future costs...")
    analyzer = PowerCostAnalyzer(
        price_per_kwh=0.25,  
        peak_hours=(18, 22),  
        peak_multiplier=2.0   
    )
    analyzer.set_thresholds(
        hourly_kW=4.0,
        daily_kWh=60.0,
        weekly_kWh=350.0,
        monthly_kWh=1400.0
    )
    try:
        projections, future_df = analyzer.predict_future_costs(forecaster, days_ahead=30)
        print("✓ Cost prediction completed successfully")
        print("\n📊 Cost Projections for Next 30 Days:")
        print("-" * 40)
        print(f"Daily Average Cost: ${projections['daily_avg_cost']:.2f}")
        print(f"Monthly Total Cost: ${projections['monthly_total_cost']:.2f}")
        print(f"Yearly Total Cost: ${projections['yearly_total_cost']:.2f}")
        print(f"Monthly Average Usage: {projections['monthly_avg_kwh']:.1f} kWh")
        print(f"Yearly Total Usage: {projections['yearly_total_kwh']:.1f} kWh")
        print(f"Peak Hours Daily: {projections['peak_hours_daily']:.1f} kWh")
        print(f"Off-Peak Hours Daily: {projections['off_peak_hours_daily']:.1f} kWh")
        aggregated, violations = analyzer.generate_cost_report(future_df)
        print("\n📈 Analysis Summary:")
        print(f"Total Predicted Cost (30 days): ${analyzer.cost_df['hourly_cost'].sum():.2f}")
        print(f"Total Predicted Usage (30 days): {analyzer.cost_df['power_kW'].sum():.1f} kWh")
        print(f"Average Daily Cost: ${aggregated['daily']['total_cost'].mean():.2f}")
        print(f"Threshold Violations: {sum(len(v) for v in violations.values())}")
        analyzer.create_visualizations('./results')
        print("\n✓ Visualizations created in ./results/")
        return analyzer, projections, aggregated, violations
    except Exception as e:
        print(f"✗ Error in cost prediction: {e}")
        return None
def demo_user_friendly_features():
    """Demonstrate the user-friendly cost prediction features."""
    print("\n🎯 User-Friendly Cost Prediction Demo")
    print("=" * 50)
    print("""
✨ Key Features for End Users:

1. 🤖 Automatic Predictions
   - No need to input complex data
   - Uses your trained AI model
   - Predicts future power consumption automatically

2. 💰 Cost Projections
   - See costs for next month, year, etc.
   - Understand peak vs off-peak usage
   - Customizable pricing settings

3. 📊 Easy Configuration
   - Set your electricity price per kWh
   - Configure peak hours (when rates are higher)
   - Set usage thresholds for monitoring

4. 📈 Visual Reports
   - Daily cost trends
   - Monthly comparisons
   - Peak/off-peak breakdowns
   - Hourly usage patterns

5. 🚨 Smart Alerts
   - Automatic threshold monitoring
   - Violation detection and reporting
   - Cost-saving recommendations

Example User Workflow:
1. Train your model (done once)
2. Set your electricity rates and preferences
3. Click "Predict Future Costs"
4. View detailed cost projections and insights
5. Adjust settings as needed

No technical knowledge required! 🎉
    """)
if __name__ == "__main__":
    result = test_cost_prediction()
    demo_user_friendly_features()
    if result:
        print("\n✅ Cost prediction feature working perfectly!")
        print("Your users can now easily predict future electricity costs! 🎊")
    else:
        print("\n⚠️  Cost prediction test failed - check your setup")