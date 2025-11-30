"""
test_cost_integration.py - Test script to integrate cost analysis with the existing model
"""
from model import PowerConsumptionForecaster
from cost_analyzer import PowerCostAnalyzer
import pandas as pd
import numpy as np
def test_cost_integration():
    """Test the integration between the forecasting model and cost analyzer."""
    print("🔌 Testing Cost Analyzer Integration")
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
        return None, None
    print("\n2. Preparing data for cost analysis...")
    test_df = forecaster.test_df.copy()
    test_df['Global_active_power'] = test_df[forecaster.target_col]
    test_df.reset_index(inplace=True)
    test_df.set_index('datetime', inplace=True)
    print(f"✓ Using {len(test_df)} hours of test data for cost analysis")
    print("\n3. Setting up cost analyzer...")
    analyzer = PowerCostAnalyzer(
        price_per_kwh=0.22,  
        peak_hours=(18, 22),  
        peak_multiplier=1.8   
    )
    analyzer.set_thresholds(
        hourly_kW=4.0,      
        daily_kWh=60.0,     
        weekly_kWh=350.0,   
        monthly_kWh=1400.0  
    )
    print("✓ Cost analyzer configured")
    print("\n4. Generating cost analysis report...")
    try:
        aggregated, violations = analyzer.generate_cost_report(test_df)
        print("✓ Cost report generated successfully")
    except Exception as e:
        print(f"✗ Error generating cost report: {e}")
        return analyzer, None
    print("\n5. Creating cost visualizations...")
    try:
        analyzer.create_visualizations('./results')
        print("✓ Visualizations created successfully")
    except Exception as e:
        print(f"✗ Error creating visualizations: {e}")
    print("\n6. Integration example complete!")
    print("\n" + "=" * 50)
    print("🎉 Integration test successful!")
    print("\nKey results:")
    print(f"  - Analyzed {len(test_df)} hours of power consumption")
    print(f"  - Total violations detected: {sum(len(v) for v in violations.values())}")
    print("  - Visualizations saved to ./results/")
    print("\nYou can now use the cost analyzer with your forecasting model!")
    print("=" * 50)
    return analyzer, aggregated, violations
def demo_usage():
    """Demonstrate typical usage of the cost analyzer with model predictions."""
    print("\n🔌 Cost Analyzer Demo with Model Predictions")
    print("=" * 50)
    print("""
# Example usage in your application:

from model import PowerConsumptionForecaster
from cost_analyzer import PowerCostAnalyzer

# 1. Load your trained model
forecaster = PowerConsumptionForecaster()
forecaster.load_model('./ai_models/power_forecaster.pkl')

# 2. Make predictions for future periods
# (Your prediction logic here)

# 3. Analyze costs of predicted consumption
predicted_df = pd.DataFrame({
    'Global_active_power': predicted_values,  # Your predictions
}, index=future_dates)  # Future datetime index

analyzer = PowerCostAnalyzer(price_per_kwh=0.20)
aggregated, violations = analyzer.generate_cost_report(predicted_df)

# 4. Create visualizations
analyzer.create_visualizations('./results')

# 5. Access results
total_cost = aggregated['monthly']['total_cost'].sum()
peak_cost_pct = (aggregated['daily']['total_cost'] * aggregated['daily']['peak_fraction']).sum() / aggregated['daily']['total_cost'].sum() * 100
    """)
if __name__ == "__main__":
    analyzer, aggregated, violations = test_cost_integration()
    demo_usage()
    print("\n✅ All tests completed!")