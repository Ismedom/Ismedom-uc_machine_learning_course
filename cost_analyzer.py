import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')
class PowerCostAnalyzer:
    def __init__(self, price_per_kwh=0.20, peak_hours=(18, 22), peak_multiplier=1.5):
        self.price_per_kwh = price_per_kwh
        self.peak_hours = peak_hours
        self.peak_multiplier = peak_multiplier
        self.df = None
        self.cost_df = None
        self.thresholds = {
            'hourly_kW': 5.0,
            'daily_kWh': 50.0,
            'weekly_kWh': 300.0,
            'monthly_kWh': 1200.0
        }
    def load_data(self, df):
        self.df = df.copy()
        if not isinstance(self.df.index, pd.DatetimeIndex):
            self.df.index = pd.to_datetime(self.df.index)
        if 'Global_active_power' not in self.df.columns:
            raise ValueError("DataFrame must contain 'Global_active_power' column")
        self.cost_df = self.df[['Global_active_power']].copy()
        self.cost_df.columns = ['power_kW']
        self.cost_df['hour'] = self.cost_df.index.hour
        self.cost_df['day'] = self.cost_df.index.date
        self.cost_df['month'] = self.cost_df.index.month
        self.cost_df['year'] = self.cost_df.index.year
        self.cost_df['weekday'] = self.cost_df.index.weekday
        self.cost_df['is_peak'] = self.cost_df['hour'].apply(
            lambda h: self.peak_hours[0] <= h < self.peak_hours[1]
        )
        self.cost_df['price_per_kwh'] = np.where(
            self.cost_df['is_peak'],
            self.price_per_kwh * self.peak_multiplier,
            self.price_per_kwh
        )
        self.cost_df['hourly_cost'] = self.cost_df['power_kW'] * self.cost_df['price_per_kwh']
        return self.cost_df
    def set_thresholds(self, hourly_kW=None, daily_kWh=None, weekly_kWh=None, monthly_kWh=None):
        if hourly_kW is not None:
            self.thresholds['hourly_kW'] = hourly_kW
        if daily_kWh is not None:
            self.thresholds['daily_kWh'] = daily_kWh
        if weekly_kWh is not None:
            self.thresholds['weekly_kWh'] = weekly_kWh
        if monthly_kWh is not None:
            self.thresholds['monthly_kWh'] = monthly_kWh
    def calculate_costs_by_period(self, period='D'):
        if self.cost_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        agg_dict = {
            'power_kW': 'sum',
            'hourly_cost': 'sum',
            'is_peak': 'mean'
        }
        aggregated = self.cost_df.resample(period).agg(agg_dict)
        aggregated.columns = ['total_kWh', 'total_cost', 'peak_fraction']
        if period == 'H':
            aggregated['avg_power_kW'] = aggregated['total_kWh']
            aggregated['avg_cost_per_hour'] = aggregated['total_cost']
        else:
            hours_per_period = {
                'D': 24, 'W': 168, 'M': 24*30, 'Y': 24*365
            }.get(period, 24)
            aggregated['avg_power_kW'] = aggregated['total_kWh'] / hours_per_period
            aggregated['avg_cost_per_period'] = aggregated['total_cost']
        return aggregated
    def check_thresholds(self):
        if self.cost_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        violations = {}
        hourly_usage = self.cost_df.resample('H')['power_kW'].sum()
        hourly_violations = hourly_usage[hourly_usage > self.thresholds['hourly_kW']]
        violations['hourly'] = pd.DataFrame({
            'timestamp': hourly_violations.index,
            'power_kW': hourly_violations.values,
            'threshold_kW': self.thresholds['hourly_kW'],
            'excess_kW': hourly_violations.values - self.thresholds['hourly_kW']
        })
        daily_usage = self.cost_df.resample('D')['power_kW'].sum()
        daily_violations = daily_usage[daily_usage > self.thresholds['daily_kWh']]
        violations['daily'] = pd.DataFrame({
            'date': daily_violations.index.date,
            'total_kWh': daily_violations.values,
            'threshold_kWh': self.thresholds['daily_kWh'],
            'excess_kWh': daily_violations.values - self.thresholds['daily_kWh']
        })
        weekly_usage = self.cost_df.resample('W')['power_kW'].sum()
        weekly_violations = weekly_usage[weekly_usage > self.thresholds['weekly_kWh']]
        violations['weekly'] = pd.DataFrame({
            'week_start': weekly_violations.index,
            'total_kWh': weekly_violations.values,
            'threshold_kWh': self.thresholds['weekly_kWh'],
            'excess_kWh': weekly_violations.values - self.thresholds['weekly_kWh']
        })
        monthly_usage = self.cost_df.resample('M')['power_kW'].sum()
        monthly_violations = monthly_usage[monthly_usage > self.thresholds['monthly_kWh']]
        violations['monthly'] = pd.DataFrame({
            'month': monthly_violations.index.strftime('%Y-%m'),
            'total_kWh': monthly_violations.values,
            'threshold_kWh': self.thresholds['monthly_kWh'],
            'excess_kWh': monthly_violations.values - self.thresholds['monthly_kWh']
        })
        return violations
    def generate_cost_report(self, df=None):
        if df is not None:
            self.load_data(df)
        if self.cost_df is None:
            raise ValueError("No data loaded. Provide DataFrame or call load_data() first.")
        aggregated = {
            'hourly': self.calculate_costs_by_period('H'),
            'daily': self.calculate_costs_by_period('D'),
            'weekly': self.calculate_costs_by_period('W'),
            'monthly': self.calculate_costs_by_period('M'),
            'yearly': self.calculate_costs_by_period('Y')
        }
        violations = self.check_thresholds()
        self._print_summary_report(aggregated, violations)
        return aggregated, violations
    def predict_future_costs(self, forecaster=None, days_ahead=30, current_prediction=None):
        import pandas as pd
        import numpy as np
        future_dates = pd.date_range(
            start=pd.Timestamp.now(),
            periods=days_ahead * 24,
            freq='H'
        )
        np.random.seed(42)
        predictions = []
        for i, future_time in enumerate(future_dates):
            hour = future_time.hour
            dayofweek = future_time.dayofweek
            base_consumption = current_prediction if current_prediction is not None else 1.5
            if 6 <= hour <= 9:
                hour_factor = 1.2
            elif 10 <= hour <= 16:
                hour_factor = 0.8
            elif 17 <= hour <= 21:  
                hour_factor = 2.0
            else:  
                hour_factor = 0.6
            if dayofweek >= 5:  
                weekend_factor = 1.3
            else:
                weekend_factor = 1.0
            month = future_time.month
            if month in [12, 1, 2]:  
                seasonal_factor = 1.2
            elif month in [6, 7, 8]:  
                seasonal_factor = 1.1
            else:
                seasonal_factor = 1.0
            prediction = base_consumption * hour_factor * weekend_factor * seasonal_factor
            prediction += np.random.normal(0, 0.3)  
            prediction = max(0.1, prediction)  
            predictions.append(prediction)
        future_df = pd.DataFrame({
            'Global_active_power': predictions
        }, index=future_dates)
        self.load_data(future_df)
        daily_costs = self.calculate_costs_by_period('D')
        monthly_costs = self.calculate_costs_by_period('M')
        yearly_costs = self.calculate_costs_by_period('Y')
        peak_df = self.cost_df[self.cost_df['is_peak']]
        off_peak_df = self.cost_df[~self.cost_df['is_peak']]
        peak_daily = peak_df.groupby(peak_df.index.date)['power_kW'].sum() if not peak_df.empty else pd.Series(dtype=float)
        off_peak_daily = off_peak_df.groupby(off_peak_df.index.date)['power_kW'].sum() if not off_peak_df.empty else pd.Series(dtype=float)
        daily_avg_cost = daily_costs['total_cost'].mean()
        daily_avg_kwh = daily_costs['total_kWh'].mean()
        projections = {
            'daily_avg_cost': daily_avg_cost,
            'monthly_total_cost': daily_avg_cost * 30,  
            'yearly_total_cost': daily_avg_cost * 365,   
            'monthly_avg_kwh': daily_avg_kwh * 30,       
            'yearly_total_kwh': daily_avg_kwh * 365,     
            'peak_hours_daily': peak_daily.mean() if not peak_daily.empty else 0,
            'off_peak_hours_daily': off_peak_daily.mean() if not off_peak_daily.empty else 0,
        }
        return projections, future_df
    def _print_summary_report(self, aggregated, violations):
        print("\n" + "="*60)
        print("  POWER CONSUMPTION COST ANALYSIS REPORT")
        print("="*60)
        total_cost = self.cost_df['hourly_cost'].sum()
        total_kwh = self.cost_df['power_kW'].sum()
        avg_cost_per_day = aggregated['daily']['total_cost'].mean()
        print("\n📊 OVERALL STATISTICS:")
        print(f"  Total Consumption: {total_kwh:.1f} kWh")
        print(f"  Total Cost: ${total_cost:.2f}")
        print(f"  Average Daily Cost: ${avg_cost_per_day:.2f}")
        print(f"  Peak Hours: {self.peak_hours[0]:02d}:00 - {self.peak_hours[1]:02d}:00 "
              f"({self.peak_multiplier:.1f}x rate)")
        peak_cost = self.cost_df[self.cost_df['is_peak']]['hourly_cost'].sum()
        off_peak_cost = self.cost_df[~self.cost_df['is_peak']]['hourly_cost'].sum()
        peak_pct = (peak_cost / total_cost) * 100 if total_cost > 0 else 0
        print("\n⚡ PEAK VS OFF-PEAK BREAKDOWN:")
        print(f"  Peak Cost: ${peak_cost:.2f} ({peak_pct:.1f}%)")
        print(f"  Off-Peak Cost: ${off_peak_cost:.2f} ({100-peak_pct:.1f}%)")
        total_violations = sum(len(v) for v in violations.values())
        print("\n🚨 THRESHOLD VIOLATIONS:")
        print(f"  Total Violations: {total_violations}")
        for period, v_df in violations.items():
            if not v_df.empty:
                print(f"  {period.capitalize()}: {len(v_df)} violations")
        print("\n" + "="*60)
    def create_visualizations(self, save_path='./results'):
        if self.cost_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        import os
        os.makedirs(save_path, exist_ok=True)
        plt.style.use('default')
        sns.set_palette("husl")
        self._plot_daily_cost_trends(save_path)
        self._plot_monthly_cost_comparison(save_path)
        self._plot_peak_offpeak_pie(save_path)
        self._plot_hourly_cost_pattern(save_path)
        print(f"\n✓ Visualizations saved to {save_path}/")
    def _plot_daily_cost_trends(self, save_path):
        monthly_costs = self.cost_df.resample('M')['hourly_cost'].sum()
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(monthly_costs)), monthly_costs.values,
                      color='skyblue', edgecolor='black', alpha=0.7)
        plt.title('Monthly Electricity Cost Comparison', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Total Cost ($)', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        month_labels = [d.strftime('%Y-%m') for d in monthly_costs.index]
        plt.xticks(range(len(month_labels)), month_labels, rotation=45)
        for bar, cost in zip(bars, monthly_costs.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(monthly_costs.values)*0.01,
                    f'${cost:.0f}', ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'monthly_cost_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    def _plot_peak_offpeak_pie(self, save_path):
        hourly_avg = self.cost_df.groupby('hour')['hourly_cost'].mean()
        plt.figure(figsize=(12, 8))
        bars = plt.bar(hourly_avg.index, hourly_avg.values,
                      color='lightgreen', edgecolor='black', alpha=0.7)
        plt.title('Average Electricity Cost by Hour of Day', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Hour of Day', fontsize=12)
        plt.ylabel('Average Cost ($)', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        plt.xticks(range(0, 24, 2))
        peak_start, peak_end = self.peak_hours
        for i in range(peak_start, peak_end):
            if i < len(bars):
                bars[i].set_color('orange')
                bars[i].set_edgecolor('darkred')
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightgreen', edgecolor='black', label='Off-Peak Hours'),
            Patch(facecolor='orange', edgecolor='darkred', label='Peak Hours')
        ]
        plt.legend(handles=legend_elements)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'hourly_cost_pattern.png'), dpi=300, bbox_inches='tight')
        plt.close()
def example_usage():
    print("🔌 Power Cost Analyzer Example")
    print("=" * 40)
    dates = pd.date_range('2024-01-01', periods=168, freq='H')  
    np.random.seed(42)
    base_power = 2.0
    hourly_variation = np.sin(np.arange(168) * 2 * np.pi / 24) * 1.5
    noise = np.random.normal(0, 0.5, 168)
    power_data = base_power + hourly_variation + noise
    power_data = np.maximum(power_data, 0.1)  
    df = pd.DataFrame({
        'Global_active_power': power_data
    }, index=dates)
    analyzer = PowerCostAnalyzer(
        price_per_kwh=0.25,  
        peak_hours=(17, 21),  
        peak_multiplier=2.0   
    )
    analyzer.set_thresholds(
        hourly_kW=6.0,
        daily_kWh=40.0,
        weekly_kWh=250.0,
        monthly_kWh=1000.0
    )
    aggregated, violations = analyzer.generate_cost_report(df)
    analyzer.create_visualizations('./results')
    print("\n✅ Analysis complete!")
    print("Check the ./results/ directory for visualizations.")
    return analyzer, aggregated, violations
if __name__ == "__main__":
    analyzer, aggregated, violations = example_usage()