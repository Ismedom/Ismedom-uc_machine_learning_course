from cost_analyzer import PowerCostAnalyzer
import os
class CostService:
    def __init__(self, cost_analyzer=None):
        self.cost_analyzer = cost_analyzer or PowerCostAnalyzer()
    def predict_future_costs(self, forecaster, request_data):
        if 'price_per_kwh' in request_data:
            self.cost_analyzer.price_per_kwh = request_data['price_per_kwh']
        if 'peak_hours' in request_data:
            self.cost_analyzer.peak_hours = tuple(request_data['peak_hours'])
        if 'peak_multiplier' in request_data:
            self.cost_analyzer.peak_multiplier = request_data['peak_multiplier']
        days_ahead = request_data.get('days_ahead', 30)
        if 'thresholds' in request_data:
            thresholds = request_data['thresholds']
            self.cost_analyzer.set_thresholds(
                hourly_kW=thresholds.get('hourly_kW'),
                daily_kWh=thresholds.get('daily_kWh'),
                weekly_kWh=thresholds.get('weekly_kWh'),
                monthly_kWh=thresholds.get('monthly_kWh')
            )
        current_prediction = request_data.get('current_prediction')
        projections, future_df = self.cost_analyzer.predict_future_costs(forecaster, days_ahead, current_prediction)
        aggregated, violations = self.cost_analyzer.generate_cost_report(future_df)
        result = {
            'projections': {
                'days_predicted': days_ahead,
                'daily_avg_cost': float(projections['daily_avg_cost']),
                'monthly_total_cost': float(projections['monthly_total_cost']),
                'yearly_total_cost': float(projections['yearly_total_cost']),
                'monthly_avg_kwh': float(projections['monthly_avg_kwh']),
                'yearly_total_kwh': float(projections['yearly_total_kwh']),
                'peak_hours_daily': float(projections['peak_hours_daily']),
                'off_peak_hours_daily': float(projections['off_peak_hours_daily'])
            },
            'aggregated': {},
            'violations': {},
            'summary': {
                'total_predicted_cost': float(self.cost_analyzer.cost_df['hourly_cost'].sum()),
                'total_predicted_kwh': float(self.cost_analyzer.cost_df['power_kW'].sum()),
                'avg_daily_cost': float(aggregated['daily']['total_cost'].mean()),
                'total_violations': sum(len(v) for v in violations.values())
            }
        }
        for period, df_agg in aggregated.items():
            if len(df_agg) <= 100:
                result['aggregated'][period] = df_agg.to_dict('records')
            else:
                sampled = df_agg.iloc[::max(1, len(df_agg)//50)]
                result['aggregated'][period] = sampled.to_dict('records')
        for period, df_viol in violations.items():
            result['violations'][period] = df_viol.to_dict('records')
        return result
    def get_settings(self):
        settings = {
            'price_per_kwh': self.cost_analyzer.price_per_kwh,
            'peak_hours': list(self.cost_analyzer.peak_hours),
            'peak_multiplier': self.cost_analyzer.peak_multiplier,
            'thresholds': self.cost_analyzer.thresholds
        }
        return settings
    def update_settings(self, settings_data):
        if 'price_per_kwh' in settings_data:
            self.cost_analyzer.price_per_kwh = settings_data['price_per_kwh']
        if 'peak_hours' in settings_data:
            self.cost_analyzer.peak_hours = tuple(settings_data['peak_hours'])
        if 'peak_multiplier' in settings_data:
            self.cost_analyzer.peak_multiplier = settings_data['peak_multiplier']
        if 'thresholds' in settings_data:
            thresholds = settings_data['thresholds']
            self.cost_analyzer.set_thresholds(
                hourly_kW=thresholds.get('hourly_kW'),
                daily_kWh=thresholds.get('daily_kWh'),
                weekly_kWh=thresholds.get('weekly_kWh'),
                monthly_kWh=thresholds.get('monthly_kWh')
            )
    def generate_visualizations(self):
        if self.cost_analyzer.cost_df is None:
            raise ValueError('No cost data available. Run cost analysis first.')
        self.cost_analyzer.create_visualizations('./results')
        results_dir = './results'
        cost_files = [f for f in os.listdir(results_dir)
                      if f.startswith(('daily_cost', 'monthly_cost', 'peak_offpeak', 'hourly_cost')) and f.endswith('.png')]
        return cost_files