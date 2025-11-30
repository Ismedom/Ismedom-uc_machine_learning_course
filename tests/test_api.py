import requests
import json
try:
    response = requests.post('http://localhost:8002/api/cost/predict',
                           json={
                               'price_per_kwh': 0.25,
                               'peak_hours': [18, 22],
                               'peak_multiplier': 2.0,
                               'thresholds': {
                                   'hourly_kW': 5.0,
                                   'daily_kWh': 50.0,
                                   'weekly_kWh': 300.0,
                                   'monthly_kWh': 1200.0
                               },
                               'days_ahead': 7
                           })
    print('Cost prediction response:')
    if response.status_code == 200:
        data = response.json()
        print(f'Success: {data["success"]}')
        if data['success']:
            proj = data['data']['projections']
            print(f'Days predicted: {proj["days_predicted"]}')
            print(f'Daily avg cost: ${proj["daily_avg_cost"]:.2f}')
            print(f'Monthly total cost: ${proj["monthly_total_cost"]:.2f}')
            print(f'Yearly total cost: ${proj["yearly_total_cost"]:.2f}')
        else:
            print(f'Error: {data.get("error", "Unknown error")}')
    else:
        print(f'HTTP Error: {response.status_code}')
        print(response.text)
except Exception as e:
    print(f'Cost prediction failed: {e}')