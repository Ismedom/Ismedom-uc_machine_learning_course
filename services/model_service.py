from model import PowerConsumptionForecaster
import pandas as pd
import numpy as np
import traceback
class ModelService:
    def __init__(self, forecaster=None):
        self.forecaster = forecaster or PowerConsumptionForecaster()
        try:
            self.forecaster.load_model('./ai_models/power_forecaster.pkl')
            print("✓ Model loaded successfully in ModelService")
        except Exception as e:
            print(f"⚠️  Could not load saved model: {e}")
            print("Model will need to be trained first")
        self.typical_ranges = {
            'lag_1h': (0.1, 8.0),
            'lag_24h': (0.1, 8.0),
            'rolling_mean_24h': (0.1, 5.0),
            'hour': (0, 23),
            'dayofweek': (0, 6)
        }
    def get_health_info(self):
        health_info = {
            'success': True,
            'status': 'healthy',
            'model_trained': self.forecaster.model is not None,
            'data_loaded': self.forecaster.df is not None,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        if self.forecaster.model is not None:
            health_info['model_info'] = {
                'type': str(type(self.forecaster.model)),
                'feature_cols_count': len(self.forecaster.feature_cols) if self.forecaster.feature_cols else 0,
                'has_data': self.forecaster.df is not None,
                'data_shape': self.forecaster.df.shape if self.forecaster.df is not None else None,
                'stats': self.forecaster.stats if hasattr(self.forecaster, 'stats') else None
            }
        return health_info
    def get_stats(self):
        return self.forecaster.get_stats()
    def validate_input(self, features_data):
        warnings = []
        required_features = ['hour', 'dayofweek', 'lag_1h', 'lag_24h', 'rolling_mean_24h']
        missing = [f for f in required_features if f not in features_data]
        if missing:
            warnings.append({
                'type': 'error',
                'message': f'Missing required features: {", ".join(missing)}'
            })
            return False, warnings
        for feature, value in features_data.items():
            if feature in self.typical_ranges:
                min_val, max_val = self.typical_ranges[feature]
                if value < min_val or value > max_val:
                    warnings.append({
                        'type': 'warning',
                        'feature': feature,
                        'value': value,
                        'expected_range': f'{min_val}-{max_val}',
                        'message': f'{feature} = {value:.2f} is outside typical range ({min_val}-{max_val})'
                    })
        if 'lag_1h' in features_data and 'lag_24h' in features_data:
            diff = abs(features_data['lag_1h'] - features_data['lag_24h'])
            if diff > 5.0:
                warnings.append({
                    'type': 'info',
                    'message': f'Large difference between lag_1h ({features_data["lag_1h"]:.2f}) and lag_24h ({features_data["lag_24h"]:.2f})'
                })
        if all(f in features_data for f in ['lag_1h', 'lag_24h', 'rolling_mean_24h']):
            avg_lag = (features_data['lag_1h'] + features_data['lag_24h']) / 2
            rolling = features_data['rolling_mean_24h']
            if abs(rolling - avg_lag) > 2.0:
                warnings.append({
                    'type': 'info',
                    'message': f'Rolling mean ({rolling:.2f}) differs significantly from recent lags (avg: {avg_lag:.2f})'
                })
        return True, warnings
    def predict_single(self, features_data):
        if self.forecaster.model is None:
            raise ValueError('Model not trained. Please train the model first.')
        is_valid, warnings = self.validate_input(features_data)
        if not is_valid:
            error_messages = [w['message'] for w in warnings if w['type'] == 'error']
            raise ValueError('; '.join(error_messages))
        default_power = 1.0
        if hasattr(self.forecaster, 'stats') and 'avg_power' in self.forecaster.stats:
            default_power = self.forecaster.stats['avg_power']
        features = {}
        now = pd.Timestamp.now()
        for col in self.forecaster.feature_cols:
            if col in features_data:
                features[col] = float(features_data[col])
            else:
                if col == 'lag_2h':
                    features[col] = features_data.get('lag_1h',
                                    features_data.get('lag_3h', default_power))
                elif col == 'lag_3h':
                    features[col] = features_data.get('lag_1h', default_power)
                elif col == 'lag_168h':
                    features[col] = features_data.get('lag_24h', default_power)
                elif col == 'rolling_mean_3h':
                    lag1 = features_data.get('lag_1h', default_power)
                    lag2 = features_data.get('lag_2h', lag1)
                    lag3 = features_data.get('lag_3h', lag1)
                    features[col] = (lag1 + lag2 + lag3) / 3.0
                elif col == 'rolling_std_24h':
                    mean_24h = features_data.get('rolling_mean_24h', default_power)
                    features[col] = mean_24h * 0.2
                elif col == 'Global_intensity':
                    lag1 = features_data.get('lag_1h', default_power)
                    features[col] = lag1 * 1000 / 230
                elif col == 'Voltage':
                    features[col] = 240.0
                elif col == 'Global_reactive_power':
                    active = features_data.get('lag_1h', default_power)
                    features[col] = active * 0.25
                elif col == 'Sub_metering_1':
                    total = features_data.get('lag_1h', default_power)
                    features[col] = total * 0.3
                elif col == 'Sub_metering_2':
                    total = features_data.get('lag_1h', default_power)
                    features[col] = total * 0.2
                elif col == 'Sub_metering_3':
                    total = features_data.get('lag_1h', default_power)
                    features[col] = total * 0.4
                elif col == 'quarter':
                    features[col] = features_data.get('month', now.month) // 3 + 1
                elif col == 'month':
                    features[col] = features_data.get('month', now.month)
                elif col == 'year':
                    features[col] = features_data.get('year', now.year)
                elif col == 'dayofyear':
                    features[col] = features_data.get('dayofyear', now.dayofyear)
                elif col == 'weekofyear':
                    features[col] = features_data.get('weekofyear', now.isocalendar()[1])
                elif col == 'hour':
                    features[col] = features_data.get('hour', 12)
                elif col == 'dayofweek':
                    features[col] = features_data.get('dayofweek', 0)
                else:
                    features[col] = 0
        X_pred = pd.DataFrame([features])
        X_pred = X_pred[self.forecaster.feature_cols]
        prediction = self.forecaster.predict(X_pred)[0]
        prediction = float(prediction)
        if prediction < 0.05:
            warnings.append({
                'type': 'warning',
                'message': f'Prediction is very low ({prediction:.3f} kW). Please verify input values.'
            })
        elif prediction > 10.0:
            warnings.append({
                'type': 'warning',
                'message': f'Prediction is very high ({prediction:.3f} kW). This is unusual for household consumption.'
            })
        confidence = self._calculate_confidence(features_data, prediction, warnings)
        context = self._get_prediction_context(features_data, prediction)
        response = {
            'prediction': prediction,
            'prediction_kwh': prediction * 1.0,
            'confidence': confidence,
            'features_used': {
                'dayofweek': features_data.get('dayofweek'),
                'hour': features_data.get('hour'),
                'lag_1h': features_data.get('lag_1h'),
                'lag_24h': features_data.get('lag_24h'),
                'rolling_mean_24h': features_data.get('rolling_mean_24h')
            },
            'all_features': features,
            'warnings': warnings,
            'context': context,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        return response
    def _calculate_confidence(self, input_features, prediction, warnings):
        score = 100
        for warning in warnings:
            if warning['type'] == 'warning':
                score -= 15
            elif warning['type'] == 'error':
                score -= 40
        if prediction < 0.1 or prediction > 8.0:
            score -= 20
        for feature, (min_val, max_val) in self.typical_ranges.items():
            if feature in input_features:
                value = input_features[feature]
                if value < min_val or value > max_val:
                    score -= 10
        if score >= 80:
            return 'high'
        elif score >= 50:
            return 'medium'
        else:
            return 'low'
    def _get_prediction_context(self, input_features, prediction):
        context = {}
        if hasattr(self.forecaster, 'stats') and 'avg_power' in self.forecaster.stats:
            avg_power = self.forecaster.stats['avg_power']
            diff_from_avg = prediction - avg_power
            pct_diff = (diff_from_avg / avg_power * 100) if avg_power > 0 else 0
            context['average_power'] = f"{avg_power:.2f} kW"
            context['difference_from_average'] = f"{diff_from_avg:+.2f} kW ({pct_diff:+.1f}%)"
            if abs(pct_diff) < 10:
                context['usage_level'] = 'Normal'
            elif pct_diff > 50:
                context['usage_level'] = 'Very High'
            elif pct_diff > 20:
                context['usage_level'] = 'High'
            elif pct_diff < -50:
                context['usage_level'] = 'Very Low'
            elif pct_diff < -20:
                context['usage_level'] = 'Low'
            else:
                context['usage_level'] = 'Normal'
        hour = input_features.get('hour', 12)
        if 6 <= hour < 9:
            context['time_period'] = 'Morning'
        elif 9 <= hour < 12:
            context['time_period'] = 'Late Morning'
        elif 12 <= hour < 14:
            context['time_period'] = 'Lunch Time'
        elif 14 <= hour < 18:
            context['time_period'] = 'Afternoon'
        elif 18 <= hour < 22:
            context['time_period'] = 'Evening (Peak Hours)'
        elif 22 <= hour < 24:
            context['time_period'] = 'Night'
        else:
            context['time_period'] = 'Late Night'
        dayofweek = input_features.get('dayofweek', 0)
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        context['day_name'] = days[dayofweek] if 0 <= dayofweek < 7 else 'Unknown'
        context['is_weekend'] = 'Yes' if dayofweek >= 5 else 'No'
        lag_1h = input_features.get('lag_1h', 0)
        lag_24h = input_features.get('lag_24h', 0)
        if lag_1h > 0 and prediction > 0:
            hourly_change = ((prediction - lag_1h) / lag_1h * 100) if lag_1h > 0 else 0
            if abs(hourly_change) < 10:
                context['hourly_trend'] = 'Stable'
            elif hourly_change > 0:
                context['hourly_trend'] = f'Increasing ({hourly_change:+.1f}%)'
            else:
                context['hourly_trend'] = f'Decreasing ({hourly_change:+.1f}%)'
        if lag_24h > 0 and prediction > 0:
            daily_change = ((prediction - lag_24h) / lag_24h * 100) if lag_24h > 0 else 0
            if abs(daily_change) < 10:
                context['daily_trend'] = 'Similar to yesterday'
            elif daily_change > 0:
                context['daily_trend'] = f'Higher than yesterday ({daily_change:+.1f}%)'
            else:
                context['daily_trend'] = f'Lower than yesterday ({daily_change:+.1f}%)'
        return context
    def get_model_info(self):
        if self.forecaster.model is None:
            raise ValueError('No model trained')
        info = {
            'model_type': 'LightGBM Gradient Boosting',
            'features_count': len(self.forecaster.feature_cols),
            'features': self.forecaster.feature_cols,
            'best_iteration': self.forecaster.model.best_iteration,
            'trained': True,
            'typical_ranges': self.typical_ranges
        }
        if hasattr(self.forecaster, 'stats'):
            info['training_stats'] = self.forecaster.stats
        return info
    def get_predictions_csv(self):
        if self.forecaster.model is None or self.forecaster.test_df is None:
            raise ValueError('No predictions available')
        X_test = self.forecaster.test_df[self.forecaster.feature_cols]
        y_test = self.forecaster.test_df[self.forecaster.target_col]
        predictions = self.forecaster.predict(X_test)
        results_df = pd.DataFrame({
            'datetime': self.forecaster.test_df.index,
            'actual': y_test.values,
            'predicted': predictions,
            'error': y_test.values - predictions,
            'error_percent': ((y_test.values - predictions) / y_test.values * 100)
        })
        return results_df
    def get_feature_importance(self, top_n=10):
        if self.forecaster.model is None:
            raise ValueError('No model trained')
        import pandas as pd
        importance = self.forecaster.model.feature_importance(importance_type='gain')
        feature_importance = pd.DataFrame({
            'feature': self.forecaster.feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)
        top_features = feature_importance.head(top_n)
        return {
            'top_features': top_features.to_dict('records'),
            'total_features': len(self.forecaster.feature_cols),
            'most_important': top_features.iloc[0]['feature'] if len(top_features) > 0 else None
        }