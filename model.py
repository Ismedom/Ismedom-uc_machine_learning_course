import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import pickle
import os
import warnings
warnings.filterwarnings('ignore')
class PowerConsumptionForecaster:
    def __init__(self, filepath=None):
        self.filepath = filepath
        self.df = None
        self.model = None
        self.train_df = None
        self.test_df = None
        self.feature_cols = None
        self.target_col = 'Global_active_power'
        self.stats = {}
    def load_data(self, filepath=None):
        if filepath:
            self.filepath = filepath
        print("Step 1: Loading data...")
        print("-" * 60)
        self.df = pd.read_csv(
            self.filepath,
            sep=',',
            parse_dates={'datetime': ['Date', 'Time']},
            infer_datetime_format=True,
            low_memory=False,
            na_values=['?']
        )
        self.df.set_index('datetime', inplace=True)
        for col in self.df.columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        self.df.fillna(method='ffill', inplace=True)
        self.df.fillna(method='bfill', inplace=True)
        self.df = self.df.resample('H').mean()
        self.stats = {
            'total_records': len(self.df),
            'avg_power': float(self.df['Global_active_power'].mean()),
            'max_power': float(self.df['Global_active_power'].max()),
            'min_power': float(self.df['Global_active_power'].min()),
            'std_power': float(self.df['Global_active_power'].std()),
            'date_range': {
                'start': str(self.df.index.min()),
                'end': str(self.df.index.max())
            }
        }
        print(f"✓ Data loaded successfully!")
        print(f"  - Total records: {len(self.df)}")
        print(f"  - Date range: {self.df.index.min()} to {self.df.index.max()}")
        print(f"  - Features: {list(self.df.columns)}")
        print()
        return self.df
    def get_stats(self):
        return self.stats
    def explore_data(self, save_path='./results'):
        print("Step 2: Exploring the data...")
        print("-" * 60)
        os.makedirs(save_path, exist_ok=True)
        print("\nBasic Statistics for Global Active Power:")
        print(f"  - Average: {self.stats['avg_power']:.3f} kW")
        print(f"  - Minimum: {self.stats['min_power']:.3f} kW")
        print(f"  - Maximum: {self.stats['max_power']:.3f} kW")
        print(f"  - Std Dev: {self.stats['std_power']:.3f} kW")
        print()
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        sample_data = self.df['Global_active_power'][:720]
        axes[0, 0].plot(sample_data.index, sample_data.values, linewidth=1)
        axes[0, 0].set_title('Power Consumption Over Time (First 30 Days)', 
                            fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Power (kW)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 1].hist(self.df['Global_active_power'].dropna(), bins=50, 
                       color='skyblue', edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Distribution of Power Consumption', 
                            fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Power (kW)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        df_temp = self.df.copy()
        df_temp['hour'] = df_temp.index.hour
        hourly_avg = df_temp.groupby('hour')['Global_active_power'].mean()
        axes[1, 0].plot(hourly_avg.index, hourly_avg.values, marker='o', 
                       linewidth=2, markersize=6, color='orange')
        axes[1, 0].set_title('Average Power by Hour of Day', 
                            fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Hour of Day')
        axes[1, 0].set_ylabel('Average Power (kW)')
        axes[1, 0].set_xticks(range(0, 24, 2))
        axes[1, 0].grid(True, alpha=0.3)
        df_temp['dayofweek'] = df_temp.index.dayofweek
        daily_avg = df_temp.groupby('dayofweek')['Global_active_power'].mean()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[1, 1].bar(days, daily_avg.values, color='green', 
                      alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Average Power by Day of Week', 
                            fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Day of Week')
        axes[1, 1].set_ylabel('Average Power (kW)')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        save_file = os.path.join(save_path, '1_data_exploration.png')
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"✓ Exploration complete! Saved as '{save_file}'")
        print()
        plt.show()
        return save_file
    def create_features(self, df):
        df = df.copy()
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['dayofyear'] = df.index.dayofyear
        df['weekofyear'] = df.index.isocalendar().week.astype(int)
        df['lag_1h'] = df['Global_active_power'].shift(1)
        df['lag_2h'] = df['Global_active_power'].shift(2)
        df['lag_3h'] = df['Global_active_power'].shift(3)
        df['lag_24h'] = df['Global_active_power'].shift(24)
        df['lag_168h'] = df['Global_active_power'].shift(168)
        df['rolling_mean_3h'] = df['Global_active_power'].rolling(window=3).mean()
        df['rolling_mean_24h'] = df['Global_active_power'].rolling(window=24).mean()
        df['rolling_std_24h'] = df['Global_active_power'].rolling(window=24).std()
        df = df.dropna()
        return df
    def prepare_data(self):
        print("Step 3: Creating features for the model...")
        print("-" * 60)
        df_features = self.create_features(self.df)
        print(f"✓ Features created successfully!")
        print(f"  - Total features: {len(df_features.columns)}")
        print(f"  - Records after feature engineering: {len(df_features)}")
        print()
        train_size = int(len(df_features) * 0.8)
        self.train_df = df_features[:train_size]
        self.test_df = df_features[train_size:]
        print(f"✓ Data split complete!")
        print(f"  - Training set: {len(self.train_df)} records")
        print(f"  - Test set: {len(self.test_df)} records")
        print()
        self.feature_cols = [col for col in df_features.columns 
                            if col != 'Global_active_power']
        print(f"Features being used: {self.feature_cols[:5]}... "
              f"(and {len(self.feature_cols)-5} more)")
        print()
        return self.train_df, self.test_df
    def train_model(self):
        print("Step 4: Training the LightGBM model...")
        print("-" * 60)
        X_train = self.train_df[self.feature_cols]
        y_train = self.train_df[self.target_col]
        X_test = self.test_df[self.feature_cols]
        y_test = self.test_df[self.target_col]
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'verbosity': -1
        }
        print("Training in progress...")
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[test_data],
            callbacks=[lgb.early_stopping(stopping_rounds=10)]
        )
        print("\n✓ Model training complete!")
        print()
        return self.model
    def evaluate_model(self):
        print("Step 5: Evaluating model performance...")
        print("-" * 60)
        X_test = self.test_df[self.feature_cols]
        y_test = self.test_df[self.target_col]
        predictions = self.model.predict(X_test, 
                                        num_iteration=self.model.best_iteration)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape)
        }
        print("Model Performance Metrics:")
        print(f"  - RMSE (Root Mean Squared Error): {rmse:.4f} kW")
        print(f"  - MAE (Mean Absolute Error): {mae:.4f} kW")
        print(f"  - R² Score: {r2:.4f}")
        print(f"  - MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
        print()
        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        print("Top 10 Most Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']:20s}: {row['importance']:.0f}")
        print()
        return predictions, metrics, feature_importance
    def visualize_results(self, predictions, save_path='./results'):
        print("Step 6: Creating visualizations...")
        print("-" * 60)
        os.makedirs(save_path, exist_ok=True)
        y_test = self.test_df[self.target_col].values
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        n_points = min(500, len(predictions))
        axes[0, 0].plot(y_test[:n_points], label='Actual', linewidth=2, alpha=0.7)
        axes[0, 0].plot(predictions[:n_points], label='Predicted', 
                       linewidth=2, alpha=0.7)
        axes[0, 0].set_title('Actual vs Predicted Power Consumption', 
                            fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Time (hours)')
        axes[0, 0].set_ylabel('Power (kW)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 1].scatter(y_test, predictions, alpha=0.5, s=10)
        axes[0, 1].plot([y_test.min(), y_test.max()], 
                       [y_test.min(), y_test.max()], 
                       'r--', linewidth=2, label='Perfect prediction')
        axes[0, 1].set_title('Prediction Accuracy', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Actual Power (kW)')
        axes[0, 1].set_ylabel('Predicted Power (kW)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        errors = y_test - predictions
        axes[1, 0].hist(errors, bins=50, color='coral', 
                       edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1, 0].set_title('Distribution of Prediction Errors', 
                            fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Error (Actual - Predicted) kW')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False).head(10)
        axes[1, 1].barh(range(len(feature_importance)), 
                       feature_importance['importance'].values,
                       color='teal', alpha=0.7, edgecolor='black')
        axes[1, 1].set_yticks(range(len(feature_importance)))
        axes[1, 1].set_yticklabels(feature_importance['feature'].values)
        axes[1, 1].set_title('Top 10 Most Important Features', 
                            fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Importance')
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        axes[1, 1].invert_yaxis()
        plt.tight_layout()
        save_file = os.path.join(save_path, '2_model_results.png')
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"✓ Visualizations saved as '{save_file}'")
        print()
        plt.show()
        return save_file
    def predict(self, features):
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        predictions = self.model.predict(features, 
                                        num_iteration=self.model.best_iteration)
        return predictions
    def save_model(self, filepath='./ai_models/power_forecaster.pkl'):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        model_data = {
            'model': self.model,
            'feature_cols': self.feature_cols,
            'stats': self.stats
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✓ Model saved to {filepath}")
    def load_model(self, filepath='./ai_models/power_forecaster.pkl'):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.feature_cols = model_data['feature_cols']
        self.stats = model_data['stats']
        print(f"✓ Model loaded from {filepath}")
    def run_complete_analysis(self, save_path='./results'):
        print("\n" + "="*60)
        print("  HOUSEHOLD POWER CONSUMPTION FORECASTING WITH LIGHTGBM")
        print("="*60)
        print()
        self.load_data()
        self.explore_data(save_path)
        self.prepare_data()
        self.train_model()
        predictions, metrics, feature_importance = self.evaluate_model()
        self.visualize_results(predictions, save_path)
        model_path = './ai_models/power_forecaster.pkl'
        self.save_model(model_path)
        print("="*60)
        print("  ANALYSIS COMPLETE!")
        print("="*60)
        print()
        print("Generated files:")
        print(f"  1. {save_path}/1_data_exploration.png - Data analysis visualizations")
        print(f"  2. {save_path}/2_model_results.png - Model performance visualizations")
        print(f"  3. {model_path} - Trained model (saved automatically)")
        print()
        print("What the model learned:")
        print("  - The model can predict power consumption with reasonable accuracy")
        print("  - Previous hours' consumption is very important for prediction")
        print("  - Time of day and day of week patterns matter")
        print("  - You can now forecast future power consumption!")
        print()
        return metrics, predictions, feature_importance
if __name__ == "__main__":
    filepath = 'resources/household_power_cleaned.csv'
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