from flask import Flask, render_template
from flask_cors import CORS
from model import PowerConsumptionForecaster
from cost_analyzer import PowerCostAnalyzer
import os
from routes import health_bp, prediction_bp, cost_bp, visualization_bp
from services import ModelService, CostService
app = Flask(__name__)
CORS(app)
os.makedirs('./models', exist_ok=True)
os.makedirs('./results', exist_ok=True)
forecaster = PowerConsumptionForecaster()
cost_analyzer = PowerCostAnalyzer(
    price_per_kwh=0.20,
    peak_hours=(18, 22),
    peak_multiplier=1.5
)
model_service = ModelService(forecaster)
cost_service = CostService(cost_analyzer)
app.model_service = model_service
app.cost_service = cost_service
app.register_blueprint(health_bp)
app.register_blueprint(prediction_bp)
app.register_blueprint(cost_bp)
app.register_blueprint(visualization_bp)
@app.route('/')
def index():
    return render_template('index.html')
@app.errorhandler(404)
def not_found(error):
    return {
        'success': False,
        'error': 'Endpoint not found'
    }, 404
@app.errorhandler(500)
def internal_error(error):
    return {
        'success': False,
        'error': 'Internal server error'
    }, 500
model_path = './ai_models/power_forecaster.pkl'
if os.path.exists(model_path):
    try:
        forecaster.load_model(model_path)
        print("✓ Model loaded from", model_path)
        print(f"  - Model type: {type(forecaster.model)}")
        print(f"  - Feature columns: {len(forecaster.feature_cols) if forecaster.feature_cols else 'None'}")
        print(f"  - Has data: {forecaster.df is not None}")
    except Exception as e:
        print("⚠ Could not load model:", str(e))
else:
    print("⚠ No saved model found at", model_path)
if __name__ == '__main__':
    print("=" * 60)
    print("  ⚡ POWER FORECASTING API SERVER")
    print("=" * 60)
    print()
    print("✓ Server starting...")
    print("✓ API available at: http://localhost:8002")
    print("✓ API documentation: http://localhost:8002")
    print()
    print("📁 Directories:")
    print("  - Results: ./results")
    print("  - Models: ./models")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    print()
    app.run(
        debug=True,
        host='0.0.0.0',
        port=8002,
        threaded=True
    )