from flask import Blueprint, request, jsonify
import traceback
from services import CostService, ModelService
cost_bp = Blueprint('cost', __name__)
def get_cost_service():
    from flask import current_app
    if not hasattr(current_app, 'cost_service'):
        current_app.cost_service = CostService()
    return current_app.cost_service
def get_model_service():
    from flask import current_app
    if not hasattr(current_app, 'model_service'):
        current_app.model_service = ModelService()
    return current_app.model_service
@cost_bp.route('/api/cost/predict', methods=['POST'])
def predict_future_costs():
    try:
        model_service = get_model_service()
        cost_service = get_cost_service()
        data = request.json if request.json else {}
        print(f"Received cost prediction request: {data}")
        if model_service.forecaster.model is None:
            print("ERROR: Model not loaded")
            return jsonify({
                'success': False,
                'error': 'Model not trained. Please train the model first.'
            }), 400
        print("Model is available, proceeding with prediction...")
        result = cost_service.predict_future_costs(model_service.forecaster, data)
        print("Response prepared successfully")
        days_ahead = data.get('days_ahead', 30)
        return jsonify({
            'success': True,
            'message': f'Cost prediction completed for {days_ahead} days',
            'data': result
        })
    except Exception as e:
        print(f"ERROR in cost prediction: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500
@cost_bp.route('/api/cost/settings', methods=['GET', 'POST'])
def cost_settings():
    try:
        cost_service = get_cost_service()
        if request.method == 'GET':
            settings = cost_service.get_settings()
            return jsonify({
                'success': True,
                'data': settings
            })
        elif request.method == 'POST':
            data = request.json if request.json else {}
            cost_service.update_settings(data)
            return jsonify({
                'success': True,
                'message': 'Cost settings updated successfully'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
@cost_bp.route('/api/cost/visualizations', methods=['POST'])
def generate_cost_visualizations():
    try:
        cost_service = get_cost_service()
        model_service = get_model_service()
        if cost_service.cost_analyzer.cost_df is None:
            if model_service.forecaster.model is None:
                return jsonify({
                    'success': False,
                    'error': 'No model available. Please train the model first.'
                }), 400
            default_data = {
                'days_ahead': 7,
                'price_per_kwh': 0.20,
                'peak_hours': [18, 22],
                'peak_multiplier': 1.5
            }
            print("No cost data available, running default cost prediction...")
            cost_service.predict_future_costs(model_service.forecaster, default_data)
        cost_files = cost_service.generate_visualizations()
        return jsonify({
            'success': True,
            'message': 'Cost visualizations generated successfully',
            'data': {
                'files': cost_files,
                'count': len(cost_files)
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500