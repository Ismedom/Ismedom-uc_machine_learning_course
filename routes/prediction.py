from flask import Blueprint, request, jsonify, send_file
import io
import traceback
prediction_bp = Blueprint('prediction', __name__)
def get_model_service():
    from flask import current_app
    return current_app.model_service
@prediction_bp.route('/api/predict_single', methods=['POST'])
def predict_single():
    try:
        model_service = get_model_service()
        data = request.json if request.json else {}
        result = model_service.predict_single(data)
        return jsonify({
            'success': True,
            'message': 'Single prediction generated successfully',
            'data': result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500
@prediction_bp.route('/api/test/model', methods=['GET'])
def test_model():
    try:
        model_service = get_model_service()
        test_data = {
            'hour': 12,
            'dayofweek': 1,
            'lag_1h': 2.5,
            'lag_24h': 2.8,
            'rolling_mean_24h': 2.3
        }
        result = model_service.predict_single(test_data)
        return jsonify({
            'success': True,
            'message': 'Model test successful',
            'data': result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500
@prediction_bp.route('/api/download/predictions', methods=['GET'])
def download_predictions():
    try:
        model_service = get_model_service()
        results_df = model_service.get_predictions_csv()
        output = io.StringIO()
        results_df.to_csv(output, index=False)
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name='predictions.csv'
        )
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500