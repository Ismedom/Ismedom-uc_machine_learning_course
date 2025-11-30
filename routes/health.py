from flask import Blueprint, jsonify
import traceback
health_bp = Blueprint('health', __name__)
def get_model_service():
    from flask import current_app
    return current_app.model_service
@health_bp.route('/api/health', methods=['GET'])
def health_check():
    try:
        model_service = get_model_service()
        health_info = model_service.get_health_info()
        return jsonify(health_info)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500
@health_bp.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        model_service = get_model_service()
        stats = model_service.get_stats()
        return jsonify({
            'success': True,
            'data': stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
@health_bp.route('/api/model/info', methods=['GET'])
def model_info():
    try:
        model_service = get_model_service()
        info = model_service.get_model_info()
        return jsonify({
            'success': True,
            'data': info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500