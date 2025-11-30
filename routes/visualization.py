from flask import Blueprint, jsonify, send_from_directory
import os
import traceback
visualization_bp = Blueprint('visualization', __name__)
@visualization_bp.route('/api/visualizations', methods=['GET'])
def get_visualizations():
    try:
        results_dir = './results'
        if not os.path.exists(results_dir):
            return jsonify({
                'success': False,
                'error': 'No visualizations generated yet'
            }), 404
        files = [f for f in os.listdir(results_dir) if f.endswith('.png')]
        return jsonify({
            'success': True,
            'data': {
                'files': files,
                'count': len(files)
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
@visualization_bp.route('/api/visualizations/<filename>', methods=['GET'])
def get_visualization_file(filename):
    try:
        return send_from_directory('./results', filename)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'File not found: {filename}'
        }), 404