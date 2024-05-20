from flask import Flask, request, jsonify
from flask_swagger_ui import get_swaggerui_blueprint
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

### Swagger specific ###
SWAGGER_URL = '/swagger'  # URL for exposing Swagger UI (relative to your app's root)
API_URL = '/static/swagger.json'  # Our API url (could be a local resource)

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,  # Swagger UI static files will be mapped to '{SWAGGER_URL}/dist/'
    API_URL,
    config={  # Swagger UI config overrides
        'app_name': "Test application"
    }
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)
### End Swagger specific ###

@app.route('/submit', methods=['POST'])
def submit():
    data = request.get_json()  # Parse the incoming JSON data
    Log = data.get('Log')
    Count = data.get('Count')

    if not Log or not Count:
        return jsonify({'error': 'Please provide both Log and Count'}), 400

    response = {
        'message': f'Received Log: {Log} and Count: {Count}'
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8003)
