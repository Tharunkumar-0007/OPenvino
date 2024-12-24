# app.py
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from skin import predict_skin_disease 

# Initialize Flask app and configurations
app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.jfif'}

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        app.logger.error('No file part in request')
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        app.logger.error('No selected file')
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Call the prediction function from skin.py
        result = predict_skin_disease(file)

        # If the result contains a message (like invalid image), return that
        if 'message' in result:
            return jsonify({'error': result['message']}), 400

        app.logger.info(f"Prediction result: {result}")
        return jsonify(result), 200

    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000, use_reloader=False)
