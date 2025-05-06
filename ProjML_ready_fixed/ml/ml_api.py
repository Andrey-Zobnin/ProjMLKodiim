
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_text = data.get('input', '')
    return jsonify({'prediction': f'Predicted: {input_text}'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
