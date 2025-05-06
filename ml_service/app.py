
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process():
    data = request.get_json()
    content = data.get('content', '')
    return jsonify({"content": f"Processed: {content}"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
