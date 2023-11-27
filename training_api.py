from flask import Flask, request, jsonify
from flask_cors import CORS  # CORS 미들웨어 추가
import threading
from training_main import train_main

app = Flask(__name__)
CORS(app)  # CORS 미들웨어를 Flask 애플리케이션에 등록

# 변수 초기화
model = None
training_thread = None
is_training = False

@app.route('/start_training', methods=['POST'])
def start_training():
    global training_thread, is_training

    if not is_training:
        try:
            request_data = request.get_json()['input']
            dataset_name = request_data['dataset_name']
            model_name = request_data['model_name']
            training_purpose = request_data['training_purpose']
        except KeyError:
            return jsonify({'error': 'Training purpose or model name missing in request'}), 400

        is_training = True
        training_thread = threading.Thread(target=train_main, args=(training_purpose,dataset_name, model_name))
        training_thread.start()
        return jsonify({'message': 'Training started'}), 200
    else:
        return jsonify({'message': 'Training is already in progress'}), 400

@app.route('/stop_training', methods=['POST'])
def stop_training():
    global training_thread, is_training

    if is_training and training_thread.is_alive():
        training_thread.join()  # 학습 스레드 종료 대기
        is_training = False
        return jsonify({'message': 'Training stopped'}), 200
    else:
        return jsonify({'message': 'No active training process to stop'}), 400

if __name__ == '__main__':
    app.run(port=8083)
