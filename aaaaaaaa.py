from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data', methods=['POST'])
def get_data():
    if request.method == 'POST':
        # 클라이언트에서 전송한 데이터를 가져옵니다.
        data_from_client = request.json  # 예: {'id': 'admin'}
        
        # 가져온 데이터를 콘솔에 출력합니다.
        print('클라이언트에서 전송한 데이터:', data_from_client)
        
        # 여기서 데이터를 처리하거나 데이터베이스에서 필요한 데이터를 조회합니다.
        # 임의로 "Data from server"라는 문자열을 JSON 형식으로 클라이언트에 반환합니다.
        data_to_send = {"message": "Data from server"}
        
        # JSON 형식으로 데이터를 클라이언트에 반환합니다.
        return jsonify(data_to_send)

if __name__ == '__main__':
    app.run(debug=True)
