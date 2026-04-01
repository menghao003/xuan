import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from flask import Flask, request, jsonify
from bert_predict_func import predict_func
import warnings

warnings.filterwarnings('ignore')

# 创建flask对象
app = Flask(__name__)

#
# 用户访问url: http://主机ip:8010/predict
@app.route('/predict', methods=['POST'])
def predict():
    # 获取json数据
    data = request.get_json()
    print(type(data))
    print(data)
    # 调用预测函数
    result = predict_func(data)
    # 返回结果
    return jsonify(result)


if __name__ == '__main__':
    # 启动服务
    app.run(host='0.0.0.0', port=8010, debug=False)
