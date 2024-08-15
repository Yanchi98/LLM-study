from . import predict_bp
from .predict import G_model
from flask import Flask, request, Response, abort
import time

@predict_bp.route('/', methods=['POST'])
def predict():
    print(request.get_json())
    prompts = request.get_json().get('prompts')
    print("用户输入", prompts)

    start = time.time()
    res = G_model.get_completion(prompts)
    end = time.time()
    print('推理耗时：', end - start)
    result = {'code': '200', 'msg': '响应成功', 'data': res}

    return Response(str(result), mimetype='application/json')