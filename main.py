from flask import Flask, request, jsonify
import json
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

app = Flask(__name__)
# 定义神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(2, 1)  # 输入层到输出层的全连接层，因为有两个输入特征

    def forward(self, x):
        x = self.fc(x)
        return x
    # 初始化模型和优化器

@app.route("/check", methods=["POST"])
def check():
    myjson = request.get_json()
    data = json.loads(myjson["msg"])

    # test1 = json.loads(myjson["msg"])
    # print(type(myjson["msg"]))
    # print(type(test1))
    # print(test1["Longitude"])

    Longitude = data["Longitude"]
    Latitude = data["Latitude"]

    # print(Longitude)

    model = Net()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    x_coordinate = torch.tensor([Longitude])  # x坐标
    y_coordinate = torch.tensor([Latitude])  # y坐标
    input = torch.stack([x_coordinate, y_coordinate], dim=1)
    with torch.no_grad():
        Amount = int(model(input))

    # Amount= 1

    result_dict = {"Amount":Amount}
    return jsonify(result_dict)


@app.route("/test", methods=["POST"])
def ttest():
    _head = {
        "applicationCode": None,
        "operationTime": " ",
        "status": "S",
        "code": "-000000",
        "msg": "成功"
    }
    myjson = request.get_json()

    # test1 = json.loads(myjson["msg"])
    # print(type(myjson["msg"]))
    # print(type(test1))
    # print(test1["Longitude"])
    Longitude = myjson["Longitude"]
    Latitude = myjson["Latitude"]
    model = Net()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    x_coordinate = torch.tensor([Longitude])  # x坐标
    y_coordinate = torch.tensor([Latitude])  # y坐标
    input = torch.stack([x_coordinate, y_coordinate], dim=1)
    with torch.no_grad():
        Amount = int(model(input))

    # Amount= 1

    _body = {
        "Amount": Amount
    }
    result_dict = {"head":_head, "body":_body}
    return jsonify(result_dict)

# 启动web服务器
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1000, debug=False)
