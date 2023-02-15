import os
import json
from flask import Flask
from azure_table_op import AzureTableOp

build_date = "unknown"
if os.path.isfile("build_date.txt"):
    with open("build_date.txt", "r") as f:
        build_date = f.read()
print(f"Build: {build_date}")

azure_table_op = AzureTableOp()
app = Flask(__name__)

@app.route('/wx_login_callback')
def handle_wx_login():
    return json.dumps(
        {
            'errcode': 0,
            'message': 'success',
        }
    )

if __name__ == '__main__':
    app.run(threaded=True, port=5000)