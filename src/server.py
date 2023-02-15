import os
import json
import traceback
from flask import Flask, request, jsonify
from azure_table_op import AzureTableOp

DEBUG = True

build_date = "unknown"
if os.path.isfile("build_date.txt"):
    with open("build_date.txt", "r") as f:
        build_date = f.read()
print(f"Build: {build_date}")

azure_table_op = AzureTableOp()
app = Flask(__name__)

@app.route('/wx_login_callback', methods=['POST'])
def handle_wx_login():
    res = {'errcode': 0, 'message': "成功"}
    if request.method == "POST":
        try:
            form_data = request.form.to_dict()

            # Check that all the necessary keys are present and not empty

            for key in ['userId', 'tempUserId', 'ipAddr']:
                if key not in form_data:
                    res['errcode'] = 2
                    res['message'] = f"Invalid request: key {key} is missing"
                    return jsonify(res)
                if len(form_data[key]) == 0:
                    res['errcode'] = 2
                    res['message'] = f"Invalid request: {key} has empty value"
                    return jsonify(res)
                
            user_id = form_data['userId']
            temp_user_id = form_data['tempUserId']
            nickname = form_data['nickname']
            avatar_url = form_data['avatar']
            ip_address = form_data['ipAddr']

            table_name = "users"
            if DEBUG:
                table_name = table_name + "Test"

            # Check if user_id already exists, if so, load the entity

            query_filter = f"PartitionKey eq @channel and RowKey eq @user_id"
            select = None
            parameters = {'channel': "wx_user", 'user_id': user_id}

            query_res = azure_table_op.query_entities(query_filter, select, parameters, table_name)

            if query_res['status'] != 0:
                res['errcode'] = query_res['status']
                res['message'] = f"Failed to query entities: {query_res['message']}"
                return jsonify(res)

            # If user is not found, create a new user
            if len(query_res['data']) <= 0:
                entity = {
                    'PartitionKey': "wx_user",
                    'RowKey': user_id,
                    'nickname': user_id if len(nickname) <= 0 else nickname,
                    'avatar_url': avatar_url,
                    'ip_address': ip_address,
                    'n_tokens': 0,
                }
                azure_table_op.update_entities(entity, table_name)

        except:
            res['errcode'] = 2
            res['message'] = f"Invalid request: {traceback.format_exc()}"
    return jsonify(res)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=DEBUG)