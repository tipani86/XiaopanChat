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

azure_table_op = AzureTableOp()
app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({
        'build': build_date
    })

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
                
            user_id = str(form_data['userId'])
            temp_user_id = str(form_data['tempUserId'])
            nickname = str(form_data['nickname'])
            avatar_url = str(form_data['avatar'])
            ip_address = str(form_data['ipAddr'])

            # Step 1: First check if the temp_user_id exists in the tempUserIds table

            table_name = "tempUserIds"
            if DEBUG:
                table_name = table_name + "Test"

            query_filter = f"PartitionKey eq @channel and RowKey eq @temp_user_id"
            select = None
            parameters = {'channel': "wx_user", 'temp_user_id': temp_user_id}

            table_res = azure_table_op.query_entities(query_filter, select, parameters, table_name)

            if table_res['status'] != 0:
                res['errcode'] = table_res['status']
                res['message'] = f"Failed to query entities: {table_res['message']}"
                return jsonify(res)
            
            # If temp_user_id is not found, return error because the POST request is not expected

            if len(table_res['data']) <= 0:
                res['errcode'] = 2
                res['message'] = f"Invalid request: temp_user_id {temp_user_id} not found"
                return jsonify(res)
            
            # Step 2: If the temp_user_id is found, add the user_id to the tempUserId entry

            entity = table_res['data'][0]
            entity['user_id'] = user_id
            entity['data'] = json.dumps({
                'nickname': nickname,
                'avatar_url': avatar_url,
                'ip_address': ip_address
            })

            table_res = azure_table_op.update_entities(entity, table_name)

            if table_res['status'] != 0:
                res['errcode'] = table_res['status']
                res['message'] = f"Failed to update entities: {table_res['message']}"
                return jsonify(res)

        except:
            res['errcode'] = 2
            res['message'] = f"Invalid request: {traceback.format_exc()}"
    return jsonify(res)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=DEBUG)