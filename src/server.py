import os
import json
import traceback
from loguru import logger
from flask import Flask, request, jsonify
from utils import AzureTableOp, get_md5_hash_7pay
from app_config import DEBUG, ORDER_VALIDATION_KEYS


errors = []
for key in [
    'AZURE_STORAGE_CONNECTION_STRING',  # For Azure Table operations
    'SEVENPAY_PID', 'SEVENPAY_PKEY',    # For payment validation
    'LOGIN_CALLBACK_ROUTE',             # Route for login API callback
    'PAYMENT_CALLBACK_ROUTE'            # Route for payment API callback
]:
    if key not in os.environ:
        errors.append(f"Environment variable {key} is not set")

build_date = "unknown"
if os.path.isfile("build_date.txt"):
    with open("build_date.txt", "r") as f:
        build_date = f.read().strip("\n")

azure_table_op = AzureTableOp()
app = Flask(__name__)


@app.route('/')
def index():
    # Default view to show to port scanners
    return jsonify({
        'build': build_date,
        'errors': errors
    })


@app.route(os.getenv('LOGIN_CALLBACK_ROUTE'), methods=['POST'])
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
                table_name += "Test"

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
                res['message'] = f"Failed to update entity: {table_res['message']}"
                return jsonify(res)

        except:
            res['errcode'] = 2
            res['message'] = f"Invalid request: {traceback.format_exc()}"
    return jsonify(res)


@app.route(os.getenv('PAYMENT_CALLBACK_ROUTE'), methods=['POST'])
def handle_sevenpay_validation():
    table_name = "orders"
    if DEBUG:
        table_name += "Test"

    if request.method == "POST":
        if DEBUG:
            logger.info(f"request data: {request.data}")
            logger.info(f"request form: {request.form}")
            logger.info(f"request args: {request.args}")
            logger.info(f"request json: {request.json}")

        try:
            json_data = request.json

            # Just dump the whole form data to orders table for debugging
            if DEBUG:
                entity = {
                    'PartitionKey': "DEBUG",
                    'RowKey': str(json_data['no']),
                    'data': json.dumps(json_data)
                }
                table_res = azure_table_op.update_entities(entity, table_name)

            # Step 1: Confirm that the signature is valid

            # First, pop the sign key from form data
            sevenpay_sign = json_data.pop('sign')

            # Second, generate our own signature to compare
            our_sign = get_md5_hash_7pay(
                json_data,
                os.getenv('SEVENPAY_PKEY')
            )
            if DEBUG:
                logger.info(json_data)
                logger.info(our_sign)
            if our_sign != sevenpay_sign:
                return f"Invalid signature: {our_sign} vs {sevenpay_sign}"

            # Step 2: Find the order and confirm that the data is correct
            query_filter = f"PartitionKey ne @debug and RowKey eq @order_id"
            select = None
            parameters = {'debug': "DEBUG", 'order_id': str(json_data['no'])}

            table_res = azure_table_op.query_entities(query_filter, select, parameters, table_name)
            if table_res['status'] != 0:
                return f"Failed to query orders: {table_res['message']}"

            # Perform order data validation
            if len(table_res['data']) <= 0:
                return "Invalid order_id"
            entity = table_res['data'][0]   # There should only be one order for the order_id
            order_data = json.loads(entity['data'])
            if DEBUG:
                logger.info(f"Order data: {order_data}")
            for our_key, sevenpay_key in ORDER_VALIDATION_KEYS:
                our_value, sevenpay_value = order_data[our_key], json_data[sevenpay_key]
                if our_key == "fee" and sevenpay_key == "money":
                    # Convert to float so they have the same number of decimals
                    our_value = float(our_value)
                    sevenpay_value = float(sevenpay_value)
                if str(our_value) != str(sevenpay_value):
                    return f"Mismatched key values: {our_key} ({our_value}) != {sevenpay_key} ({sevenpay_value})"

            # Step 3: Return early success if status is already paid
            if entity['status'] == "paid":
                return "success"

            # Step 4: Update the order status to paid and update table data
            entity['status'] = "paid"
            table_res = azure_table_op.update_entities(entity, table_name)
            if table_res['status'] != 0:
                return f"Failed to update order: {table_res['message']}"

            # Return success
            return "success"
        except:
            return f"Could not handle validation request: {traceback.format_exc()}"


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=DEBUG)
