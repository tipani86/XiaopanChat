import os
import time
import json
import zlib
import hashlib
import traceback
from azure.data.tables import TableClient, UpdateMode
from azure.core.exceptions import ResourceExistsError, HttpResponseError, ResourceNotFoundError


def get_md5_hash_7pay(
    data: dict,
    pid: str,
    pkey: str,
) -> str:
    # Get md5 hash of input data based on 7-pay documentation
    # (Ref: http://7-pay.cn/doc.php#d6)

    # Step 1: Insert pid as one key-value pair into data
    data['pid'] = pid
    # Step 2: Build a string with all key-value pairs in data sorted alphabetically by key
    data_str = ""
    for key in sorted(data.keys()):
        data_str += f"{key}={data[key]}&"
    # Step 3: Append pkey to the end of the string but with a key 'key'
    data_str += f"key={pkey}"
    # Step 4: Strip the string of all empty spaces
    data_str = data_str.replace(" ", "")
    # Step 5: Return the lowercase version of the calculated md5 hash of the string
    return hashlib.md5(data_str.encode('utf-8')).hexdigest().lower()


class AzureTableOp:
    def __init__(
        self,
        BLOB_KEY: str = "AZURE_STORAGE_CONNECTION_STRING"
    ) -> None:
        self.connection_string = None
        self.NETWORK_RETRY_NUM = 3

        if BLOB_KEY not in os.environ:
            raise Exception(f"Environment variable {BLOB_KEY} missing!")
        self.connection_string = os.getenv(BLOB_KEY)

    def create_entity(
        self,
        entity,
        table_name: str = "test"
    ) -> dict:
        '''create a new entity, if you cannot make sure whether it exists, you can use update update_entities function.'''

        with TableClient.from_connection_string(self.connection_string, table_name) as table_client:
            for _ in range(self.NETWORK_RETRY_NUM):
                res = {'status': 0, 'message': "Success"}
                error_msg = []
                try:
                    table_client.create_table()
                except HttpResponseError as e:
                    msg = "Table already exists"
                    error_msg.append(f"{msg}: {e}")

                # * create entity
                try:
                    resp = table_client.create_entity(entity=entity)
                    return res
                except ResourceExistsError as e:
                    msg = "Entity already exists"
                    error_msg.append(f"{msg}: {e}")
                    res['message'] = ""
                    for msg in error_msg:
                        res['message'] += f"{msg}\n"
                    return res
                except Exception as e:
                    res['message'] = str(e)
                    time.sleep(5 * _)
            res['status'] = 1
            res['message'] = f"Error: entity creation failed after {self.NETWORK_RETRY_NUM} attempts: {res['message']}"
            return res

    def delete_entity(
        self,
        entity,
        table_name: str = "test"
    ) -> dict:
        '''delete entity'''
        with TableClient.from_connection_string(self.connection_string, table_name) as table_client:
            for _ in range(self.NETWORK_RETRY_NUM):
                res = {'status': 0, 'message': "Success"}
                try:
                    table_client.delete_entity(
                        row_key=entity['RowKey'],
                        partition_key=entity['PartitionKey']
                    )
                    return res
                except Exception as e:
                    res['message'] = str(e)
                    time.sleep(5 * _)
            res['status'] = 1
            res['message'] = f"Error: entity deletion failed after {self.NETWORK_RETRY_NUM} attempts: {res['message']}"
            return res

            # print("Successfully deleted!")

    def list_all_entities(
        self,
        table_name: str = "test"
    ) -> dict:
        '''return list of all entities'''
        with TableClient.from_connection_string(self.connection_string, table_name) as table_client:
            for _ in range(self.NETWORK_RETRY_NUM):
                res = {'status': 0, 'message': "Success", 'data': None}
                try:
                    entities = list(table_client.list_entities())
                    res['data'] = entities
                    return res
                except HttpResponseError as e:
                    res['message'] = str(e)
                    time.sleep(5 * _)
                except Exception as e:
                    res['status'] = 2
                    res['message'] = f"Exception while listing all entities: {e}"
                    return res
            res['status'] = 1
            res['message'] = f"Error: entity listing failed after {self.NETWORK_RETRY_NUM} attempts: {res['message']}"
            return res

    def query_entities(
        self,
        query_filter: str,
        select: list,
        parameters: dict,
        table_name: str = "test"
    ) -> dict:
        '''query entity, there are some samples in main following'''
        with TableClient.from_connection_string(self.connection_string, table_name) as table_client:
            for _ in range(self.NETWORK_RETRY_NUM):
                res = {'status': 0, 'message': "Success", 'data': None}
                results = []
                try:
                    queried_entities = table_client.query_entities(
                        query_filter=query_filter, select=select, parameters=parameters
                    )
                    for entity in queried_entities:
                        entity = self._decompress_entity(entity)
                        results.append(entity)
                    res['data'] = results
                    return res
                except HttpResponseError as e:
                    res['message'] = str(e)
                    time.sleep(5 * _)
                except Exception as e:
                    res['status'] = 2
                    res['message'] = f"Exception while querying entities: {e}"
                    return res
            res['status'] = 1
            res['message'] = f"Error: entity querying failed after {self.NETWORK_RETRY_NUM} attempts: {res['message']}"
            return res

    def get_entity(
        self,
        partition_key: str,
        row_key: str,
        table_name: str = "test"
    ) -> dict:
        '''get one entity based on partition_key and row_key.'''
        with TableClient.from_connection_string(self.connection_string, table_name) as table_client:
            for _ in range(self.NETWORK_RETRY_NUM):
                res = {'status': 0, 'message': "Success", 'data': None}
                try:
                    res_entity = table_client.get_entity(
                        partition_key=partition_key, row_key=row_key)
                    res['data'] = self._decompress_entity(res_entity)
                    return res
                except HttpResponseError as e:
                    res['message'] = str(e)
                    time.sleep(5 * _)
                except ResourceNotFoundError as e:
                    res['status'] = 2
                    res['message'] = f"Resource not found: {e}"
                    return res
            res['status'] = 1
            res['message'] = f"Error: get entity failed after {self.NETWORK_RETRY_NUM} attempts: {res['message']}"
            return res

    def update_entities(
        self,
        entity,
        table_name: str = "test"
    ) -> dict:
        """
        Check for size limit: The property value cannot exceed the maximum allowed size (64KB).
        If the property value is a string, it is UTF-16 encoded and the maximum number of characters should be 32K or less.
        """
        payload_size = len(json.dumps(entity).encode('utf-16'))
        if payload_size / 1024 > 30:
            entity = self._compress_entity(entity)

        '''update entity, if the entity doesn't exist, create it'''
        with TableClient.from_connection_string(self.connection_string, table_name=table_name) as table_client:
            for _ in range(self.NETWORK_RETRY_NUM):
                res = {'status': 0, 'message': "Success"}
                try:
                    table_client.create_table()
                except HttpResponseError:
                    pass
                try:
                    resp = table_client.upsert_entity(
                        mode=UpdateMode.REPLACE, entity=entity)
                    return res
                except Exception as e:
                    msg = f"Exception {e} encountered while upserting entity! Traceback: {traceback.format_exc()}"
                    res['message'] = msg
                    time.sleep(5 * _)
            res['status'] = 1
            res['message'] = f"Error: get entity failed after {self.NETWORK_RETRY_NUM} attempts: {res['message']}"
            return res

    def _decompress_entity(
        self,
        entity
    ) -> dict:
        decompressed_entity = {}
        for key in entity.keys():
            if type(entity[key]) == str and entity[key].startswith("gzip_"):
                decompressed_data = zlib.decompress(
                    eval(entity[key].split("gzip_", 1)[1])).decode('utf-8')
                decompressed_entity[key] = decompressed_data
            else:
                decompressed_entity[key] = entity[key]
        return decompressed_entity

    def _compress_entity(
        self,
        entity
    ) -> dict:
        ignore_keys = ["PartitionKey", "RowKey"]
        compressed_entity = {
            'PartitionKey': entity['PartitionKey'],
            'RowKey': entity['RowKey'],
        }
        for key in entity.keys():
            if key not in ignore_keys and type(entity[key]) == str:
                if entity[key].startswith("gzip_"):
                    compressed_entity[key] = entity[key]
                else:
                    compressed_entity[key] = f"gzip_{zlib.compress(entity[key].encode('utf-8'), level=9)}"

        payload_size = len(json.dumps(compressed_entity).encode('utf-8'))
        return compressed_entity


class User:
    def __init__(
        self,
        channel: str,
        user_id: str,
        db_op,
        table_name: str = "users"
    ) -> None:

        # Initialize the basic info
        self.channel = channel
        self.user_id = user_id
        self.db_op = db_op
        self.table_name = table_name

    def sync_from_db(self) -> dict:
        res = {'status': 0, 'message': "Success"}
        # Fetch the remaining info from database
        query_filter = f"PartitionKey eq @channel and RowKey eq @user_id"
        select = None
        parameters = {'channel': self.channel, 'user_id': self.user_id}
        db_res = self.db_op.query_entities(query_filter, select, parameters, self.table_name)
        if db_res['status'] != 0:
            return db_res
        if len(db_res['data']) <= 0:
            res['status'] = 3
            res['message'] = f"Channel {self.channel} user {self.user_id} not found in database"
            return res
        elif len(db_res['data']) > 1:
            res['status'] = 2
            res['message'] = f"Possible data error: more than one entry found with type {self.channel} and user_id {self.user_id}"
            return res
        else:
            # Load rest of the info
            entity = db_res['data'][0]
            self.n_tokens = entity['n_tokens']
            self.nickname = entity['nickname']
            self.avatar_url = entity['avatar_url']
            self.ip_history = json.loads(entity['ip_history'])
        return res

    def sync_to_db(self) -> dict:
        entity = {
            "PartitionKey": self.channel,
            "RowKey": self.user_id,
            "n_tokens": self.n_tokens,
            'nickname': self.nickname,
            'avatar_url': self.avatar_url,
            'ip_history': json.dumps(self.ip_history),
        }
        return self.db_op.update_entities(entity, self.table_name)

    def initialize_on_db(
        self,
        user_data: dict,
        initial_token_amount: int
    ) -> dict:

        entity = {
            "PartitionKey": self.channel,
            "RowKey": self.user_id,
            "n_tokens": initial_token_amount,
            'nickname': self.user_id[:10] if len(user_data['nickname']) <= 0 else user_data['nickname'],
            'avatar_url': user_data['avatar_url'],
            'ip_history': json.dumps(
                [(user_data['timestamp'], user_data['ip_address'])]
            )
        }

        db_res = self.db_op.update_entities(entity, self.table_name)
        if db_res['status'] != 0:
            return db_res
        return self.sync_from_db()

    def consume_token(self) -> dict:
        res = {'status': 0, 'message': "Success"}
        api_res = self.sync_from_db()
        if api_res['status'] != 0:
            return api_res
        if self.n_tokens <= 0:
            res['status'] = 2
            res['message'] = f"User {self.user_id} has no tokens left"
            return res
        self.n_tokens -= 1
        return self.sync_to_db()

    def update_ip_history(self, user_data: dict) -> dict:
        self.ip_history.append((user_data['timestamp'], user_data['ip_address']))
        # We only want to retain 10 latest ip addresses
        self.ip_history = self.ip_history[-10:]
        return self.sync_to_db()


if __name__ == "__main__":
    # azure_table_op = AzureTableOp()
    # azure_table_op._insert_random_entities()

    # #################* test create and delete one entity #########
    # entity = {
    #     "PartitionKey": "color",
    #     "RowKey": "brand",
    #     "text": "Marker",
    #     "color": "Purple",
    #     "price": "6",
    # }

    # azure_table_op.create_entity(entity=entity)

    # azure_table_op.delete_entity(entity=entity)

    # # ###################* test query ###################
    # #* find one cloumn's all data
    # query_filter = None
    # select = ["Value"]
    # parameters = None

    # # #* find item by single data filter
    # # query_filter = "Name eq @name and Brand eq @brand"
    # # select = ["Brand", "Color"]
    # # parameters = {"name": "marker", "brand": "Crayola"}

    # #* find item by range
    # query_filter = "Value gt @lower and Value lt @upper"
    # select = None
    # parameters = {"lower": 25, "upper": 50}

    # arrays = azure_table_op.query_entities(query_filter, select, parameters)
    # print(arrays)

    # ###################* test get one entity###################
    # partition_key = "pk"
    # row_key = "row100"
    # got_entitiy = azure_table_op.get_entity(partition_key, row_key)
    # print("Recieved entity: {}".format(got_entitiy))

    # ###################* test list all entities###############
    # all = azure_table_op.list_all_entities(table_name="table")
    # print(all)

    #################* test update entity##################
    # entity = {
    #     "PartitionKey": "hot",
    #     "RowKey": "brand",
    #     "text": "Marker",
    #     "color": "Purple",
    #     "price": "8",
    # }

    # table_name = "test1"

    # azure_table_op.create_entity(entity=entity, table_name=table_name)

    # azure_table_op.update_entities(entity=entity, table_name=table_name)

    # azure_table_op.delete_entity(entity, table_name=table_name)

    # query_filter = "RowKey eq @rowkey"
    # select = ["data"]
    # parameters = {"rowkey": "project_health"}
    # print(azure_table_op.query_entities(query_filter, select, parameters, table_name=table_name))

    # print(azure_table_op.list_all_entities(table_name=table_name))
    pass
