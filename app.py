from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio
import websockets
import logging
import requests
import json
import datetime

app = Flask(__name__)
CORS(app) 

logging.basicConfig(level=logging.INFO)

clients = set()

log_id = None
box_id = None
item_type = None
user_id = None
start_time = None
logs = []

async def handler(websocket, path):
    global log_id, box_id, item_type, user_id, start_time, logs
    clients.add(websocket)
    try:
        initial_data = await websocket.recv()
        get_data(initial_data)
        while True:
            try:
                data = await websocket.recv()
                log_info()
            except:
                logging.info("Client disconnected")
                break
    finally:
        clients.remove(websocket)
        send_data_to_backend()
        await websocket.close()

def send_data_to_backend():
    global log_id, box_id, item_type, user_id, start_time, logs
    logs_str = " ".join(logs)

    end_time = datetime.datetime.now().isoformat()

    payload = {
        "logId": log_id,
        "boxId": box_id,
        "itemType": item_type,
        "userId": user_id,
        "totalCount": 0,
        "startTime": start_time,
        "endTime": end_time,
        "fullLogFile": logs_str
    }

    try:
        response = requests.post("http://host.docker.internal:8007/log/api/Log", json=payload)
        response.raise_for_status()
        logging.info(f"Successfully sent data to endpoint: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error sending data to endpoint: {e}")

def get_data(data):
    global log_id, box_id, item_type, user_id, start_time
    data_lines = data.decode("utf-8").split("\n")
    for line in data_lines:
        if line.startswith("LogId:"):
            log_id = line.split(":")[1]
        elif line.startswith("BoxId:"):
            box_id = line.split(":")[1]
        elif line.startswith("ItemType:"):
            item_type = line.split(":")[1]
        elif line.startswith("UserId:"):
            user_id = line.split(":")[1]
    start_time = datetime.datetime.now().isoformat()
    
def log_info():
    global log_id, box_id, item_type, user_id
    log_time = datetime.datetime.now().isoformat()
    logging.info(f"LogId: {log_id}, BoxId: {box_id}, ItemType: {item_type}, UserId: {user_id}, LogTime: {log_time}")

    log = f"LogId: {log_id}, BoxId: {box_id}, ItemType: {item_type}, UserId: {user_id}, LogTime: {log_time}"
    logs.append(log + "\n")
    
if __name__ == "__main__":
    start_server = websockets.serve(handler, host='0.0.0.0', port=8009)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
