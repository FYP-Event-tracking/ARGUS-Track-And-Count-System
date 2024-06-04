from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio
import websockets
import datetime
import logging

app = Flask(__name__)
CORS(app) 

logging.basicConfig(level=logging.INFO)

clients = set()

async def handler(websocket, path):
    clients.add(websocket)
    try:
        while True:
            data = await websocket.recv()
            log_received_data(data) 
            reply = f"Data received as: {data}!"
            await websocket.send(reply)
    except websockets.ConnectionClosed:
        clients.remove(websocket)

def log_received_data(data):
    log_id = None
    box_id = None
    item_type = None
    user_id = None
    start_time = None
    
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
        elif line.startswith("StartTime:"):
            start_time = line.split(":")[1]
        logging.info(f"LogId: {log_id}, BoxId: {box_id}, ItemType: {item_type}, UserId: {user_id}, StartTime: {start_time}")

if __name__ == "__main__":
    start_server = websockets.serve(handler, host='0.0.0.0', port=8009)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
