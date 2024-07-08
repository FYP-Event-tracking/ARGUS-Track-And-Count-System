from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import supervision as sv
from ultralytics import YOLO
import os
import glob
import logging
import requests
import json
import datetime
import pytz

app = Flask(__name__)
CORS(app) 

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

log_id = None
box_id = None
item_type = None
user_id = None
start_time = None
image_data = None
frame = None
logs = []

IST = pytz.timezone('Asia/Kolkata')

def model_run(SOURCE_VIDEO_PATH, TARGET_VIDEO_PATH):
    model = YOLO(os.path.relpath("best.pt"))
    model.fuse()

    LINE_START = sv.Point(150, 1000)
    LINE_END = sv.Point(1100, 100)

    line_counter = sv.LineZone(start=LINE_START, end=LINE_END)

    line_annotator = sv.LineZoneAnnotator(
        thickness=4, 
        text_thickness=4, 
        text_scale=2
    )

    box_annotator = sv.BoxAnnotator(
        thickness=4,
        text_thickness=4,
        text_scale=2
    )

    in_count = 0
    out_count = 0

    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

    generator = sv.video.get_video_frames_generator(SOURCE_VIDEO_PATH)

    with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:  
        for result in model.track(source=SOURCE_VIDEO_PATH, tracker = 'bytetrack.yaml', show=False, stream=True, agnostic_nms=True, persist=True ):

            frame = result.orig_img
            detections = sv.Detections.from_yolov8(result)

            if result.boxes.id is not None:
                detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
            
            labels = [
                f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, tracker_id
                in detections
            ]

            line_counter.trigger(detections=detections)
            line_annotator.annotate(frame=frame, line_counter=line_counter)
            frame = box_annotator.annotate(
                scene=frame, 
                detections=detections,
                labels=labels
            )
            in_count = line_counter.in_count
            out_count = line_counter.out_count
            sink.write_frame(frame)
            time = datetime.datetime.now().isoformat()
            time_dt = datetime.datetime.fromisoformat(time)
            time_str = time_dt.strftime("%Y-%m-%d %H:%M:%S")

            log = f"{time_str}    Current-Count: {in_count}  ->  {labels}"
            logs.append(log + "\n")

    return in_count, out_count

def send_data_to_backend(in_count):
    global log_id, box_id, item_type, user_id, start_time, logs
    logs_str = " ".join(logs)

    end_time = datetime.datetime.now().isoformat()

    payload = {
        "logId": log_id,
        "boxId": box_id,
        "itemType": item_type,
        "userId": user_id,
        "totalCount": in_count,
        "startTime": start_time,
        "endTime": end_time,
        "fullLogFile": logs_str
    }

    try:
        response = requests.post("http://localhost:8007/log/api/Log", json=payload)
        response.raise_for_status()
        logging.info(f"Successfully sent data to endpoint: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error sending data to endpoint: {e}")

@app.route('/submit', methods=['POST'])
def upload_file():
    global log_id, box_id, item_type, user_id, start_time
    if 'file' not in request.files:
        print('No file part')
        return jsonify({'message': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        print('No selected file')
        return jsonify({'message': 'No selected file'}), 400

    if file and file.filename.lower().endswith('.mp4'):
        log_id = request.form.get('logId')
        box_id = request.form.get('boxId')
        item_type = request.form.get('itemType')
        user_id = request.form.get('userId')
        start_time = datetime.datetime.now().isoformat()

        if not log_id or not box_id or not item_type:
            print('Missing form data')
            return jsonify({'message': 'Missing form data'}), 400

        filename = file.filename
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        source_video_path = os.path.join(UPLOAD_FOLDER, filename)
        target_video_path = os.path.join(UPLOAD_FOLDER, f"{log_id}.mp4")
        in_count, out_count = model_run(source_video_path, target_video_path)
        send_data_to_backend(in_count)
        return jsonify({'message': 'File successfully processed'}), 200
    else:
        print('Invalid file type')
        return jsonify({'message': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8011)