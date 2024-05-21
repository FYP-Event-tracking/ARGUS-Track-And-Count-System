from flask import Flask, request, jsonify
from flask_swagger_ui import get_swaggerui_blueprint
from flask_cors import CORS
import numpy as np
import supervision as sv
from ultralytics import YOLO
import os
import glob

app = Flask(__name__)
CORS(app) 

SWAGGER_URL = '/swagger'  
API_URL = '/static/swagger.json' 

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,  
    API_URL,
    config={  
        'app_name': "Track and Count System"
    }
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)


def model_run(SOURCE_VIDEO_PATH, TARGET_VIDEO_PATH):
    model = YOLO(os.path.relpath("best.pt"))
    model.fuse()

    LINE_START = sv.Point(1000, 1000)
    LINE_END = sv.Point(1500, 200)

    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

    generator = sv.video.get_video_frames_generator(SOURCE_VIDEO_PATH)

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

    return in_count, out_count

@app.route('/submit', methods=['POST'])
def submit():
    data = request.get_json()

    LogId = data.get('LogId')
    BoxId = data.get('BoxId')
    ItemType = data.get('ItemType')
    UserId = data.get('UserId')
    StartTime = data.get('StartTime')

    in_count, out_count = model_run("source.mp4", "target.mp4")


    if not Log or not Count or not BoxId or not ItemType or not UserId or not StartTime:
        return jsonify({'error': 'Incomplete Request'}), 400

    response = {
        'message': f'LogId: {LogId}, BoxId: {BoxId}, ItemType: {ItemType}, UserId: {UserId}, StartTime: {StartTime}, InCount: {in_count}, OutCount: {out_count}'
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8003)
