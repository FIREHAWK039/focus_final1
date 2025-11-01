from flask import Flask, Response, render_template, jsonify
import threading
import time
from focus_detector import FocusDetector

app = Flask(__name__)

# Start detector in a separate thread so Flask can serve pages
detector = FocusDetector(camera_idx=0, enable_logging=False)

def mjpeg_generator():
    while True:
        jpg, focus, features = detector.get_frame()
        if jpg is None:
            time.sleep(0.05)
            continue
        frame = jpg
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(mjpeg_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/focus')
def focus_api():
    # return latest focus value and features
    jpg, focus, features = detector.get_frame()
    if jpg is None:
        return jsonify({'focus': None, 'features': None}), 503
    return jsonify({'focus': float(focus), 'features': features})


def run_app():
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    finally:
        detector.close()


if __name__ == '__main__':
    run_app()
