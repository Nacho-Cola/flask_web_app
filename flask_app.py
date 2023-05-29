from flask import Flask, render_template, Response, jsonify, redirect, url_for
from flask import request
from camera import VideoCamera
from detect import detect_img
import numpy as np

app = Flask(__name__)

video_stream = VideoCamera()


@app.route('/')

def index():

    return render_template('index.html')

# report 페이지에 주요 변수들과 함수들을 정의해 준다.

@app.route('/play')
def play():
    ytid = request.args.get('ytid')
    return render_template('play.html',ytid=ytid)  

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
@app.route('/up_load', methods=["POST"])
def up_load():
    global i
    i = 0
    uploaded_files = request.files["file"]
    uploaded_files.save("static/img/detect{}.jpeg".format(i))
    img, ret = detect_img("static/img/detect{}.jpeg".format(i)) #img 얼굴 사진, ret 결과
    i+=1
    return render_template('result.html',ret = ret, i=i-1, img=img)



@app.route('/video_feed')
def video_feed():
     return Response(gen(video_stream),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':

    app.run(debug=True)