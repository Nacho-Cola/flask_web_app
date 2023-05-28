from flask import Flask, render_template, Response, jsonify
from flask import request
from camera import VideoCamera



playlist = ['tx1QSKI1UPs','i0RCcSBPjuU','6dDdkDifAxM']


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
        
@app.route('/video_feed')
def video_feed():
     return Response(gen(video_stream),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':

    app.run(debug=True)