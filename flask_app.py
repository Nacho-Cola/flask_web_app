from flask import Flask, render_template, Response
from flask import request
from camera import get_frame
from detect import detect_img
from io import BytesIO
import numpy as np
from PIL import Image
import cv2 as cv
import base64

from time import time
import time 

tm =time.localtime()

app = Flask(__name__)
@app.route('/')

def index():

    return render_template('index.html')

@app.route('/play',methods=['GET','POST'])
def play():
    ytid = request.args.get('ytid')
    return render_template('play.html',ytid=ytid)  


@app.route('/send_pic',methods=['GET','POST'])
def send_pic():
    print("Image recieved")
    data_url = request.data
    # data_url = data_url.decode('utf-8')
    # offset = data_url.index(',')+1 
    if data_url == b'':
        return ""
    img_bytes = base64.b64decode(data_url[22:])
    # img = Image.open(BytesIO(img_bytes))
    # img  = np.array(img) 
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)

    
    img = get_frame(img)
    img = img.tobytes()
    img = b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n\r\n'
    return Response(str(img),minetype='multipart/x-mixed-replace;boundary=frame')
    


@app.route('/up_load', methods=["POST"])
def up_load():
    global i
    i = 0
    uploaded_files = request.files["file"] 
    uploaded_files.save("static/img/detect{}.jpeg".format(i))
    img, ret = detect_img("static/img/detect{}.jpeg".format(i)) #img,method='POST 얼굴 사진, ret 결과
    i+=1
    return render_template('result.html',ret = ret, i=i-1, img=img)




# @app.route('/video_feed')
# def video_feed():
#      return render_template('video_feed.html')

# @app.route('/post', methods=["POST"])
# def post():
#     global im
#     im = request.get_json()
#     im = im.get('data')
#     print(im)
#     sleep(0.1)
#     imgdata = base64.b64decode(im)
#     dataBytesIO = io.BytesIO(imgdata)
#     image = Image.open(dataBytesIO)
#     img = cv.cvtColor(np.array(image), cv.COLOR_BGR2RGB)
#     frame = get_frame(img)
#     print(frame)
#     return jsonify({"data": im, "status": HTTPStatus.OK})
#             yield (b'Content-Type: image/jpeg\r\n\r\n'+ img + b'\r\n')
if __name__ == '__main__':

    app.run(host="0.0.0.0",debug=True)