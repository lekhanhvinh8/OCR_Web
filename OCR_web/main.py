from flask import Flask, render_template, request
from flask_restful import  Resource, Api, reqparse
import werkzeug
from services.ocr_service import OCRService
import cv2
import numpy as np

ocr_service = OCRService()

app = Flask(__name__)
api = Api(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ourwork')
def ourwork():
    return render_template('ourwork.html')

class UploadImages(Resource):
    def post(self):
        if 'files' not in request.files:
            return 'No file part'

        files = request.files.getlist('files')
        result = []

        for file in files:
            #read image file string data
            filestr = file.read()
            #convert string data to numpy array
            npimg = np.fromstring(filestr, np.uint8)
            # convert numpy array to image
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

            fin_boxes = ocr_service.detect(img)
            final_texts = ocr_service.recognit_many(img, fin_boxes)
            restructure = ocr_service.restructure(img, fin_boxes, final_texts)

            fisrt_char_line = 0

            # get rid of all white space line
            new_restructure_box = []

            for row in restructure:
                if set("".join(row)) != {' '}:
                    new_row = [word for word in row if word != " "]

                    new_restructure_box.append(" ".join(new_row))

            new_restructure_box = "\n".join(["".join(row) for row in new_restructure_box])

            result.append("---------------------------------------------------------------")
            result.append(new_restructure_box)


        return "\n".join(result)

api.add_resource(UploadImages, "/api/upload")

if __name__ == '__main__':
    app.run(debug=False)
