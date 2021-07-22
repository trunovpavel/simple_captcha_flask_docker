from flask import Flask, request, flash, redirect
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from predict import predict

UPLOAD_FOLDER = './uploads/'
ALLOWED_EXTENTIONS = set(['jpg', 'jpeg', 'png'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENTIONS

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'no file part: ' + str(request.files)
        file = request.files['file']
        if file.filename == '':
            return('no selected file')
        if not file:
            return 'no file'
        if file and not allowed_file(file.filename):
            return 'filename is not allowed: ' + file.filename

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            saved_filename = filename + datetime.today().strftime('_%Y_%m_%d__%H:%M:%S')+'.jpg'
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], saved_filename))
            print(UPLOAD_FOLDER + saved_filename)
            y_pred, prob = predict(UPLOAD_FOLDER + saved_filename)
            return ''.join(y_pred)
            #return f'y_pred: {y_pred}, prob: {prob}'
        return 'request failed'
    else:
        return 'secret service'


if __name__ == '__main__':
    app.run(debug=False)