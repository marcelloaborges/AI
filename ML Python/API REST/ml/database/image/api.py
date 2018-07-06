import os, sys
from flask.views import MethodView
from flask import jsonify, request, redirect, url_for
from flask import render_template
import uuid
from werkzeug.utils import secure_filename
import io
from PIL import Image as img

from application import db
from ml.database.image.models.image import Image

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'bmp'])
IMAGE_LABEL = \
    { 
        0: 'caminhao', 
        1: 'lata'
    }
IMAGE_LABEL_LATA = \
    {
        0: 'skol', 
        1: 'brahma',
        2: 'antartica',
    }

class DatabaseImageApi(MethodView):

    def get(self): 
        return render_template('ml/database/image/index.html'), 200    

    def post(self):                   
        if not request.files:                        
            return jsonify('No file part'), 400

        files = request.files

        for f in files:
            file = files.get(f)

            if file.filename == '':                        
                return 'No selected file', 400        

            if(allowed_file(file.filename)):
                category = request.form.get('category')
                sub_category = request.form.get('sub_category')
                filename = secure_filename(file.filename)

                image = Image()         
                
                image.name = filename
                image.category = category
                image.sub_category = sub_category

                image.file.put(file)

                print('saving...')
                image.save()
            
            else:
                return 'Not allowed file \n alowed [png, jpg, bmp]', 400

        return jsonify('sucess'), 200        
            

def allowed_file(filename):
    return '.' in filename and \
        get_extension(filename) in ALLOWED_EXTENSIONS

def get_extension(filename):
    return filename.rsplit('.', 1)[1].lower()

class DatabaseImageExportApi(MethodView):

    def post(self):
        directory = 'ml/database/image/export'

        images = Image.objects
        for i in images:
            _id = uuid.uuid4()
            extension = '.' + get_extension(i.name)
            file = i.file.read()
            image = img.open(io.BytesIO(file))            
            image.save(directory + '/' + str(_id) + extension)

        return jsonify('exported'), 200