from flask.views import MethodView
from flask import jsonify, request, redirect, url_for
from flask import render_template

#ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
ALLOWED_EXTENSIONS = set(['csv'])

from core.ml.regression.simple_linear_regression import SimpleLinearRegression

class SimpleLinearRegressionApi(MethodView):

    def get(self):
        return render_template('ml/regression/simple_linear_regression/index.html'), 200    

    def post(self):        
        if 'file' not in request.files:                        
            return jsonify('No file part'), 400

        file = request.files['file']

        if file.filename == '':                        
            return 'No selected file', 400

        if file and allowed_file(file.filename):    
            csv = file.stream.read().decode('utf-8')

            regressor = SimpleLinearRegression()        
            result = regressor.Run(csv)

            return jsonify(result), 200
        
        return 'Not allowed file', 400

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS