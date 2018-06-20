from flask.views import MethodView
from flask import jsonify, request, redirect, url_for

#ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
ALLOWED_EXTENSIONS = set(['csv'])

from core.ml.regression.polinomial_regression import PolynomialRegression

class PolynomialRegressionApi(MethodView):

    def get(self):
        return "Polynomial Regression Home", 200    

    def post(self):        
        if 'file' not in request.files:                        
            return jsonify('No file part'), 400

        file = request.files['file']

        if file.filename == '':                        
            return 'No selected file', 400

        if file and allowed_file(file.filename):    
            csv = file.stream.read().decode('utf-8')

            regressor = PolynomialRegression()        
            result = regressor.Run(csv)

            return jsonify(result), 200
        
        return 'Not allowed file', 400

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS