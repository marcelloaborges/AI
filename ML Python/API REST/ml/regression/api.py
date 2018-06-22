from flask.views import MethodView
from flask import render_template

class RegressionApi(MethodView):

    def get(self):
        return render_template('ml/regression/index.html')