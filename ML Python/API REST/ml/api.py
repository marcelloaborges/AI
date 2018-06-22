from flask.views import MethodView
from flask import render_template

class MLApi(MethodView):

    def get(self):
        return render_template('ml/index.html')