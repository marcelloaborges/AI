from flask.views import MethodView

class RegressionApi(MethodView):

    def get(self):
        return "Regression Home"