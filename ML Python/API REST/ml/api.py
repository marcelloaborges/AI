from flask.views import MethodView

class MLApi(MethodView):

    def get(self):
        return "Machine Learning Home"