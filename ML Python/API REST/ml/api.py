from flask.views import MethodView

class ML(MethodView):

    def get(self):
        return "Machine Learning Home"