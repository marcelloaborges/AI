from flask.views import MethodView

class ReinforcementLearningApi(MethodView):

    def get(self):
        return "Reinforcement Learning Home"