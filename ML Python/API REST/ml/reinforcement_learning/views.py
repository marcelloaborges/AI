from flask import Blueprint

from ml.reinforcement_learning.api import ReinforcementLearningApi

reinforcement_learning_app = Blueprint('reinforcement_learning_app', __name__)

reinforcement_learning_views = ReinforcementLearningApi.as_view('reinforcement_learning')

reinforcement_learning_app.add_url_rule('/ml/reinforcement_learning', view_func=reinforcement_learning_views, methods=['GET',])