from flask import Blueprint

from ml.reinforcement_learning.AB_test.api import ABTestApi

ab_test_app = Blueprint('ab_test_app', __name__)

#VIEWS
ab_test_views = ABTestApi.as_view('ab_test')

#ROUTES
ab_test_app.add_url_rule('/ml/reinforcement_learning/ab_test', view_func=ab_test_views, methods=['GET',])

ab_test_app.add_url_rule('/ml/reinforcement_learning/ab_test', view_func=ab_test_views, methods=['POST',])