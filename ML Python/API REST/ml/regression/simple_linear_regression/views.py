from flask import Blueprint

from ml.regression.simple_linear_regression.api import SimpleLinearRegression

simple_linear_regression_app = Blueprint('simple_linear_regression_app', __name__)

#VIEWS
simple_linear_regression_view = SimpleLinearRegression.as_view('simple_linear_regression_api')

#ROUTES
simple_linear_regression_app.add_url_rule('/ml/simple_linear_regression', endpoint="", 
    view_func=simple_linear_regression_view, methods=['GET',])

simple_linear_regression_app.add_url_rule('/ml/simple_linear_regression', 
    view_func=simple_linear_regression_view, methods=['POST',])

# regression_app.add_url_rule('/ml/regression/polynomial_regression', view_func=regression_view, methods=['POST',])

# regression_app.add_url_rule('/ml/regression/logistic_regression', view_func=regression_view, methods=['POST',])