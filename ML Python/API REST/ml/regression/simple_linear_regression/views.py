from flask import Blueprint

from ml.regression.simple_linear_regression.api import SimpleLinearRegressionApi

simple_linear_regression_app = Blueprint('simple_linear_regression_app', __name__)

#VIEWS
simple_linear_regression_view = SimpleLinearRegressionApi.as_view('simple_linear_regression_api')

#ROUTES
simple_linear_regression_app.add_url_rule('/ml/regression/simple_linear_regression', endpoint="", 
    view_func=simple_linear_regression_view, methods=['GET', 'POST'])