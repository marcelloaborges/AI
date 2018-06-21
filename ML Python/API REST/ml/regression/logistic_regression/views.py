from flask import Blueprint

from ml.regression.logistic_regression.api import LogisticRegressionApi

logistic_regression_app = Blueprint('logistic_regression_app', __name__)

#VIEWS
logistic_regression_view = LogisticRegressionApi.as_view('logistic_regression_api')

#ROUTES
logistic_regression_app.add_url_rule('/ml/regression/logistic_regression', endpoint="", 
    view_func=logistic_regression_view, methods=['GET', 'POST'])