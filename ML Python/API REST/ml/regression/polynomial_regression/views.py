from flask import Blueprint

from ml.regression.polynomial_regression.api import PolynomialRegressionApi

polynomial_regression_app = Blueprint('polynomial_regression_app', __name__)

#VIEWS
polynomial_regression_view = PolynomialRegressionApi.as_view('polynomial_regression_api')

#ROUTES
polynomial_regression_app.add_url_rule('/ml/regression/polynomial_regression', endpoint="", 
    view_func=polynomial_regression_view, methods=['GET',])

polynomial_regression_app.add_url_rule('/ml/regression/polynomial_regression', 
    view_func=polynomial_regression_view, methods=['POST',])