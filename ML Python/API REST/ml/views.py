from flask import Blueprint

from ml.api import MLApi

ml_app = Blueprint('ml_app', __name__)

ml_views = MLApi.as_view('ml')

ml_app.add_url_rule('/ml', view_func=ml_views, methods=['GET',])