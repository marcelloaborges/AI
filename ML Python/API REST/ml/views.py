from flask import Blueprint

from ml.api import ML

ml_app = Blueprint('ml_app', __name__)

ml_views = ML.as_view('ml')

ml_app.add_url_rule('/ml', view_func=ml_views, methods=['GET',])