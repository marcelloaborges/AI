from flask import Blueprint

from ml.database.image.api import DatabaseImageApi
from ml.database.image.api import DatabaseImageExportApi

database_image_app = Blueprint('database_image_app', __name__)
database_image_export_app = Blueprint('database_image_export_app', __name__)

#VIEWS
database_image_view = DatabaseImageApi.as_view('database_image_api')
database_image_export_view = DatabaseImageExportApi.as_view('database_image_export_api')

#ROUTES
database_image_app.add_url_rule('/ml/database/image', endpoint="", 
    view_func=database_image_view, methods=['GET', 'POST'])

database_image_export_app.add_url_rule('/ml/database/image/export', endpoint="", 
    view_func=database_image_export_view, methods=['POST'])