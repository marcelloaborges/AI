from flask import Flask
from flask_mongoengine import MongoEngine

db = MongoEngine()

def create_app(**config_overrides):
    app = Flask(__name__)

    # Load config
    app.config.from_pyfile('settings.py')

    # apply overrides for tests
    app.config.update(config_overrides)            

    #db
    db = MongoEngine(app)

    # import blueprints
    from home.views import home_app    
    from ml.views import ml_app

    from ml.regression.views import regression_app
    from ml.regression.simple_linear_regression.views import simple_linear_regression_app
    from ml.regression.polynomial_regression.views import polynomial_regression_app
    from ml.regression.logistic_regression.views import logistic_regression_app

    from ml.reinforcement_learning.views import reinforcement_learning_app
    from ml.reinforcement_learning.AB_test.views import ab_test_app

    from ml.database.image.views import database_image_app    
    from ml.database.image.views import database_image_export_app

    # register blueprints
    app.register_blueprint(home_app)    
    app.register_blueprint(ml_app)

    app.register_blueprint(regression_app)
    app.register_blueprint(simple_linear_regression_app)
    app.register_blueprint(polynomial_regression_app)
    app.register_blueprint(logistic_regression_app)

    app.register_blueprint(reinforcement_learning_app)
    app.register_blueprint(ab_test_app)

    app.register_blueprint(database_image_app)
    app.register_blueprint(database_image_export_app)

    return app