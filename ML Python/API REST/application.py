from flask import Flask

def create_app(**config_overrides):
    app = Flask(__name__)

    # Load config
    app.config.from_pyfile('settings.py')

    # apply overrides for tests
    app.config.update(config_overrides)        

    # import blueprints
    from home.views import home_app    
    from ml.views import ml_app
    
    from ml.regression.simple_linear_regression.views import simple_linear_regression_app

    # register blueprints
    app.register_blueprint(home_app)    
    app.register_blueprint(ml_app)

    app.register_blueprint(simple_linear_regression_app)

    return app