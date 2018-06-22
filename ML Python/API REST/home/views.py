from flask import Blueprint
from flask import render_template

home_app = Blueprint('home_app', __name__)

@home_app.route('/')
def home():
    return render_template('home/index.html')