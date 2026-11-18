from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

db = SQLAlchemy()
DB_NAME = "database.db"

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'sarthaklikesmen menlikesavipriya'
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
    db.init_app(app)

    

    from .views import views
    from .auth import auth

    app.register_blueprint(views, url_prefix = '/')
    app.register_blueprint(auth, url_prefix = '/auth/')

    from .models import User, Detection

    create_database(app)
    login_manager = LoginManager()
    login_manager.login_view = 'views.index'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id))


    return app

def create_database(app):
    # Flask-SQLAlchemy resolves relative SQLite URLs from app.instance_path.
    # Calling create_all is idempotent and avoids checking a different path
    # based on whichever directory was used to launch the application.
    with app.app_context():
        db.create_all()
        
