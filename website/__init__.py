from flask import Flask


def create_app():
    app = Flask(__name__, template_folder="template")
    app.config[
        "SECRET_KEY"
    ] = "GHRD^1jN7RyQ}}JkYn%4pjMQ7[.r-q(7&+k[av#|Z_JC[7`j&wjO})Am4kbbH6a"

    from .views import views

    app.register_blueprint(views, url_prefix="/")

    return app
