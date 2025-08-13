import os
from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from routes import api_bp  
from celery_utils import celery_app  

load_dotenv()  

def create_app():
    app = Flask(__name__)

    frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:3000')
    origins = [frontend_url]
    if "localhost" not in frontend_url:
        origins.append("http://localhost:3000")
    CORS(app, resources={r"/api/*": {"origins": origins}})

    app.register_blueprint(api_bp, url_prefix='/api')  

    # Configure Celery with Flask context
    celery_app.conf.update(app.config)
    class ContextTask(celery_app.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)
    celery_app.Task = ContextTask

    @app.route('/')
    def index():
        return jsonify({"message": "Backend is alive and running!"})

    return app

app = create_app()

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
