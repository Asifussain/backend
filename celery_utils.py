import os
from celery import Celery
from config import REDIS_URL

is_secure_redis = REDIS_URL.startswith("rediss://")

ssl_options = {'ssl_cert_reqs': 'none'} if is_secure_redis else {}

celery_app = Celery(
    __name__,
    broker=REDIS_URL,
    backend=REDIS_URL,
    broker_use_ssl=ssl_options,
    redis_backend_use_ssl=ssl_options,
    imports=('routes.predict_api',)
)

print("--- Celery instance created and tasks imported ---")