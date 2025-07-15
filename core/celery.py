# celery.py (in your project root, same level as settings.py)
import os
from celery import Celery

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

app = Celery('core')

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Load task modules from all registered Django apps.
app.autodiscover_tasks()

# Optional: Add explicit task discovery if autodiscover isn't working
app.autodiscover_tasks([
    'voiceapi',  # Make sure this matches your app name
    # Add other apps that contain tasks
])

@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')