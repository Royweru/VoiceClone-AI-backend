from celery import shared_task
from django.utils import timezone
from .models import VoiceSample, VoiceModel, TrainingTask
from .voice_utils import VoiceTrainer, TrainingConfig
import logging

logger = logging.getLogger(__name__)

@shared_task
def validate_voice_sample_task(sample_id):
    """Validate a single voice sample"""
    try:
        sample = VoiceSample.objects.get(id=sample_id)
        sample.status = VoiceSample.SampleStatus.PROCESSING
        sample.save()
        
        # Initialize trainer and validate
        trainer = VoiceTrainer()
        result = trainer.process_sample(sample_id)
        
        if result['valid']:
            logger.info(f"Sample {sample_id} validated successfully")
            return {'success': True, 'sample_id': sample_id}
        else:
            logger.warning(f"Sample {sample_id} validation failed: {result['reason']}")
            return {'success': False, 'sample_id': sample_id, 'reason': result['reason']}
            
    except Exception as e:
        logger.error(f"Validation task failed for sample {sample_id}: {str(e)}")
        # Mark sample as invalid if task fails
        try:
            sample = VoiceSample.objects.get(id=sample_id)
            sample.status = VoiceSample.SampleStatus.INVALID
            sample.save()
        except:
            pass
        raise

from django.db import IntegrityError, transaction

@shared_task(bind=True)
def train_voice_model_task(self, user_id):
    """Train a voice model for the user"""
    task_record = None
    try:
        # Handle task record creation with race condition protection
        try:
            with transaction.atomic():
                task_record, created = TrainingTask.objects.get_or_create(
                    user_id=user_id,
                    task_id=self.request.id,
                    defaults={
                        'status': VoiceModel.TrainingStatus.PENDING,
                        'progress': 0
                    }
                )
                if not created:
                    # Task already exists, check its status
                    if task_record.status in [VoiceModel.TrainingStatus.COMPLETED, VoiceModel.TrainingStatus.FAILED]:
                        logger.info(f"Task {self.request.id} already processed with status: {task_record.status}")
                        return {'success': False, 'message': 'Task already processed'}
        except IntegrityError:
            # Race condition occurred, try to get the existing record
            task_record = TrainingTask.objects.get(task_id=self.request.id)
            if task_record.status in [VoiceModel.TrainingStatus.COMPLETED, VoiceModel.TrainingStatus.FAILED]:
                logger.info(f"Task {self.request.id} already processed with status: {task_record.status}")
                return {'success': False, 'message': 'Task already processed'}
        
        # Update status to running
        task_record.status = VoiceModel.TrainingStatus.TRAINING
        task_record.save()
        
        # Initialize trainer and start training
        trainer = VoiceTrainer()
        result = trainer.train_voice_model(user_id)
        
        if result.get('success'):
            # Create voice model record with current timestamp
            voice_model = VoiceModel.objects.create(
                user_id=user_id,
                model_path=result['model_path'],
                config_path=result.get('config_path', ''),
                status=VoiceModel.TrainingStatus.COMPLETED,
                sample_count=result['sample_count'],
                completed_at=timezone.now(),
                is_active=True
            )
            
            # Deactivate other models for this user
            VoiceModel.objects.filter(
                user_id=user_id,
                is_active=True
            ).exclude(id=voice_model.id).update(is_active=False)
            
            # Update task record
            task_record.status = VoiceModel.TrainingStatus.COMPLETED
            task_record.voice_model = voice_model
            task_record.progress = 100
            task_record.save()
            
            logger.info(f"Training completed for user {user_id}")
            return {'success': True, 'model_id': voice_model.id}
        else:
            raise Exception("Training failed without exception")
        
    except Exception as e:
        logger.error(f"Training failed for user {user_id}: {str(e)}", exc_info=True)
        
        # Update task record with failure
        if task_record:
            task_record.status = VoiceModel.TrainingStatus.FAILED
            task_record.error_message = str(e)[:500]  # Limit error message length
            task_record.save()
        
        # Create failed model record for debugging
        VoiceModel.objects.create(
            user_id=user_id,
            model_path='',
            status=VoiceModel.TrainingStatus.FAILED,
            error_message=str(e)[:500],
            completed_at=timezone.now()
        )
        
        raise self.retry(exc=e, countdown=60, max_retries=3)