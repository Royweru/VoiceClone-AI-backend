from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator
from django.utils import timezone

class VoiceSample(models.Model):
    class SampleStatus(models.TextChoices):
        UPLOADED = 'uploaded', 'Uploaded'
        PROCESSING = 'processing', 'Processing'
        VALID = 'valid', 'Valid'
        INVALID = 'invalid', 'Invalid'
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='voice_samples')
    audio_file = models.FileField(upload_to='voice_samples/%Y/%m/%d/')
    duration = models.FloatField(null=True, blank=True)  # in seconds
    sample_rate = models.IntegerField(null=True, blank=True)  # in Hz
    channels = models.IntegerField(default=1)
    file_size = models.BigIntegerField(null=True , blank=True)  # in bytes
    status = models.CharField(
        max_length=10,
        choices=SampleStatus.choices,
        default=SampleStatus.UPLOADED
    )
    transcription = models.TextField(null=True, blank=True)
    audio_quality = models.FloatField(
        null=True, 
        blank=True,
        validators=[MinValueValidator(0), MaxValueValidator(1)]
    )
    created_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', 'status']),
            models.Index(fields=['created_at']),
        ]
    
    def __str__(self):
        return f"Sample {self.id} - {self.user.username}"

class VoiceModel(models.Model):
    class TrainingStatus(models.TextChoices):
        PENDING = 'pending', 'Pending'
        TRAINING = 'training', 'Training'
        COMPLETED = 'completed', 'Completed'
        FAILED = 'failed', 'Failed'
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='voice_models')
    name = models.CharField(max_length=100, blank=True)
    model_path = models.CharField(max_length=255)
    config_path = models.CharField(max_length=255,default="models/default_config.json")
    is_active = models.BooleanField(default=False)
    status = models.CharField(
        max_length=10,
        choices=TrainingStatus.choices,
        default=TrainingStatus.PENDING
    )
    progress = models.FloatField(default=0)  # 0-100
    sample_count = models.IntegerField(null=True, blank=True)
    epochs = models.IntegerField(default=100)
    batch_size = models.IntegerField(default=16)
    learning_rate = models.FloatField(default=1e-4)
    training_time = models.FloatField(null=True, blank=True)  # in seconds
    loss = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        constraints = [
            models.UniqueConstraint(
                fields=['user'],
                condition=models.Q(is_active=True),
                name='unique_active_model_per_user'
            )
        ]
    
    def save(self, *args, **kwargs):
        if not self.name:
            # Use current time if created_at isn't set yet
            timestamp = self.created_at if self.created_at else timezone.now()
            self.name = f"{self.user.username}'s Model {timestamp.strftime('%Y-%m-%d')}"
        super().save(*args, **kwargs)
    def __str__(self):
        return f"Model {self.id} - {self.user.username}"

class TrainingTask(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='training_tasks')
    task_id = models.CharField(max_length=255, unique=True)
    voice_model = models.ForeignKey(
        VoiceModel, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True
    )
    status = models.CharField(
        max_length=10,
        choices=VoiceModel.TrainingStatus.choices,
        default=VoiceModel.TrainingStatus.PENDING
    )
    progress = models.FloatField(default=0)
    error_message = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['task_id']),
            models.Index(fields=['user', 'status']),
        ]
    
    def __str__(self):
        return f"Task {self.task_id} - {self.status}"

class AudioConversion(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='audio_conversions')
    input_text = models.TextField(null=True, blank=True)
    input_audio = models.FileField(upload_to='conversions/input/%Y/%m/%d/', null=True, blank=True)
    output_audio = models.FileField(upload_to='conversions/output/%Y/%m/%d/')
    model_used = models.ForeignKey(VoiceModel, on_delete=models.SET_NULL, null=True, blank=True)
    processing_time = models.FloatField()  # in seconds
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Conversion {self.id} - {self.user.username}"