from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.core.files.storage import default_storage
from django.shortcuts import get_object_or_404
from .models import VoiceSample, VoiceModel, TrainingTask
from .tasks import train_voice_model_task,validate_voice_sample_task
from .voice_utils import get_voice_trainer
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from rest_framework import generics
from django.contrib.auth.models import User
from .serializers import UserSerializer, VoiceSampleSerializer, VoiceModelSerializer
from django.db import transaction,models
import os


trainer = get_voice_trainer()

class UserDetailView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        serializer = UserSerializer(request.user)
        return Response(serializer.data)

class RegisterView(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer

class CustomTokenObtainPairView(TokenObtainPairView):
    pass

class CustomTokenRefreshView(TokenRefreshView):
    pass

class VoiceSampleListView(APIView):
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        samples = VoiceSample.objects.filter(user=request.user).order_by('-created_at')
        serializer = VoiceSampleSerializer(samples, many=True)
        return Response(serializer.data)

class UploadVoiceSampleView(APIView):
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        # Get all files with the 'audio' key
        audio_files = request.FILES.getlist('audio')
        
        if not audio_files:
            return Response({'error': 'No audio files provided'}, status=status.HTTP_400_BAD_REQUEST)
        
        created_samples = []
        errors = []
        
        try:
            with transaction.atomic():
                for audio_file in audio_files:
                    try:
                        # Basic validation
                        if audio_file.size > 50 * 1024 * 1024:  # 50MB limit
                            errors.append(f'{audio_file.name}: File too large (max 50MB)')
                            continue
                        
                        # Check file extension
                        allowed_extensions = ['.wav', '.mp3', '.m4a', '.ogg', '.flac']
                        if not any(audio_file.name.lower().endswith(ext) for ext in allowed_extensions):
                            errors.append(f'{audio_file.name}: Unsupported file format')
                            continue
                            
                        # Create sample with UPLOADED status
                        voice_sample = VoiceSample(
                            user=request.user,
                            audio_file=audio_file,
                            file_size=audio_file.size,
                            status=VoiceSample.SampleStatus.UPLOADED
                        )
                        voice_sample.save()
                        created_samples.append(voice_sample)
                        
                        # Trigger validation task
                        print("Triggered validation process..")
                        validate_voice_sample_task.delay(voice_sample.id)
                        
                    except Exception as e:
                        errors.append(f'{audio_file.name}: {str(e)}')
                        continue
            
            # Serialize the created samples
            serializer = VoiceSampleSerializer(created_samples, many=True, context={'request': request})
            
            response_data = {
                'samples': serializer.data,
                'created_count': len(created_samples),
                'total_count': len(audio_files),
                'message': f'Successfully uploaded {len(created_samples)} files. Validation in progress.'
            }
            
            if errors:
                response_data['errors'] = errors
                
            return Response(response_data, status=status.HTTP_201_CREATED)
                     
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
   
class VoiceSampleDeleteView(APIView):
    permission_classes = [IsAuthenticated]
    
    def delete(self, request, sample_id):
        sample = get_object_or_404(VoiceSample, id=sample_id, user=request.user)
        sample.audio_file.delete()  # Delete the file from storage
        sample.delete()  # Delete the record
        return Response(status=status.HTTP_204_NO_CONTENT)


class VoiceModelListView(APIView):
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        models = VoiceModel.objects.filter(user=request.user).order_by('-created_at')
        serializer = VoiceModelSerializer(models, many=True)
        return Response(serializer.data)
    
class VoiceSampleStatsView(APIView):
    """New endpoint to get sample statistics"""
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        samples = VoiceSample.objects.filter(user=request.user)
        
        stats = {
            'total_samples': samples.count(),
            'valid_samples': samples.filter(status=VoiceSample.SampleStatus.VALID).count(),
            'processing_samples': samples.filter(status=VoiceSample.SampleStatus.PROCESSING).count(),
            'invalid_samples': samples.filter(status=VoiceSample.SampleStatus.INVALID).count(),
            'uploaded_samples': samples.filter(status=VoiceSample.SampleStatus.UPLOADED).count(),
            'can_train': samples.filter(status=VoiceSample.SampleStatus.VALID).count() >= 5,
            'total_duration': sum(s.duration or 0 for s in samples if s.duration),
            'average_quality': samples.filter(
                audio_quality__isnull=False
            ).aggregate(avg_quality=models.Avg('audio_quality'))['avg_quality'] or 0
        }
        
        return Response(stats)
class GenerateVoiceModelView(APIView):
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        valid_samples = VoiceSample.objects.filter(
            user=request.user,
            status=VoiceSample.SampleStatus.VALID
        ).count()
        
        if valid_samples < 3:  # Match your TrainingConfig.min_samples
            return Response(
                {'error': f'Need at least 3 valid samples (have {valid_samples})'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Check if there's already a pending/running task for this user
        existing_task = TrainingTask.objects.filter(
            user=request.user,
            status__in=[VoiceModel.TrainingStatus.PENDING, VoiceModel.TrainingStatus.TRAINING]
        ).first()
        
        if existing_task:
            return Response(
                {
                    'error': 'Training already in progress',
                    'task_id': existing_task.task_id
                },
                status=status.HTTP_409_CONFLICT
            )
        
        # Start training task - let the task function handle TrainingTask creation
        task = train_voice_model_task.delay(request.user.id)
        
        return Response(
            {'task_id': task.id},
            status=status.HTTP_202_ACCEPTED
        )

class TrainingStatusView(APIView):
    permission_classes = [IsAuthenticated]
    
    def get(self, request, task_id):
        task = get_object_or_404(
            TrainingTask,
            task_id=task_id,
            user=request.user
        )
        
        response_data = {
            'status': task.status,
            'progress': task.progress,
            'created_at': task.created_at,
            'updated_at': task.updated_at
        }
        
        if task.voice_model:
            response_data['model_id'] = task.voice_model.id
        
        return Response(response_data)

class TextToSpeechView(APIView):
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        text = request.data.get('text')
        if not text:
            return Response(
                {'error': 'No text provided'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        voice_model = get_object_or_404(
            VoiceModel,
            user=request.user,
            is_active=True
        )
        
        try:
            audio_path = trainer.generate_audio(text, voice_model.model_path)
            return Response({'audio_url': request.build_absolute_uri(audio_path)})
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class SpeechToSpeechView(APIView):
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        audio_file = request.FILES.get('audio')
        if not audio_file:
            return Response(
                {'error': 'No audio file provided'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Save temporary file
            temp_path = f"temp/{audio_file.name}"
            with default_storage.open(temp_path, 'wb+') as destination:
                for chunk in audio_file.chunks():
                    destination.write(chunk)
            
            # Transcribe
            text = trainer._transcribe_audio(default_storage.path(temp_path))
            
            # Get active model
            voice_model = get_object_or_404(
                VoiceModel,
                user=request.user,
                is_active=True
            )
            
            # Generate audio
            output_path = trainer.generate_audio(text, voice_model.model_path)
            
            return Response({
                'text': text,
                'audio_url': request.build_absolute_uri(output_path)
            })
            
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        finally:
            if temp_path:
                default_storage.delete(temp_path)