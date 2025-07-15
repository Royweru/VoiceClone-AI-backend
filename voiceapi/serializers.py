from rest_framework import serializers
from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from .models import VoiceSample, VoiceModel, TrainingTask

class UserSerializer(serializers.ModelSerializer):
    password = serializers.CharField(
        write_only=True,
        required=True,
        validators=[validate_password]
    )
    password2 = serializers.CharField(write_only=True, required=True)

    class Meta:
        model = User
        fields = ('id', 'username', 'password', 'password2', 'email')
        extra_kwargs = {
            'email': {'required': False}
        }

    def validate(self, attrs):
        if attrs['password'] != attrs['password2']:
            raise serializers.ValidationError(
                {"password": "Password fields didn't match."})
        return attrs

    def create(self, validated_data):
        user = User.objects.create(
            username=validated_data['username'],
            email=validated_data.get('email', '')
        )
        user.set_password(validated_data['password'])
        user.save()
        return user

class VoiceSampleSerializer(serializers.ModelSerializer):
    audio_url = serializers.SerializerMethodField()
    formatted_duration = serializers.SerializerMethodField()
    formatted_size = serializers.SerializerMethodField()

    class Meta:
        model = VoiceSample
        fields = [
            'id', 'audio_file', 'audio_url', 'duration', 'formatted_duration',
            'file_size', 'formatted_size', 'status', 'created_at'
        ]
        read_only_fields = [
            'id', 'duration', 'file_size', 'status', 'created_at'
        ]

    def get_audio_url(self, obj):
        request = self.context.get('request')
        if obj.audio_file and request:
            return request.build_absolute_uri(obj.audio_file.url)
        return None

    def get_formatted_duration(self, obj):
        if obj.duration:
            mins = int(obj.duration // 60)
            secs = int(obj.duration % 60)
            return f"{mins}:{secs:02d}"
        return None

    def get_formatted_size(self, obj):
        if obj.file_size:
            for unit in ['B', 'KB', 'MB', 'GB']:
                if obj.file_size < 1024:
                    return f"{obj.file_size:.1f} {unit}"
                obj.file_size /= 1024
        return None

class VoiceModelSerializer(serializers.ModelSerializer):
    formatted_training_time = serializers.SerializerMethodField()
    formatted_created_at = serializers.SerializerMethodField()

    class Meta:
        model = VoiceModel
        fields = [
            'id', 'name', 'is_active', 'status', 'progress',
            'sample_count', 'formatted_training_time',
            'formatted_created_at', 'created_at'
        ]
        read_only_fields = fields

    def get_formatted_training_time(self, obj):
        if obj.training_time:
            hours = int(obj.training_time // 3600)
            mins = int((obj.training_time % 3600) // 60)
            return f"{hours}h {mins}m"
        return None

    def get_formatted_created_at(self, obj):
        return obj.created_at.strftime('%Y-%m-%d %H:%M')

class TrainingTaskSerializer(serializers.ModelSerializer):
    model = VoiceModelSerializer(source='voice_model', read_only=True)

    class Meta:
        model = TrainingTask
        fields = [
            'task_id', 'status', 'progress', 'error_message',
            'created_at', 'updated_at', 'model'
        ]
        read_only_fields = fields