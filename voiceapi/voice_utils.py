import os
import torch
import librosa
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from TTS.api import TTS
from TTS.config import load_config
from TTS.utils.manage import ModelManager
import whisper
import logging
from datetime import datetime
from django.core.files.storage import default_storage
from .models import VoiceSample
import tempfile
import shutil
import json
import subprocess
from pydub import AudioSegment
import soundfile as sf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    min_audio_length: float = 1.0  # seconds
    max_audio_length: float = 20.0  # seconds
    target_sample_rate: int = 22050
    min_samples: int = 3
    training_method: str = "xtts" 
    max_samples: int = 50
    validation_split: float = 0.1
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-4
    save_step: int = 1000
    eval_step: int = 500

class VoiceTrainer:
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Fix PyTorch weights_only loading issue
        self._setup_torch_globals()
        
        self.whisper_model = whisper.load_model("base")
        # Initialize TTS model for inference only
        self.tts = None

    def _setup_torch_globals(self):
        """Setup PyTorch global settings to handle XTTS model loading"""
        try:
            # Add safe globals for XTTS model loading
            import torch.serialization
            from TTS.tts.configs.xtts_config import XttsConfig
            
            # Add XTTS config to safe globals
            torch.serialization.add_safe_globals([
                XttsConfig,
                # Add other TTS related classes that might be needed
            ])
            
            logger.info("PyTorch safe globals configured for TTS models")
            
        except ImportError as e:
            logger.warning(f"Could not import XttsConfig for safe globals: {e}")
        except Exception as e:
            logger.warning(f"Could not setup torch safe globals: {e}")

    def _initialize_tts(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"):
        """Initialize TTS model with robust error handling"""
        try:
            logger.info(f"Initializing TTS model: {model_name}")
            
            # Method 1: Try standard initialization
            try:
                tts = TTS(model_name=model_name, progress_bar=False).to(self.device)
                logger.info(f"TTS model initialized successfully on {self.device}")
                return tts
            except Exception as e1:
                logger.warning(f"Standard TTS initialization failed: {e1}")
                
                # Method 2: Try with CPU fallback
                try:
                    logger.info("Trying CPU fallback...")
                    tts = TTS(model_name=model_name, progress_bar=False).to("cpu")
                    logger.info("TTS model initialized on CPU")
                    return tts
                except Exception as e2:
                    logger.warning(f"CPU fallback failed: {e2}")
                    
                    # Method 3: Try handling weights_only issue
                    try:
                        logger.info("Attempting to handle PyTorch security restrictions...")
                        
                        # Store original torch.load
                        original_load = torch.load
                        
                        def patched_load(*args, **kwargs):
                            # Force weights_only=False for TTS loading
                            if 'weights_only' not in kwargs:
                                kwargs['weights_only'] = False
                            return original_load(*args, **kwargs)
                        
                        # Temporarily patch torch.load
                        torch.load = patched_load
                        
                        try:
                            tts = TTS(model_name=model_name, progress_bar=False).to(self.device)
                            logger.info("TTS model initialized with patched loader")
                            return tts
                        finally:
                            # Restore original torch.load
                            torch.load = original_load
                            
                    except Exception as e3:
                        logger.error(f"All TTS initialization methods failed: {e3}")
                        raise Exception(f"Failed to initialize TTS model: {e3}")
                        
        except Exception as e:
            logger.error(f"TTS initialization completely failed: {e}")
            raise

    def process_sample(self, sample_id: int) -> Dict:
        """Process a single voice sample with improved transcription"""
        sample = VoiceSample.objects.get(id=sample_id)
        try:
            # Get the file path
            if hasattr(sample.audio_file, 'path'):
                path = sample.audio_file.path
            else:
                path = default_storage.path(sample.audio_file.name)
            
            logger.info(f"Processing sample {sample_id} at path: {path}")
            
            # Verify file exists
            if not os.path.exists(path):
                logger.error(f"File does not exist at path: {path}")
                sample.status = VoiceSample.SampleStatus.INVALID
                sample.save()
                return {'valid': False, 'reason': 'File not found'}
            
            # Check file size
            file_size = os.path.getsize(path)
            if file_size == 0:
                logger.error(f"File is empty: {path}")
                sample.status = VoiceSample.SampleStatus.INVALID
                sample.save()
                return {'valid': False, 'reason': 'Empty file'}
            
            logger.info(f"File size: {file_size} bytes")
            
            # Load audio with librosa (your existing improved loading code)
            # Try multiple loading approaches
            waveform = None
            sample_rate = None
        
            # Method 1: Try librosa with different parameters
            try:
                logger.info("Attempting to load with librosa...")
                waveform, sample_rate = librosa.load(path, sr=None, mono=True)
                logger.info(f"Librosa loading successful")
            except Exception as e:
                logger.warning(f"Librosa failed: {str(e)}")
                
                # Method 2: Try with ffmpeg backend explicitly
                try:
                    logger.info("Attempting to load with librosa + ffmpeg...")
                    waveform, sample_rate = librosa.load(path, sr=None, mono=True, 
                                                    offset=0.0, duration=None)
                except Exception as e2:
                    logger.warning(f"Librosa + ffmpeg failed: {str(e2)}")
                    
                    # Method 3: Try converting path to short path (Windows)
                    try:
                        if os.name == 'nt':  # Windows
                            import win32api
                            short_path = win32api.GetShortPathName(path)
                            logger.info(f"Trying short path: {short_path}")
                            waveform, sample_rate = librosa.load(short_path, sr=None, mono=True)
                    except Exception as e3:
                        logger.error(f"All loading methods failed. Last error: {str(e3)}")
                        sample.status = VoiceSample.SampleStatus.INVALID
                        sample.save()
                        return {'': False, 'reason': f'Audio loading failed: {str(e)}'}
            
            if waveform is None:
                sample.status = VoiceSample.SampleStatus.INVALID
                sample.save()
                return {'valid': False, 'reason': 'Failed to load audio data'}
            
            duration = len(waveform) / sample_rate
            
            # Update sample metadata
            sample.duration = duration
            sample.sample_rate = sample_rate
            sample.channels = 1
            
            logger.info(f"Audio loaded - Duration: {duration}s, Sample Rate: {sample_rate}Hz")
            
            # Validate duration
            if duration < self.config.min_audio_length:
                sample.status = VoiceSample.SampleStatus.INVALID
                sample.save()
                return {'valid': False, 'reason': f'Too short ({duration:.2f}s < {self.config.min_audio_length}s)'}
            
            if duration > self.config.max_audio_length:
                sample.status = VoiceSample.SampleStatus.INVALID
                sample.save()
                return {'valid': False, 'reason': f'Too long ({duration:.2f}s > {self.config.max_audio_length}s)'}
            
            # Calculate RMS for audio quality
            rms = np.sqrt(np.mean(waveform**2))
            sample.audio_quality = float(rms)
            
            logger.info(f"Audio RMS: {rms}")
            
            if rms < 0.01:  # Very quiet audio
                sample.status = VoiceSample.SampleStatus.INVALID
                sample.save()
                return {'valid': False, 'reason': f'Too quiet (RMS: {rms:.4f})'}
            
            # IMPROVED TRANSCRIPTION APPROACH
            transcription = ""
            try:
                logger.info(f"Starting transcription for sample {sample_id}")
                
                # Method 1: Try transcribing from the loaded waveform directly
                try:
                    # Whisper expects 16kHz sample rate
                    if sample_rate != 16000:
                        waveform_16k = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
                    else:
                        waveform_16k = waveform
                    
                    # Transcribe from numpy array
                    transcription_result = self.whisper_model.transcribe(waveform_16k)
                    transcription = transcription_result["text"].strip()
                    logger.info(f"Transcription from waveform successful: {transcription}")
                    
                except Exception as e1:
                    logger.warning(f"Transcription from waveform failed: {str(e1)}")
                    
                    # Method 2: Create a temporary WAV file in a simple path
                    try:
                        logger.info("Attempting transcription via temporary file")
                        
                        # Create a temporary file with a simple name
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                            temp_path = temp_file.name
                        
                        logger.info(f"Created temporary file: {temp_path}")
                        
                        # Save waveform as WAV using soundfile (more reliable than librosa.output.write_wav)
                        import soundfile as sf
                        sf.write(temp_path, waveform, sample_rate)
                        
                        # Transcribe the temporary file
                        transcription_result = self.whisper_model.transcribe(temp_path)
                        transcription = transcription_result["text"].strip()
                        
                        # Clean up temporary file
                        try:
                            os.unlink(temp_path)
                        except:
                            pass
                        
                        logger.info(f"Transcription from temp file successful: {transcription}")
                        
                    except Exception as e2:
                        logger.warning(f"Transcription from temp file failed: {str(e2)}")
                        
                        # Method 3: Copy original file to temp location with simple name
                        try:
                            logger.info("Attempting transcription via copied file")
                            
                            # Create temp file with original extension
                            file_ext = os.path.splitext(path)[1]
                            with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as temp_file:
                                temp_path = temp_file.name
                            
                            # Copy the original file
                            shutil.copy2(path, temp_path)
                            logger.info(f"Copied file to: {temp_path}")
                            
                            # Transcribe
                            transcription_result = self.whisper_model.transcribe(temp_path)
                            transcription = transcription_result["text"].strip()
                            
                            # Clean up
                            try:
                                os.unlink(temp_path)
                            except:
                                pass
                            
                            logger.info(f"Transcription from copied file successful: {transcription}")
                            
                        except Exception as e3:
                            logger.error(f"All transcription methods failed. Last error: {str(e3)}")
                            # Don't fail the entire process, just skip transcription
                            transcription = "[Transcription failed]"
            
            except Exception as e:
                logger.error(f"Transcription process error: {str(e)}")
                transcription = "[Transcription failed]"
            
            # Store transcription (even if failed)
            sample.transcription = transcription
            
            # Don't invalidate sample if transcription fails, but log it
            if transcription == "[Transcription failed]":
                logger.warning(f"Sample {sample_id} processed without transcription")
            elif len(transcription) < 5:  # Very short transcription
                logger.warning(f"Sample {sample_id} has very short transcription: '{transcription}'")
            
            # Mark as valid regardless of transcription success
            sample.status = VoiceSample.SampleStatus.VALID
            sample.processed_at = datetime.now()
            sample.save()
            
            logger.info(f"Sample {sample_id} processed successfully")
            
            return {
                'valid': True,
                'duration': duration,
                'transcription': transcription,
                'sample_rate': sample_rate,
                'rms': rms
            }
        
        except Exception as e:
            logger.error(f"Error processing sample {sample_id}: {str(e)}")
            sample.status = VoiceSample.SampleStatus.INVALID
            sample.save()
            return {'valid': False, 'reason': str(e)}

    def train_voice_model(self, user_id: int) -> Dict:
        """Main training function - chooses method based on config"""
        training_method = self.config.training_method
        
        if training_method == 'xtts' or training_method == 'coqui':
            return self.train_voice_model_coqui(user_id)
        else:
            raise ValueError(f"Unknown training method: {training_method}")

    def train_voice_model_coqui(self, user_id: int) -> Dict:
        """Train a voice model using Coqui TTS"""
        try:
            valid_samples = VoiceSample.objects.filter(
                user_id=user_id,
                status=VoiceSample.SampleStatus.VALID
            )
            
            if valid_samples.count() < self.config.min_samples:
                raise ValueError(f"Need at least {self.config.min_samples} valid samples")
            
            # Prepare training data
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_dir = Path("media") / "voice_models" / f"user_{user_id}" / timestamp
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Create dataset structure
            dataset_dir = model_dir / "dataset"
            dataset_dir.mkdir(exist_ok=True)
            
            # Copy and prepare audio files
            sample_paths = []
            metadata_lines = []
            
            for i, sample in enumerate(valid_samples):
                # Copy audio file to dataset directory
                audio_filename = f"audio_{i:04d}.wav"
                audio_path = dataset_dir / audio_filename
                logger.info(f"Processing sample {sample.id}: {sample.audio_file.path}")
            
                # Convert to WAV if needed and ensure proper format
                self._prepare_audio_file(sample.audio_file.path, str(audio_path))
                
                sample_paths.append(str(audio_path))
                # Use transcription if available, otherwise use a default
                transcript_text = sample.transcription if sample.transcription and sample.transcription != "[Transcription failed]" else "sample audio"
                metadata_lines.append(f"{audio_filename}|{transcript_text}|user_{user_id}")
            
            # Create metadata file
            metadata_path = dataset_dir / "metadata.csv"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(metadata_lines))
            
            # XTTS Fine-tuning approach
            return self._train_xtts_model(user_id, model_dir, sample_paths)
                
        except Exception as e:
            logger.error(f"Coqui training failed for user {user_id}: {str(e)}")
            raise
        
    def _train_xtts_model(self, user_id: int, model_dir: Path, sample_paths: List[str]) -> Dict:
        """Train XTTS model for voice cloning with improved error handling"""
        try:
            # Initialize XTTS model
            model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
            tts = self._initialize_tts(model_name)
            
            # Verify all sample paths exist
            valid_samples = []
            for sample_path in sample_paths:
                if os.path.exists(sample_path) and os.path.getsize(sample_path) > 0:
                    valid_samples.append(sample_path)
                    logger.info(f"Validated sample: {sample_path}")
                else:
                    logger.warning(f"Invalid sample path: {sample_path}")
            
            if not valid_samples:
                raise ValueError("No valid sample paths found")
            
            # Test voice cloning with the first sample
            test_text = "Hello, this is a test of the voice cloning system."
            test_output_path = model_dir / "test_output.wav"
            
            # Generate test audio with multiple approaches
            test_success = False
            
            try:
                # Method 1: Try tts_to_file if available
                if hasattr(tts, 'tts_to_file'):
                    logger.info("Using tts_to_file method")
                    tts.tts_to_file(
                        text=test_text,
                        speaker_wav=valid_samples[0],
                        file_path=str(test_output_path),
                        language="en"
                    )
                    test_success = True
                    logger.info("Test audio generated with tts_to_file")
                else:
                    logger.info("tts_to_file not available, trying alternative methods")
                
            except Exception as e1:
                logger.warning(f"tts_to_file failed: {e1}")
            
            if not test_success:
                try:
                    # Method 2: Try direct tts method
                    logger.info("Using direct tts method")
                    wav = tts.tts(
                        text=test_text,
                        speaker_wav=valid_samples[0],
                        language="en"
                    )
                    
                    # Save the audio
                    if isinstance(wav, np.ndarray):
                        # Get sample rate from model if available
                        sample_rate = getattr(tts.synthesizer, 'output_sample_rate', 22050) if hasattr(tts, 'synthesizer') else 22050
                        sf.write(str(test_output_path), wav, sample_rate)
                        test_success = True
                        logger.info("Test audio generated with direct tts method")
                    else:
                        logger.warning("TTS returned unexpected format")
                        
                except Exception as e2:
                    logger.warning(f"Direct tts method failed: {e2}")
            
            if not test_success:
                try:
                    # Method 3: Try tts_with_vc if available (some versions have this)
                    if hasattr(tts, 'tts_with_vc'):
                        logger.info("Using tts_with_vc method: ")
                        tts.tts_with_vc(
                            text=test_text,
                            speaker_wav=valid_samples[0],
                            file_path=str(test_output_path),
                            language="en"
                        )
                        test_success = True
                        logger.info("Test audio generated with tts_with_vc")
                        
                except Exception as e3:
                    logger.warning(f"tts_with_vc failed: {e3}")
            
            # Don't fail if test generation doesn't work - the model might still be usable
            if not test_success:
                logger.warning("Test audio generation failed, but continuing with model setup")
                test_output_path = model_dir / "test_failed.txt"
                with open(test_output_path, 'w') as f:
                    f.write("Test audio generation failed but model is configured")
            
            # Save model configuration
            config_data = {
                'user_id': user_id,
                'model_type': 'xtts_v2',
                'reference_samples': valid_samples,
                'created_at': datetime.now().isoformat(),
                'language': 'en',
                'test_output': str(test_output_path),
                'sample_rate': 22050,
                'test_success': test_success
            }
            
            config_path = model_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Voice model configured for user {user_id} with {len(valid_samples)} samples")
            
            return {
                'success': True,
                'model_path': str(model_dir),
                'model_type': 'xtts',
                'sample_count': len(valid_samples),
                'config_path': str(config_path),
                'test_output': str(test_output_path),
                'test_success': test_success
            }
            
        except Exception as e:
            logger.error(f"XTTS training failed: {str(e)}")
            raise

    def _prepare_audio_file(self, input_path: str, output_path: str):
        """Convert and prepare audio file for training"""
        try: 
            logger.info(f"Preparing audio file: {input_path} -> {output_path}")
            
            # Handle different path formats
            actual_input_path = None
            temp_file_created = False
            
            # Method 1: Try the path as-is first
            if os.path.exists(input_path):
                actual_input_path = input_path
                logger.info(f"Using direct path: {actual_input_path}")
            else:
                # Method 2: Try with default_storage.path()
                try:
                    if hasattr(default_storage, 'path'):
                        storage_path = default_storage.path(input_path)
                        if os.path.exists(storage_path):
                            actual_input_path = storage_path
                            logger.info(f"Using storage path: {actual_input_path}")
                    else:
                        logger.warning("Storage doesn't have path method")
                except Exception as e:
                    logger.warning(f"Storage path failed: {str(e)}")
                
                # Method 3: Try with media root prefix
                if not actual_input_path:
                    try:
                        from django.conf import settings
                        if hasattr(settings, 'MEDIA_ROOT'):
                            media_path = os.path.join(settings.MEDIA_ROOT, input_path)
                            if os.path.exists(media_path):
                                actual_input_path = media_path
                                logger.info(f"Using media root path: {actual_input_path}")
                    except Exception as e:
                        logger.warning(f"Media root path failed: {str(e)}")
                
                # Method 4: Try to open with storage and copy to temp file
                if not actual_input_path:
                    try:
                        logger.info("Attempting to read file through storage API")
                        with default_storage.open(input_path, 'rb') as source_file:
                            # Create a temporary file with proper extension
                            file_ext = os.path.splitext(input_path)[1] or '.wav'
                            temp_fd, temp_path = tempfile.mkstemp(suffix=file_ext)
                            
                            try:
                                # Copy content to temp file
                                with os.fdopen(temp_fd, 'wb') as temp_dest:
                                    # Handle both file-like objects and iterables
                                    if hasattr(source_file, 'read'):
                                        shutil.copyfileobj(source_file, temp_dest)
                                    else:
                                        # For Django files that have chunks()
                                        for chunk in source_file.chunks():
                                            temp_dest.write(chunk)
                                
                                actual_input_path = temp_path
                                temp_file_created = True
                                logger.info(f"Created temporary file: {actual_input_path}")
                            except Exception as e:
                                # Clean up temp file if creation failed
                                try:
                                    os.close(temp_fd)
                                    os.unlink(temp_path)
                                except:
                                    pass
                                raise e
                                
                    except Exception as e:
                        logger.error(f"Storage API read failed: {str(e)}")
            
            if not actual_input_path:
                raise FileNotFoundError(f"Could not locate audio file: {input_path}")
            
            # Verify the file exists and is readable
            if not os.path.exists(actual_input_path):
                raise FileNotFoundError(f"File does not exist: {actual_input_path}")
            
            file_size = os.path.getsize(actual_input_path)
            if file_size == 0:
                raise ValueError(f"File is empty: {actual_input_path}")
            
            logger.info(f"File verified - Size: {file_size} bytes")
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Load and convert audio file using multiple approaches
            audio = None
            
            # Method 1: Try with AudioSegment auto-detection
            try:
                audio = AudioSegment.from_file(actual_input_path)
                logger.info(f"Audio loaded successfully - Duration: {len(audio)}ms, Channels: {audio.channels}")
            except Exception as e1:
                logger.warning(f"AudioSegment auto-detection failed: {str(e1)}")
                
                # Method 2: Try with explicit format detection
                try:
                    file_ext = os.path.splitext(actual_input_path)[1].lower()
                    format_map = {
                        '.mp3': 'mp3', 
                        '.wav': 'wav', 
                        '.m4a': 'mp4',  # pydub uses 'mp4' for m4a files
                        '.mp4': 'mp4',
                        '.ogg': 'ogg', 
                        '.flac': 'flac',
                        '.aac': 'aac',
                        '.wma': 'wma'
                    }
                    audio_format = format_map.get(file_ext, 'wav')
                    
                    logger.info(f"Trying with explicit format: {audio_format} for extension: {file_ext}")
                    audio = AudioSegment.from_file(actual_input_path, format=audio_format)
                    logger.info(f"Audio loaded with explicit format - Duration: {len(audio)}ms")
                except Exception as e2:
                    logger.warning(f"Explicit format failed: {str(e2)}")
                    
                    # Method 3: Try converting with ffmpeg first
                    try:
                        logger.info("Attempting ffmpeg conversion to WAV")
                        temp_wav_fd, temp_wav_path = tempfile.mkstemp(suffix='.wav')
                        os.close(temp_wav_fd)
                        
                        # Use ffmpeg to convert to WAV
                        import subprocess
                        ffmpeg_cmd = [
                            'ffmpeg', '-i', actual_input_path, 
                            '-acodec', 'pcm_s16le', 
                            '-ar', str(self.config.target_sample_rate),
                            '-ac', '1',  # mono
                            '-y',  # overwrite output
                            temp_wav_path
                        ]
                        
                        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                        if result.returncode == 0:
                            audio = AudioSegment.from_wav(temp_wav_path)
                            logger.info(f"Audio converted with ffmpeg - Duration: {len(audio)}ms")
                        else:
                            logger.error(f"ffmpeg failed: {result.stderr}")
                            raise Exception(f"ffmpeg conversion failed: {result.stderr}")
                        
                        # Clean up temp WAV file
                        try:
                            os.unlink(temp_wav_path)
                        except:
                            pass
                            
                    except Exception as e3:
                        logger.error(f"All audio loading methods failed. Last error: {str(e3)}")
                        raise Exception(f"Could not load audio file: {actual_input_path}. Errors: {str(e1)}, {str(e2)}, {str(e3)}")
            
            if audio is None:
                raise Exception("Failed to load audio with all methods")
            
            # Convert to required format
            audio = audio.set_frame_rate(self.config.target_sample_rate)
            audio = audio.set_channels(1)  # Mono
            audio = audio.set_sample_width(2)  # 16-bit
            
            # Export as WAV
            audio.export(output_path, format="wav")
            logger.info(f"Audio prepared successfully: {output_path}")
            
            # Clean up temporary file if we created one
            if temp_file_created and os.path.exists(actual_input_path):
                try:
                    os.unlink(actual_input_path)
                    logger.info(f"Cleaned up temporary file: {actual_input_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file: {str(e)}")
            
        except Exception as e:
            logger.error(f"Audio preparation failed for {input_path}: {str(e)}")
            logger.error(f"Current working directory: {os.getcwd()}")
            
            # Additional debugging info
            try:
                logger.error(f"Input path type: {type(input_path)}")
                if isinstance(input_path, str):
                    logger.error(f"Input path exists: {os.path.exists(input_path)}")
                    logger.error(f"Input path absolute: {os.path.abspath(input_path)}")
            except:
                pass
            
            raise

    def synthesize_with_trained_model(self, user_id: int, text: str, model_path: str) -> str:
        """Synthesize speech using a trained voice model"""
        try:
            config_path = Path(model_path) / "config.json"
            
            if not config_path.exists():
                raise ValueError("Model configuration not found")
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            model_type = config.get('model_type')
            
            if model_type == 'xtts_v2':
                return self._synthesize_xtts(text, config, model_path)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
        except Exception as e:
            logger.error(f"Synthesis failed: {str(e)}")
            raise


    def _synthesize_xtts(self, text: str, config: Dict, model_path: str) -> str:
        """Synthesize using XTTS model with robust API handling"""
        try:
            # Initialize TTS model
            tts = self._initialize_tts("tts_models/multilingual/multi-dataset/xtts_v2")
            
            # Use reference sample for voice cloning
            reference_samples = config.get('reference_samples', [])
            if not reference_samples:
                raise ValueError("No reference samples found in model config")
            
            # Verify reference sample exists
            reference_wav = reference_samples[0]
            if not os.path.exists(reference_wav):
                raise FileNotFoundError(f"Reference sample not found: {reference_wav}")
            
            # Create output path
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = Path(model_path) / f"synthesis_{timestamp}.wav"
            
            # Generate speech with multiple approaches
            synthesis_success = False
            
            try:
                # Method 1: Try tts_to_file
                if hasattr(tts, 'tts_to_file'):
                    logger.info("Using tts_to_file for synthesis")
                    tts.tts_to_file(
                        text=text,
                        speaker_wav=reference_wav,
                        file_path=str(output_path),
                        language="en"
                    )
                    synthesis_success = True
                    
            except Exception as e1:
                logger.warning(f"tts_to_file synthesis failed: {e1}")
            
            if not synthesis_success:
                try:
                    # Method 2: Direct tts method
                    logger.info("Using direct tts method for synthesis")
                    wav = tts.tts(
                        text=text,
                        speaker_wav=reference_wav,
                        language="en"
                    )
                    
                    if isinstance(wav, np.ndarray):
                        sample_rate = config.get('sample_rate', 22050)
                        sf.write(str(output_path), wav, sample_rate)
                        synthesis_success = True
                    
                except Exception as e2:
                    logger.warning(f"Direct tts synthesis failed: {e2}")
            
            if not synthesis_success:
                try:
                    # Method 3: Try tts_with_vc
                    if hasattr(tts, 'tts_with_vc'):
                        logger.info("Using tts_with_vc for synthesis")
                        tts.tts_with_vc(
                            text=text,
                            speaker_wav=reference_wav,
                            file_path=str(output_path),
                            language="en"
                        )
                        synthesis_success = True
                        
                except Exception as e3:
                    logger.warning(f"tts_with_vc synthesis failed: {e3}")
            
            if not synthesis_success:
                raise Exception("All synthesis methods failed")
            
            logger.info(f"Speech synthesized successfully: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"XTTS synthesis failed: {str(e)}")
            raise


    def generate_audio(self, text: str, model_dir: str = None, user_id: int = None) -> str:
        """Generate audio from text using the specified model or default TTS"""
        try:
            output_dir = Path("media") / "outputs"
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = output_dir / f"output_{timestamp}.wav"
            
            if model_dir and user_id:
                # Use trained model
                return self.synthesize_with_trained_model(user_id, text, model_dir)
            else:
                # Use default TTS
                tts = self._initialize_tts()
                tts.tts_to_file(
                    text=text,
                    file_path=str(output_path),
                    language="en"
                )
                
                return str(output_path)
                
        except Exception as e:
            logger.error(f"Audio generation failed: {str(e)}")
            raise

    def get_user_models(self, user_id: int) -> List[Dict]:
        """ Get all trained models for a user """
        try:
            models_dir = Path("media") / "voice_models" / f"user_{user_id}"
            if not models_dir.exists():
                return []
            
            models = []
            for model_dir in models_dir.iterdir():
                if model_dir.is_dir():
                    config_path = model_dir / "config.json"
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                        
                        models.append({
                            'path': str(model_dir),
                            'created_at': config.get('created_at'),
                            'model_type': config.get('model_type'),
                            'sample_count': len(config.get('reference_samples', []))
                        })
            
            # Sort by creation date (newest first)
            models.sort(key=lambda x: x['created_at'], reverse=True)
            return models
            
        except Exception as e:
            logger.error(f"Error getting user models: {str(e)}")
            return []

def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio to text using Whisper"""
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise

# Helper function to create trainer instance
def get_voice_trainer(config: TrainingConfig = None) -> VoiceTrainer:
    """Get a configured voice trainer instance"""
    return VoiceTrainer(config or TrainingConfig())