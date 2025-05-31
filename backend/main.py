from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import json
import logging
import tempfile
import os
import requests
from google.cloud import speech
import io
from google.oauth2 import service_account
import numpy as np
from scipy.io import wavfile
from typing import Dict, Any, List, Optional, Union
import uvicorn
import librosa
import soundfile as sf
from pathlib import Path
import gc
import warnings
import whisper
from transformers import pipeline
import google.generativeai as genai
import time
import traceback
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Keys
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Validate API keys
if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY environment variable not set - using fallback analysis")
else:
    logger.info("Google API key configured")

if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY environment variable not set")
    raise ValueError("GROQ_API_KEY environment variable is required")
else:
    logger.info("Groq API key configured")

# Global variables for models
whisper_model = None
emotion_analyzer = None
sentiment_analyzer = None
text_generator = None
groq_client = None
gemini_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global whisper_model, emotion_analyzer, sentiment_analyzer, groq_client, gemini_model
    
    logger.info("Starting up FastAPI application...")
    
    # Initialize Groq client
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("Successfully initialized Groq client")
    except Exception as e:
        logger.error(f"Error initializing Groq client: {str(e)}")
        raise

    # Initialize models with better error handling
    try:
        whisper_model = whisper.load_model("base")
        logger.info("Successfully loaded Whisper model")
    except Exception as e:
        logger.error(f"Error loading Whisper model: {str(e)}")

    try:
        emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
        sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        logger.info("Successfully loaded emotion and sentiment analyzers")
    except Exception as e:
        logger.error(f"Error loading emotion/sentiment models: {str(e)}")

    # Initialize Gemini with proper error handling
    try:
        if GOOGLE_API_KEY:
            genai.configure(api_key=GOOGLE_API_KEY)
            gemini_model = genai.GenerativeModel('gemini-1.5-flash')  # Use the latest model
            logger.info("Successfully initialized Gemini model")
        else:
            logger.warning("Gemini model not initialized - API key missing")
    except Exception as e:
        logger.error(f"Error initializing Gemini model: {str(e)}")
    
    logger.info("Registered routes:")
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            logger.info(f"Route: {route.path}, Methods: {route.methods}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")

app = FastAPI(lifespan=lifespan)

# Development CORS configuration
origins = ["*"]  

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

class CustomerRequest(BaseModel):
    scenario: str
    agentMessage: str
    model: str

class SellingSkillRequest(BaseModel):
    audioFeatures: Dict[str, Union[float, List[float]]]
    voiceMetrics: Dict[str, float]

class FeedbackRequest(BaseModel):
    customerMessage: str
    agentMessage: str
    model: str

def convert_numpy_types(obj: Any) -> Any:
    """Convert NumPy types to Python native types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def call_groq_api(prompt: str, model: str = "llama3-8b-8192", max_tokens: int = 1000, temperature: float = 0.7) -> str:
    """
    Call Groq API with the given prompt and return the response.
    
    Available models:
    - llama3-8b-8192
    - llama3-70b-8192
    - mixtral-8x7b-32768
    - gemma-7b-it
    """
    try:
        logger.info(f"Calling Groq API with model: {model}")
        
        # Map model names for compatibility with the original interface
        model_mapping = {
            "llama3": "llama3-8b-8192",
            "llama3:8b": "llama3-8b-8192", 
            "llama3:70b": "llama3-70b-8192",
            "mixtral": "mixtral-8x7b-32768",
            "gemma": "gemma-7b-it"
        }
        
        # Use mapped model or fallback to default
        groq_model = model_mapping.get(model, "llama3-8b-8192")
        
        completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=groq_model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            stream=False,
            stop=None,
        )
        
        response = completion.choices[0].message.content
        logger.info(f"Groq API response received (length: {len(response)})")
        return response
        
    except Exception as e:
        logger.error(f"Error calling Groq API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calling Groq API: {str(e)}")

class AudioAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = sentiment_analyzer
        self.emotion_analyzer = emotion_analyzer
        self.whisper_model = whisper_model
        logger.info("AudioAnalyzer initialized successfully")

    def extract_audio_features(self, audio_path: str) -> Dict[str, Any]:
        try:
            # Load audio with librosa
            y, sr = librosa.load(audio_path, sr=None)
            
            # Handle NaN values and extract features safely
            def safe_mean(arr):
                """Calculate mean while handling NaN values"""
                arr = np.array(arr)
                if np.isnan(arr).all():
                    return 0.0
                return float(np.nanmean(arr))

            def safe_feature_extract(feature_func, *args, **kwargs):
                """Safely extract features with error handling"""
                try:
                    result = feature_func(*args, **kwargs)
                    if hasattr(result, 'mean'):
                        return safe_mean(result)
                    return float(result) if not np.isnan(result) else 0.0
                except Exception:
                    return 0.0
            
            # Extract features with error handling
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_means = [safe_mean(mfcc[i]) for i in range(min(13, mfcc.shape[0]))]
            
            features = {
                'mfcc': mfcc_means,
                'spectral_centroid': safe_feature_extract(lambda: librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()),
                'spectral_bandwidth': safe_feature_extract(lambda: librosa.feature.spectral_bandwidth(y=y, sr=sr)[0].mean()),
                'spectral_rolloff': safe_feature_extract(lambda: librosa.feature.spectral_rolloff(y=y, sr=sr)[0].mean()),
                'zero_crossing_rate': safe_feature_extract(lambda: librosa.feature.zero_crossing_rate(y=y)[0].mean()),
                'rms': safe_feature_extract(lambda: librosa.feature.rms(y=y)[0].mean()),
                'tempo': safe_feature_extract(lambda: librosa.beat.tempo(y=y, sr=sr)[0]),
                'onset_strength': safe_feature_extract(lambda: librosa.onset.onset_strength(y=y, sr=sr).mean()),
                'duration': float(librosa.get_duration(y=y, sr=sr))
            }
            
            # Try to extract pitch with fallback
            try:
                pitch = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
                features['pitch'] = safe_mean(pitch)
            except Exception:
                features['pitch'] = 0.0
            
            return features
        except Exception as e:
            logger.error(f"Error extracting audio features: {str(e)}")
            # Return default features instead of raising exception
            return {
                'mfcc': [0.0] * 13,
                'spectral_centroid': 0.0,
                'spectral_bandwidth': 0.0,
                'spectral_rolloff': 0.0,
                'zero_crossing_rate': 0.0,
                'rms': 0.0,
                'tempo': 0.0,
                'onset_strength': 0.0,
                'pitch': 0.0,
                'duration': 0.0
            }

    def load_audio(self, audio_path: str) -> tuple:
        try:
            # Try loading with librosa first
            y, sr = librosa.load(audio_path, sr=None)
            return y, sr
        except Exception as e:
            logger.error(f"Error loading audio with librosa: {str(e)}")
            try:
                # Fallback to soundfile
                y, sr = sf.read(audio_path)
                return y, sr
            except Exception as e:
                logger.error(f"Error loading audio with soundfile: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to load audio file: {str(e)}")

    def calculate_voice_metrics(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        try:
            # Calculate basic metrics with error handling
            def safe_calculation(func, default=0.0):
                try:
                    result = func()
                    return float(result) if not np.isnan(result) else default
                except Exception:
                    return default
            
            # Calculate RMS energy (volume)
            rms = librosa.feature.rms(y=y)[0]
            volume = safe_calculation(lambda: np.mean(rms))
            
            # Calculate pitch stability
            pitch_stability = safe_calculation(lambda: 1.0 / (np.std(y) + 1e-6))
            
            # Calculate pace (tempo-based)
            tempo = safe_calculation(lambda: librosa.beat.tempo(y=y, sr=sr)[0], 120.0)
            pace = safe_calculation(lambda: tempo / 60.0)
            
            # Calculate clarity (signal quality)
            S = np.abs(librosa.stft(y))
            clarity = safe_calculation(lambda: np.mean(S) / (np.std(S) + 1e-6))
            
            # Normalize metrics to 0-1 range
            metrics = {
                'volume': min(1.0, max(0.0, volume * 10)),
                'pitch_stability': min(1.0, max(0.0, pitch_stability / 100)),
                'pace': min(1.0, max(0.0, pace / 5.0)),
                'clarity': min(1.0, max(0.0, clarity / 100))
            }
            
            return metrics
        except Exception as e:
            logger.error(f"Error calculating voice metrics: {str(e)}")
            # Return default metrics
            return {
                'volume': 0.5,
                'pitch_stability': 0.5,
                'pace': 0.5,
                'clarity': 0.5
            }

    def detect_emotion(self, text: str) -> Dict[str, Any]:
        try:
            if not text or not text.strip():
                return {
                    'emotion': {'emotion': 'neutral', 'confidence': 0.0},
                    'sentiment': {'sentiment': 'NEUTRAL', 'confidence': 0.0}
                }
            
            emotion_result = {'label': 'neutral', 'score': 0.0}
            sentiment_result = {'label': 'NEUTRAL', 'score': 0.0}
            
            if self.emotion_analyzer:
                try:
                    emotion_result = self.emotion_analyzer(text)[0]
                except Exception as e:
                    logger.error(f"Error in emotion analysis: {str(e)}")
            
            if self.sentiment_analyzer:
                try:
                    sentiment_result = self.sentiment_analyzer(text)[0]
                except Exception as e:
                    logger.error(f"Error in sentiment analysis: {str(e)}")
            
            return {
                'emotion': {
                    'emotion': emotion_result['label'],
                    'confidence': float(emotion_result['score'])
                },
                'sentiment': {
                    'sentiment': sentiment_result['label'],
                    'confidence': float(sentiment_result['score'])
                }
            }
        except Exception as e:
            logger.error(f"Error detecting emotion: {str(e)}")
            return {
                'emotion': {'emotion': 'neutral', 'confidence': 0.0},
                'sentiment': {'sentiment': 'NEUTRAL', 'confidence': 0.0}
            }

    def transcribe_audio(self, audio_path: str) -> str:
        try:
            if not self.whisper_model:
                return "Transcription unavailable - model not loaded"
            
            result = self.whisper_model.transcribe(audio_path)
            return result['text'] if result and 'text' in result else ""
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return ""

# Initialize the analyzer
analyzer = AudioAnalyzer()

def create_advanced_fallback_analysis(voice_metrics: dict, audio_features: dict):
    strengths = []
    weaknesses = []

    # Interpret Voice Metrics
    volume = voice_metrics.get('volume', 0)
    clarity = voice_metrics.get('clarity', 0)
    pace = voice_metrics.get('pace', 0)
    pitch_stability = voice_metrics.get('pitch_stability', 0)

    # Interpret Audio Features (Simplified interpretation for fallback)
    # These are simplified interpretations based on the original technical metrics
    # You would need more detailed logic to interpret all 13 MFCCs, spectral features etc.
    # This fallback focuses on the more easily interpretable metrics.
    rms_energy = audio_features.get('rms', 0)
    zero_crossing_rate = audio_features.get('zero_crossing_rate', 0)
    tempo = audio_features.get('tempo', 0) # Assuming tempo is available

    # --- Practical Interpretation and Feedback Generation ---

    # Example interpretation based on thresholds - adjust these as needed

    # Volume/Energy
    if volume > 0.7 and rms_energy > 0.5:
        strengths.append("Good vocal energy and projection, making your voice clear and engaging.")
    elif volume < 0.4 or rms_energy < 0.3:
        weaknesses.append("Your volume or vocal energy could be higher to ensure your message is heard clearly and impactfuly.")

    # Clarity/Pronunciation
    if clarity > 0.7:
        strengths.append("Your pronunciation is clear, making it easy for clients to understand you.")
    elif clarity < 0.5:
        weaknesses.append("Focus on speaking more distinctly to improve clarity.")

    # Pace/Tempo
    # Assuming an ideal tempo range like 100-140 BPM
    # Note: tempo calculation is complex and might not be perfectly accurate in simple feature extraction
    if tempo > 100 and tempo < 140:
         # This check might need refinement based on the actual tempo calculation method
         # For fallback, we might rely more on 'pace' metric if tempo is unreliable
         pass # Add strength if tempo is reliably calculated and in good range
    if pace > 0.5: # Using the pace metric as a proxy if tempo is complex
         strengths.append("Your speaking pace is effective, keeping listeners engaged.")
    elif pace < 0.3:
        weaknesses.append("Your pace might be too slow, try to speak a bit faster to maintain momentum.")
    elif pace > 0.8:
         weaknesses.append("Your pace might be too fast, try to slow down slightly for better comprehension.")

    # Pitch Stability (can relate to confidence)
    if pitch_stability > 0.6:
        strengths.append("Stable pitch suggests confidence and control in your delivery.")
    elif pitch_stability < 0.4:
        weaknesses.append("Inconsistent pitch can make your voice sound less confident. Work on vocal variety.")

    # Ensure exactly two strengths and two weaknesses for consistency
    while len(strengths) < 2:
        strengths.append("Further analysis needed for additional strengths.") # Placeholder or generic
    while len(weaknesses) < 2:
        weaknesses.append("Further analysis needed for additional areas for improvement.") # Placeholder or generic

    return {
        "strengths": strengths[:2], # Take the first two if more generated
        "weaknesses": weaknesses[:2] # Take the first two if more generated
    }

def call_gemini_api_direct(audio_features: dict, voice_metrics: dict, prompt_text: str = "Analyze the following audio features and voice metrics to provide specific strengths and areas for improvement related to insurance sales communication. Interpret the technical metrics into practical insights. Provide exactly two strengths and exactly two areas for improvement in a JSON object with keys 'strengths' and 'weaknesses'. Avoid technical jargon in the feedback."):
    """Direct API call to Gemini without SDK dependencies"""
    try:
        if not GOOGLE_API_KEY:
            raise Exception("No API key available")
        
        prompt = f"""You are an expert insurance sales coach analyzing voice characteristics for sales effectiveness.

Detailed Audio Analysis Data:

Voice Metrics (0-1 scale):
- Volume: {voice_metrics.get('volume', 0.5)} (Higher is better)
- Pitch Stability: {voice_metrics.get('pitch_stability', 0.5)} (Higher is better)
- Pace: {voice_metrics.get('pace', 0.5)} (0.4-0.7 is optimal)
- Clarity: {voice_metrics.get('clarity', 0.5)} (Higher is better)

Audio Features:
- MFCC (Mel-frequency cepstral coefficients): {json.dumps(audio_features.get('mfcc', [0.0] * 13), indent=2)}
  (Indicates voice timbre and quality)
- Spectral Centroid: {audio_features.get('spectral_centroid', 0.0)}
  (Higher values indicate brighter, more energetic voice)
- Spectral Bandwidth: {audio_features.get('spectral_bandwidth', 0.0)}
  (Indicates voice richness and complexity)
- Spectral Rolloff: {audio_features.get('spectral_rolloff', 0.0)}
  (Indicates voice brightness and energy distribution)
- Zero Crossing Rate: {audio_features.get('zero_crossing_rate', 0.0)}
  (Indicates voice clarity and articulation)
- RMS Energy: {audio_features.get('rms', 0.0)}
  (Indicates overall voice power and presence)
- Tempo: {audio_features.get('tempo', 120.0)} BPM
  (Optimal range: 100-140 BPM for sales presentations)
- Onset Strength: {audio_features.get('onset_strength', 0.0)}
  (Indicates speech rhythm and emphasis)
- Pitch: {audio_features.get('pitch', 0.0)} Hz
  (Indicates voice pitch characteristics)
- Duration: {audio_features.get('duration', 0.0)} seconds
  (Length of the speech segment)

Analyze this comprehensive data and provide exactly 2 strengths and 2 areas for improvement related to insurance sales communication. Focus on:
1. Voice quality and clarity
2. Speaking pace and rhythm
3. Vocal energy and engagement
4. Professional presentation
5. Client communication effectiveness

Respond in this EXACT JSON format (no other text):
{{
  "strengths": [
    "First strength related to insurance sales communication",
    "Second strength related to insurance sales communication"
  ],
  "weaknesses": [
    "First area for improvement in insurance sales context", 
    "Second area for improvement in insurance sales context"
  ]
}}"""

        # Use the correct Gemini 1.5 Flash endpoint
        api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": GOOGLE_API_KEY
        }
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": f"{prompt_text}\n\nAudio Features: {json.dumps(audio_features, indent=2)}\n\nVoice Metrics: {json.dumps(voice_metrics, indent=2)}"}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "topP": 0.95,
                "topK": 40,
                "maxOutputTokens": 500
            }
        }
        
        logger.info("Making direct API call to Gemini...")
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 403:
            logger.error("API key authentication failed - using fallback")
            raise Exception("API authentication failed")
        
        response.raise_for_status()
        
        result_data = response.json()
        logger.info(f"Raw API response: {result_data}")
        
        if 'candidates' not in result_data or not result_data['candidates']:
            raise Exception("No candidates in response")
        
        raw_text = result_data["candidates"][0]["content"]["parts"][0]["text"]
        logger.info(f"Generated text: {raw_text}")
        
        # Extract JSON from response
        start_idx = raw_text.find('{')
        end_idx = raw_text.rfind('}') + 1
        
        if start_idx == -1 or end_idx <= start_idx:
            raise Exception("No JSON found in response")
        
        json_str = raw_text[start_idx:end_idx]
        result = json.loads(json_str)
        
        # Validate structure
        if (not isinstance(result.get("strengths"), list) or len(result["strengths"]) < 2 or
            not isinstance(result.get("weaknesses"), list) or len(result["weaknesses"]) < 2):
            raise Exception("Invalid response structure")
        
        final_result = {
            "strengths": [str(s).strip() for s in result["strengths"][:2] if str(s).strip()],
            "weaknesses": [str(w).strip() for w in result["weaknesses"][:2] if str(w).strip()]
        }
        
        logger.info(f"Successfully parsed API response: {json.dumps(final_result, indent=2)}")
        return final_result
        
    except Exception as e:
        logger.error(f"Direct API call failed: {str(e)}")
        raise e

@app.get("/")
async def root():
    return {"message": "Voice Analyzer API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models": {
            "whisper": whisper_model is not None,
            "emotion_analyzer": emotion_analyzer is not None,
            "sentiment_analyzer": sentiment_analyzer is not None,
            "gemini": GOOGLE_API_KEY is not None,
            "groq": groq_client is not None
        }
    }

@app.post("/api/generate-customer-response")
async def generate_customer_response(request: CustomerRequest):
    try:
        logger.info(f"Received customer response request: {request}")
        
        # Construct the prompt
        prompt = f"""You are a customer in a roleplay scenario. The scenario is: {request.scenario}
        The agent (sales representative) just said: {request.agentMessage}
        Respond as the customer would, keeping it concise and realistic."""

        # Call Groq API instead of Ollama
        logger.info(f"Calling Groq API with model: {request.model}")
        response = call_groq_api(prompt, request.model, max_tokens=500, temperature=0.8)
        
        logger.info(f"Generated response: {response}")
        return {"response": response}

    except Exception as e:
        logger.error(f"Error in generate_customer_response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-feedback")
async def generate_feedback(request: FeedbackRequest):
    try:
        logger.info(f"Received feedback request: {request}")
        
        # Construct the feedback prompt
        prompt = f"""Analyze this sales conversation and provide structured feedback:

        Customer: {request.customerMessage}
        Agent: {request.agentMessage}

        Provide feedback in this format:
        1. Communication Effectiveness
        [Your feedback here]
        Score: [1-10]
        Level: [Low/Moderate/High]
        Presence: [Present/Missing]
        Impact: [Potential improvement impact]

        2. Technical Accuracy
        [Your feedback here]
        Score: [1-10]
        Level: [Low/Moderate/High]
        Presence: [Present/Missing]
        Impact: [Potential improvement impact]

        3. Areas for Improvement
        [Your feedback here]
        Score: [1-10]
        Level: [Low/Moderate/High]
        Presence: [Present/Missing]
        Impact: [Potential improvement impact]

        4. What was done well
        [Your feedback here]"""

        # Call Groq API instead of Ollama
        logger.info(f"Calling Groq API with model: {request.model}")
        feedback = call_groq_api(prompt, request.model, max_tokens=1500, temperature=0.7)
        
        logger.info(f"Generated feedback: {feedback}")
        return {"feedback": feedback}

    except Exception as e:
        logger.error(f"Error in generate_feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):
    temp_path = None
    try:
        logger.info("Starting audio analysis...")
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            temp_path = temp_file.name
            logger.info(f"Created temporary file at: {temp_path}")

        try:
            # Load and analyze audio
            logger.info("Loading audio file...")
            y, sr = analyzer.load_audio(temp_path)
            logger.info("Audio loaded successfully")
            
            # Get analysis results
            logger.info("Transcribing audio...")
            transcript = analyzer.transcribe_audio(temp_path)
            logger.info(f"Transcript: {transcript}")
            
            logger.info("Analyzing emotion...")
            emotion_analysis = analyzer.detect_emotion(transcript)
            
            logger.info("Calculating voice metrics...")
            voice_metrics = analyzer.calculate_voice_metrics(y, sr)
            
            logger.info("Extracting audio features...")
            audio_features = analyzer.extract_audio_features(temp_path)
            
            # Clean up
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
                logger.info("Temporary file cleaned up")
            
            gc.collect()
            
            # Convert all NumPy types to Python native types
            response_data = {
                'transcript': transcript,
                'emotion': emotion_analysis['emotion'],
                'sentiment': emotion_analysis['sentiment'],
                'voice_metrics': convert_numpy_types(voice_metrics),
                'audio_features': convert_numpy_types(audio_features)
            }
            
            logger.info("Analysis completed successfully")
            return response_data
            
        except Exception as e:
            logger.exception("Error processing audio")
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
                logger.info("Temporary file cleaned up after error")
            
            return JSONResponse(
                status_code=200,
                content={
                    "error": str(e),
                    "error_type": type(e)._name_,
                    "message": "Audio processing failed",
                    "transcript": "",
                    "emotion": {'emotion': 'neutral', 'confidence': 0.0},
                    "sentiment": {'sentiment': 'NEUTRAL', 'confidence': 0.0},
                    "voice_metrics": {'volume': 0.5, 'pitch_stability': 0.5, 'pace': 0.5, 'clarity': 0.5},
                    "audio_features": {'mfcc': [0.0] * 13, 'spectral_centroid': 0.0}
                }
            )
            
    except Exception as e:
        logger.exception("Error handling upload")
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
            logger.info("Temporary file cleaned up after error")
        
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "error_type": type(e)._name_,
                "message": "File upload failed"
            }
        )

@app.post("/api/analyze-selling-skill")
async def analyze_selling_skill(payload: SellingSkillRequest):
    try:
        logger.info("Starting selling skill analysis...")
        logger.info(f"Received payload: {json.dumps(convert_numpy_types(payload.dict()), indent=2)}")
        
        # Clean and validate input data
        audio_features = convert_numpy_types(payload.audioFeatures)
        voice_metrics = convert_numpy_types(payload.voiceMetrics)
        
        # Try direct API call first
        try:
            logger.info("Attempting direct Gemini API call...")
            result = call_gemini_api_direct(audio_features, voice_metrics)
            logger.info(f"Successfully got API response: {json.dumps(result, indent=2)}")
            return result
        except Exception as e:
            logger.error(f"Direct API call failed: {str(e)}")
        
        # Try SDK as fallback
        if gemini_model:
            try:
                logger.info("Attempting Gemini SDK analysis...")
                
                prompt = f"""You are an expert insurance sales coach. Analyze these voice metrics and audio features for an insurance salesman.

Audio Features:
{json.dumps(audio_features, indent=2)}

Voice Metrics:
{json.dumps(voice_metrics, indent=2)}

Based on this data, provide exactly 2 strengths and 2 weaknesses. Return your response in this exact JSON format:

{{
  "strengths": ["strength 1", "strength 2"],
  "weaknesses": ["weakness 1", "weakness 2"]
}}

Focus on practical sales coaching advice related to voice characteristics, speaking patterns, and communication effectiveness."""

                response = gemini_model.generate_content(prompt)
                raw_text = response.text
                logger.info(f"Gemini SDK response: {raw_text}")
                
                # Extract JSON from response
                start_idx = raw_text.find('{')
                end_idx = raw_text.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = raw_text[start_idx:end_idx]
                    result = json.loads(json_str)
                    
                    # Validate structure
                    if (isinstance(result.get("strengths"), list) and len(result["strengths"]) >= 2 and
                        isinstance(result.get("weaknesses"), list) and len(result["weaknesses"]) >= 2):
                        
                        final_result = {
                            "strengths": [str(s).strip() for s in result["strengths"][:2] if str(s).strip()],
                            "weaknesses": [str(w).strip() for w in result["weaknesses"][:2] if str(w).strip()]
                        }
                        
                        logger.info(f"Successfully parsed Gemini SDK response: {json.dumps(final_result, indent=2)}")
                        return final_result
                        
            except Exception as e:
                logger.error(f"Gemini SDK error: {str(e)}")
        
        # Use advanced fallback analysis
        logger.info("Using advanced fallback analysis...")
        fallback_result = create_advanced_fallback_analysis(voice_metrics, audio_features)
        logger.info(f"Advanced fallback analysis result: {json.dumps(fallback_result, indent=2)}")
        return fallback_result
        
    except Exception as e:
        logger.exception("Critical error in analyze_selling_skill")
        
        # Return a safe fallback
        return {
            "strengths": [
                "Professional voice characteristics detected in your communication style",
                "Voice patterns suitable for effective insurance sales interactions"
            ],
            "weaknesses": [
                "Consider voice training to enhance client engagement and trust-building",
                "Practice varying vocal dynamics to maintain client interest during presentations"
            ]
        }

if __name__ == "__main__":
    logger.info("Starting FastAPI server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)