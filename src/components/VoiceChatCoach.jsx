import React, { useState, useRef, useEffect } from 'react';
import { Box, Button, Typography, CircularProgress, Paper, Container, Grid } from '@mui/material';
import MicIcon from '@mui/icons-material/Mic';
import StopIcon from '@mui/icons-material/Stop';
import SentimentSatisfiedIcon from '@mui/icons-material/SentimentSatisfied';
import axios from 'axios';

const VoiceChatCoach = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [transcriptionError, setTranscriptionError] = useState(null);
  const [transcript, setTranscript] = useState('');
  const [recordingTime, setRecordingTime] = useState(0);
  const [toneAnalysis, setToneAnalysis] = useState(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const streamRef = useRef(null);
  const timerRef = useRef(null);

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, []);

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const startRecording = async () => {
    try {
      // Clear previous transcript and analysis when starting new recording
      setTranscript('');
      setToneAnalysis(null);
      
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];
      setRecordingTime(0);

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        await transcribeAudio(audioBlob);
        
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop());
          streamRef.current = null;
        }
        if (timerRef.current) {
          clearInterval(timerRef.current);
          timerRef.current = null;
        }
      };

      mediaRecorder.start();
      setIsRecording(true);
      setTranscriptionError(null);
      
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
    } catch (error) {
      console.error('Error accessing microphone:', error);
      setTranscriptionError('Error accessing microphone. Please ensure you have granted microphone permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }
  };

  const analyzeTone = async (audioBlob) => {
    try {
      const formData = new FormData();
      formData.append('file', audioBlob, 'recording.webm');

      const response = await axios.post('http://127.0.0.1:8000/api/analyze-audio', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 15000,
      });

      if (response.data && response.data.emotion && response.data.sentiment) {
        setToneAnalysis({
          emotion: {
            emotion: response.data.emotion.emotion || 'neutral',
            confidence: response.data.emotion.confidence || 0
          },
          sentiment: {
            sentiment: response.data.sentiment.sentiment || 'NEUTRAL',
            confidence: response.data.sentiment.confidence || 0
          }
        });
      }
    } catch (error) {
      console.error('Error analyzing tone:', error);
    }
  };

  const transcribeAudio = async (audioBlob) => {
    setIsTranscribing(true);
    setTranscriptionError(null);
    
    try {
      const formData = new FormData();
      formData.append('file', audioBlob, 'recording.webm');

      const response = await axios.post('http://127.0.0.1:8000/transcribe/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 15000,
      });

      const transcribedText = response.data.transcript;
      setTranscript(transcribedText);
      
      // Analyze tone using the same audio blob
      await analyzeTone(audioBlob);
    } catch (error) {
      console.error('Error transcribing audio:', error);
      let errorMessage = 'Error transcribing audio. ';
      
      if (axios.isAxiosError(error)) {
        if (error.code === 'ERR_NETWORK') {
          errorMessage += 'Unable to connect to the server. Please ensure the backend server is running at http://127.0.0.1:8000';
        } else if (error.code === 'ECONNABORTED') {
          errorMessage += 'Request timed out. The server took too long to respond.';
        } else if (error.response) {
          errorMessage += `Server error: ${error.response.data.detail || 'An unexpected error occurred'}`;
        }
      }
      
      setTranscriptionError(errorMessage);
    } finally {
      setIsTranscribing(false);
    }
  };

  const getEmotionColor = (emotion) => {
    const colors = {
      joy: '#4caf50',
      sadness: '#2196f3',
      anger: '#f44336',
      fear: '#9c27b0',
      surprise: '#ff9800',
      neutral: '#9e9e9e'
    };
    return colors[emotion?.toLowerCase()] || colors.neutral;
  };

  const formatConfidence = (confidence) => {
    if (typeof confidence === 'number') {
      return `${(confidence * 100).toFixed(1)}%`;
    }
    return '0%';
  };

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Box sx={{ mb: 4, textAlign: 'center' }}>
        <Typography variant="h4" component="h1" gutterBottom sx={{ fontWeight: 'bold' }}>
          Voice Chat Coach
        </Typography>
        <Typography variant="subtitle1" color="text.secondary" sx={{ maxWidth: 600, mx: 'auto' }}>
          Record your voice to get instant transcription and feedback. Perfect for practicing your speaking skills.
        </Typography>
      </Box>

      <Paper 
        elevation={3} 
        sx={{ 
          p: 4, 
          borderRadius: 2,
          bgcolor: 'background.paper',
          boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
        }}
      >
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3 }}>
          <Button
            variant="contained"
            color={isRecording ? 'error' : 'primary'}
            size="large"
            startIcon={isRecording ? <StopIcon /> : <MicIcon />}
            onClick={isRecording ? stopRecording : startRecording}
            disabled={isTranscribing}
            sx={{
              minWidth: 200,
              py: 1.5,
              borderRadius: 2,
              textTransform: 'none',
              fontSize: '1.1rem',
              boxShadow: isRecording ? '0 0 10px rgba(244, 67, 54, 0.3)' : '0 0 10px rgba(25, 118, 210, 0.3)',
              '&:hover': {
                boxShadow: isRecording ? '0 0 15px rgba(244, 67, 54, 0.4)' : '0 0 15px rgba(25, 118, 210, 0.4)',
              }
            }}
          >
            {isRecording ? 'Stop Recording' : 'Start Recording'}
          </Button>

          {isRecording && (
            <Typography 
              variant="h6" 
              color="error" 
              sx={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: 1,
                animation: 'pulse 1.5s infinite'
              }}
            >
              Recording: {formatTime(recordingTime)}
            </Typography>
          )}

          {transcriptionError && (
            <Typography 
              color="error" 
              sx={{ 
                textAlign: 'center',
                bgcolor: 'error.light',
                color: 'error.contrastText',
                p: 2,
                borderRadius: 1,
                width: '100%'
              }}
            >
              {transcriptionError}
            </Typography>
          )}

          {isTranscribing && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <CircularProgress size={24} />
              <Typography variant="body1">Transcribing your voice...</Typography>
            </Box>
          )}

          {transcript && (
            <Paper 
              elevation={1} 
              sx={{ 
                p: 3, 
                width: '100%',
                bgcolor: 'grey.50',
                borderRadius: 2
              }}
            >
              <Typography variant="h6" gutterBottom sx={{ color: 'primary.main' }}>
                Transcription
              </Typography>
              <Typography 
                variant="body1" 
                sx={{ 
                  whiteSpace: 'pre-wrap',
                  lineHeight: 1.6
                }}
              >
                {transcript}
              </Typography>
            </Paper>
          )}

          {toneAnalysis && (
            <Grid container spacing={2} sx={{ width: '100%', mt: 2 }}>
              <Grid item xs={12} md={6}>
                <Paper 
                  elevation={1} 
                  sx={{ 
                    p: 2,
                    bgcolor: 'grey.50',
                    borderRadius: 2
                  }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                    <SentimentSatisfiedIcon sx={{ color: getEmotionColor(toneAnalysis.emotion?.emotion) }} />
                    <Typography variant="h6">
                      Emotion Analysis
                    </Typography>
                  </Box>
                  <Typography variant="body1" sx={{ mb: 1 }}>
                    Emotion: {toneAnalysis.emotion?.emotion || 'Unknown'} 
                    {toneAnalysis.emotion?.confidence && ` (${formatConfidence(toneAnalysis.emotion.confidence)})`}
                  </Typography>
                  <Typography variant="body1">
                    Sentiment: {toneAnalysis.sentiment?.sentiment || 'Unknown'}
                    {toneAnalysis.sentiment?.confidence && ` (${formatConfidence(toneAnalysis.sentiment.confidence)})`}
                  </Typography>
                </Paper>
              </Grid>
            </Grid>
          )}
        </Box>
      </Paper>
    </Container>
  );
};

export default VoiceChatCoach;