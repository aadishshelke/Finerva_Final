import React, { useState, useRef, useEffect } from 'react';
import { Mic, MicOff, Loader2, AlertCircle, Star, Play, Clock, Users, Target } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from "@/components/ui/textarea";
import axios from 'axios';

interface Message {
  role: 'user';
  content: string;
}

export const VoiceChatCoach: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [transcriptionError, setTranscriptionError] = useState<string | null>(null);
  const [recordingTime, setRecordingTime] = useState<number>(0);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  // Cleanup function to stop all tracks when component unmounts
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

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const startRecording = async () => {
    try {
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
        
        // Stop all tracks
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
      
      // Start the timer
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

  const transcribeAudio = async (audioBlob: Blob) => {
    setIsTranscribing(true);
    setTranscriptionError(null);
    
    try {
      const formData = new FormData();
      formData.append('file', audioBlob, 'recording.webm');

      const response = await axios.post('http://127.0.0.1:8000/transcribe/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 15000, // 15 second timeout
      });

      const transcribedText = response.data.transcript;
      setMessages(prev => [...prev, { role: 'user', content: transcribedText }]);
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

  return (
    <div className="space-y-6 p-4 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Voice Chat Coach</h1>
          <p className="text-gray-600 mt-1">Practice your speaking skills with AI-powered voice coaching.</p>
        </div>
        <div className="flex items-center space-x-4">
          <div className="text-right">
            <p className="text-sm text-gray-600">Today's Progress</p>
            <p className="text-lg font-semibold text-gray-900">3/5 Sessions</p>
          </div>
          <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center">
            <Star className="w-6 h-6 text-blue-600" />
          </div>
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="bg-gradient-to-r from-blue-500 to-blue-600 p-6 text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-blue-100">Speaking Skills</p>
              <div className="flex items-center mt-2">
                {[1, 2, 3, 4, 5].map((star) => (
                  <Star key={star} className="w-4 h-4 fill-current text-yellow-300" />
                ))}
              </div>
            </div>
            <Target className="w-8 h-8 text-blue-200" />
          </div>
        </Card>

        <Card className="bg-gradient-to-r from-green-500 to-green-600 p-6 text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-green-100">Pronunciation</p>
              <div className="flex items-center mt-2">
                {[1, 2, 3, 4].map((star) => (
                  <Star key={star} className="w-4 h-4 fill-current text-yellow-300" />
                ))}
                <Star className="w-4 h-4 text-green-200" />
              </div>
            </div>
            <Users className="w-8 h-8 text-green-200" />
          </div>
        </Card>

        <Card className="bg-gradient-to-r from-purple-500 to-purple-600 p-6 text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-purple-100">Clarity</p>
              <div className="flex items-center mt-2">
                {[1, 2, 3].map((star) => (
                  <Star key={star} className="w-4 h-4 fill-current text-yellow-300" />
                ))}
                <Star className="w-4 h-4 text-purple-200" />
                <Star className="w-4 h-4 text-purple-200" />
              </div>
            </div>
            <Target className="w-8 h-8 text-purple-200" />
          </div>
        </Card>
      </div>

      {/* Chat Interface */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Practice Tips */}
        <div className="lg:col-span-1 space-y-4">
          <h2 className="text-xl font-semibold mb-4">Practice Tips</h2>
          <Card className="p-4 hover:shadow-lg transition-all duration-200">
            <div className="flex items-start justify-between mb-2">
              <h3 className="text-lg font-semibold text-gray-900">Speaking Clearly</h3>
              <span className="px-2 py-1 rounded-full text-xs font-medium text-green-600 bg-green-100">
                Beginner
              </span>
            </div>
            <p className="text-gray-600 text-sm mb-2">Focus on clear pronunciation and steady pace.</p>
            <div className="flex items-center text-sm text-blue-600">
              <Star className="w-4 h-4 mr-2" />
              100 XP
            </div>
          </Card>

          <Card className="p-4 hover:shadow-lg transition-all duration-200">
            <div className="flex items-start justify-between mb-2">
              <h3 className="text-lg font-semibold text-gray-900">Voice Modulation</h3>
              <span className="px-2 py-1 rounded-full text-xs font-medium text-yellow-600 bg-yellow-100">
                Intermediate
              </span>
            </div>
            <p className="text-gray-600 text-sm mb-2">Practice varying your tone and pitch.</p>
            <div className="flex items-center text-sm text-blue-600">
              <Star className="w-4 h-4 mr-2" />
              150 XP
            </div>
          </Card>
        </div>

        {/* Chat Area */}
        <div className="lg:col-span-2">
          <Card className="p-4">
            <div className="space-y-4 mb-4">
              {messages.map((msg, idx) => (
                <div
                  key={idx}
                  className="rounded p-3 border bg-green-50 border-green-200"
                >
                  <strong>You:</strong> {msg.content}
                </div>
              ))}
            </div>

            <div className="mt-4">
              <div className="flex gap-2">
                <Button 
                  variant={isRecording ? "destructive" : "outline"}
                  size="icon"
                  className={`${
                    isTranscribing ? 'opacity-50 cursor-not-allowed' : ''
                  }`}
                  onClick={isRecording ? stopRecording : startRecording}
                  disabled={isTranscribing}
                >
                  {isTranscribing ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    <Mic className={`w-5 h-5 ${isRecording ? 'animate-pulse' : ''}`} />
                  )}
                </Button>
                {isRecording && (
                  <div className="flex items-center text-2xl font-mono font-medium text-gray-700">
                    {formatTime(recordingTime)}
                  </div>
                )}
              </div>
              
              {transcriptionError && (
                <div className="mt-2 p-2 bg-red-50 border border-red-200 rounded text-sm text-red-600">
                  <AlertCircle className="w-4 h-4 inline mr-1" />
                  {transcriptionError}
                </div>
              )}
            </div>
          </Card>
        </div>
      </div>

      <Card className="bg-blue-50 p-6 border border-blue-200">
        <h3 className="text-lg font-semibold text-blue-900 mb-3">💡 Pro Tips for Better Speaking</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h4 className="font-medium text-blue-800 mb-2">Clear Pronunciation</h4>
            <p className="text-blue-700 text-sm">Speak slowly and clearly, enunciating each word properly.</p>
          </div>
          <div>
            <h4 className="font-medium text-blue-800 mb-2">Confident Tone</h4>
            <p className="text-blue-700 text-sm">Maintain a steady, confident voice throughout your speech.</p>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default VoiceChatCoach; 