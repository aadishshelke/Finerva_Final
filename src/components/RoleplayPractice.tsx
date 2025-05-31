import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card } from "@/components/ui/card";
import { Star, Play, Clock, Users, Target, Loader2, CheckCircle2, AlertCircle, Lightbulb, Sparkles, Mic } from 'lucide-react';
import { useLocation } from 'react-router-dom';

const MODEL_NAME = "llama3";

interface Message {
  role: 'customer' | 'user';
  content: string;
}

interface Scenario {
  id: string;
  title: string;
  description: string;
  difficulty: string;
  duration: string;
  xpReward: string;
  tags: string[];
  initialMessage: string;
}

interface CustomScenario {
  customerProfile: string;
  customerNeeds: string;
  customerConcerns: string;
  salesFocus: string;
}

interface FeedbackItem {
  title: string;
  summary: string;
  details?: string;
  metrics?: {
    score?: number;
    level?: 'Low' | 'Moderate' | 'High';
    presence?: boolean;
    impact?: string;
  };
}

const RoleplayPractice: React.FC = () => {
  const location = useLocation();
  const [messages, setMessages] = useState<Message[]>([
    { 
      role: 'customer', 
      content: "Hi, I'm a 39-year-old single mother who works as a school teacher. I want to secure life insurance to protect my 8-year-old daughter's future. I'm concerned about affordability and want to understand the benefits clearly."
    }
  ]);
  const [feedback, setFeedback] = useState<string | null>(null);
  const [scenario, setScenario] = useState<string>("Customer: Hi, I'm a 39-year-old single mother who works as a school teacher. I want to secure life insurance to protect my 8-year-old daughter's future. I'm concerned about affordability and want to understand the benefits clearly.");
  const [userInput, setUserInput] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [selectedScenario, setSelectedScenario] = useState<string | null>(null);
  const [typingIndicator, setTypingIndicator] = useState<boolean>(false);
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [transcriptionError, setTranscriptionError] = useState<string | null>(null);

  const scenarios: Scenario[] = [
    {
      id: '1',
      title: 'Skeptical Customer',
      description: 'Handle objections from a doubtful customer who questions insurance value',
      difficulty: 'Intermediate',
      duration: '10-15 min',
      xpReward: '150 XP',
      tags: ['Objection Handling', 'Insurance Basics'],
      initialMessage: "I'm not sure if I really need life insurance. It seems like an unnecessary expense."
    },
    {
      id: '2',
      title: 'Budget-Conscious Client',
      description: 'Navigate price sensitivity while highlighting value proposition',
      difficulty: 'Beginner',
      duration: '8-12 min',
      xpReward: '100 XP',
      tags: ['Value Selling', 'Price Handling'],
      initialMessage: "I'm interested in life insurance, but I'm worried about the cost. What are my options?"
    },
    {
      id: '3',
      title: 'Enterprise Decision Maker',
      description: 'Engage with a C-level executive seeking comprehensive coverage',
      difficulty: 'Advanced',
      duration: '15-20 min',
      xpReward: '200 XP',
      tags: ['Enterprise Sales', 'Decision Makers'],
      initialMessage: "We need to review our company's life insurance policies. What comprehensive solutions do you offer?"
    }
  ];

  useEffect(() => {
    // Handle scenario data from dashboard
    if (location.state?.scenario) {
      try {
        const customScenario: CustomScenario = JSON.parse(location.state.scenario);
        const initialMessage = `Customer: Hi, I'm a ${customScenario.customerProfile}. ${customScenario.customerNeeds}. ${customScenario.customerConcerns}.`;
        setScenario(initialMessage);
        setMessages([{ role: 'customer', content: initialMessage.split("Customer: ")[1] }]);
        
        // Auto-start if specified
        if (location.state.autoStart) {
          setSelectedScenario('custom');
        }
      } catch (error) {
        console.error('Error parsing scenario:', error);
      }
    }
  }, [location.state]);

  const handleUserSubmit = async () => {
    if (!userInput.trim()) return;

    const updatedMessages: Message[] = [...messages, { role: 'user', content: userInput }];
    setMessages(updatedMessages);
    setUserInput("");
    setLoading(true);
    setTypingIndicator(true);

    try {
      console.log('Sending request to backend...');
      const customerResponse = await axios.post('/api/generate-customer-response', {
        scenario,
        agentMessage: userInput,
        model: MODEL_NAME
      }, {
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        timeout: 120000 // 2 minute timeout
      });

      console.log('Received customer response:', customerResponse.data);
      const responseText = customerResponse.data.response;
      const customerMessage: Message = { role: 'customer', content: responseText };

      console.log('Sending feedback request...');
      const feedbackRes = await axios.post('/api/generate-feedback', {
        customerMessage: getLatestCustomerMessage(updatedMessages),
        agentMessage: userInput,
        model: MODEL_NAME
      }, {
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        timeout: 120000 // 2 minute timeout
      });

      console.log('Received feedback:', feedbackRes.data);
      setMessages([...updatedMessages, customerMessage]);
      setFeedback(feedbackRes.data.feedback);
    } catch (err) {
      console.error('Error details:', err);
      if (axios.isAxiosError(err)) {
        if (err.code === 'ERR_NETWORK') {
          alert('Unable to connect to the server. Please make sure the backend server is running at http://localhost:8000');
        } else if (err.code === 'ECONNABORTED') {
          alert('Request timed out. The server took too long to respond. Please try again.');
        } else if (err.response) {
          // Handle different HTTP status codes
          switch (err.response.status) {
            case 400:
              alert(`Invalid request: ${err.response.data.detail || 'Please check your input'}`);
              break;
            case 404:
              alert('API endpoint not found. Please check the server configuration.');
              break;
            case 500:
              alert(`Server error: ${err.response.data.detail || 'An unexpected error occurred on the server'}`);
              break;
            case 504:
              alert('Server timeout. The request took too long to process.');
              break;
            default:
              alert(`Error ${err.response.status}: ${err.response.data.detail || 'An error occurred'}`);
          }
        } else {
          alert(`Error: ${err.message}`);
        }
      } else {
        alert('An unexpected error occurred. Please try again.');
      }
    } finally {
      setLoading(false);
      setTypingIndicator(false);
    }
  };

  const getLatestCustomerMessage = (msgs: Message[]): string => {
    for (let i = msgs.length - 1; i >= 0; i--) {
      if (msgs[i].role === 'customer') return msgs[i].content;
    }
    return "";
  };

  const loadSampleScenario = (scenarioId: string) => {
    const selected = scenarios.find(s => s.id === scenarioId);
    if (selected) {
      setScenario(`Customer: ${selected.initialMessage}`);
      setMessages([{ role: 'customer', content: selected.initialMessage }]);
      setFeedback(null);
      setSelectedScenario(scenarioId);
    }
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'Beginner': return 'text-green-600 bg-green-100';
      case 'Intermediate': return 'text-yellow-600 bg-yellow-100';
      case 'Advanced': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const parseFeedback = (feedbackText: string): FeedbackItem[] => {
    const items: FeedbackItem[] = [];
    const sections = feedbackText.split(/\*\*\d+\./);
    
    console.log('Raw feedback text:', feedbackText);
    console.log('Split sections:', sections);

    sections.forEach(section => {
      if (!section.trim()) return;

      const lines = section.split('\n').map(line => line.trim()).filter(line => line);
      if (lines.length === 0) return;

      const title = lines[0].replace(/\*/g, '').trim();
      let summary = '';
      let details = '';
      const metrics: any = {};

      for (let i = 1; i < lines.length; i++) {
        const line = lines[i];
        
        if (line.startsWith('Score:')) {
          const score = parseFloat(line.match(/\d+(\.\d+)?/)?.[0] || '0');
          metrics.score = score;
        } else if (line.startsWith('Level:')) {
          const level = line.match(/Low|Moderate|High/i)?.[0] as 'Low' | 'Moderate' | 'High';
          metrics.level = level;
        } else if (line.startsWith('Presence:')) {
          const presence = line.toLowerCase().includes('present') || line.toLowerCase().includes('yes');
          metrics.presence = presence;
        } else if (line.startsWith('Impact:')) {
          const impact = line.split(':')[1]?.trim();
          metrics.impact = impact;
        } else if (line.startsWith('What was done well')) {
          details = lines.slice(i + 1).join('\n');
          break;
        } else {
          summary += line + ' ';
        }
      }

      items.push({
        title,
        summary: summary.trim(),
        details: details.trim(),
        metrics: Object.keys(metrics).length > 0 ? metrics : undefined
      });
    });

    console.log('Parsed feedback items:', items);
    return items;
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        await transcribeAudio(audioBlob);
        
        // Stop all tracks
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setTranscriptionError(null);
    } catch (error) {
      console.error('Error accessing microphone:', error);
      setTranscriptionError('Error accessing microphone. Please ensure you have granted microphone permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
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

      const transcript = response.data.transcript;
      setUserInput(prev => prev + (prev ? ' ' : '') + transcript);
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
          <h1 className="text-3xl font-bold text-gray-900">Roleplay Practice</h1>
          <p className="text-gray-600 mt-1">Practice real-world scenarios with AI-powered customer personas.</p>
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
              <p className="text-blue-100">Conversation Skills</p>
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
              <p className="text-green-100">Objection Handling</p>
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
              <p className="text-purple-100">Product Knowledge</p>
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
        {/* Scenarios List */}
        <div className="lg:col-span-1 space-y-4">
          <h2 className="text-xl font-semibold mb-4">Available Scenarios</h2>
          {scenarios.map((scenario) => (
            <Card
              key={scenario.id}
              className={`p-4 hover:shadow-lg transition-all duration-200 cursor-pointer ${
                selectedScenario === scenario.id ? 'ring-2 ring-blue-500' : ''
              }`}
              onClick={() => loadSampleScenario(scenario.id)}
            >
              <div className="flex items-start justify-between mb-2">
                <h3 className="text-lg font-semibold text-gray-900">{scenario.title}</h3>
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${getDifficultyColor(scenario.difficulty)}`}>
                  {scenario.difficulty}
                </span>
              </div>
              <p className="text-gray-600 text-sm mb-2">{scenario.description}</p>
              <div className="flex items-center text-sm text-blue-600">
                <Star className="w-4 h-4 mr-2" />
                {scenario.xpReward}
              </div>
            </Card>
          ))}
        </div>

        {/* Chat Area */}
        <div className="lg:col-span-2">
          <Card className="p-4">
            <div className="space-y-4 mb-4">
              {messages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`rounded p-3 border ${
                    msg.role === 'customer'
                      ? 'bg-blue-50 border-blue-200'
                      : msg.role === 'user'
                      ? 'bg-green-50 border-green-200'
                      : 'bg-gray-100 border-gray-300'
                  }`}
                >
                  <strong>{msg.role === 'customer' ? 'Customer' : 'You'}:</strong> {msg.content}
                </div>
              ))}
            </div>

            <div className="mt-4">
              {typingIndicator && (
                <div className="flex items-center text-sm text-gray-600 mb-2">
                  <Loader2 className="w-4 h-4 mr-2 animate-spin text-blue-500" />
                  Customer is typing<span className="typing-dots">...</span>
                </div>
              )}
              <Textarea
                placeholder="Your response here..."
                value={userInput}
                onChange={(e) => setUserInput(e.target.value)}
                rows={4}
                disabled={loading}
                className={`transition-all duration-200 ${
                  loading ? 'bg-gray-100 opacity-60 cursor-not-allowed' : 'bg-white'
                }`}
              />
              <div className="flex gap-2 mt-2">
                <Button 
                  variant={isRecording ? "destructive" : "outline"}
                  size="icon"
                  className={`${
                    loading || isTranscribing ? 'opacity-50 cursor-not-allowed' : ''
                  }`}
                  onClick={isRecording ? stopRecording : startRecording}
                  disabled={loading || isTranscribing}
                >
                  {isTranscribing ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    <Mic className={`w-5 h-5 ${isRecording ? 'animate-pulse' : ''}`} />
                  )}
                </Button>
                <Button 
                  className={`flex-1 sm:flex-none ${
                    loading || isTranscribing ? 'opacity-50 cursor-not-allowed' : ''
                  }`}
                  onClick={handleUserSubmit} 
                  disabled={loading || isTranscribing}
                >
                  {loading ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    'Send'
                  )}
                </Button>
              </div>
              
              {transcriptionError && (
                <div className="mt-2 p-2 bg-red-50 border border-red-200 rounded text-sm text-red-600">
                  <AlertCircle className="w-4 h-4 inline mr-1" />
                  {transcriptionError}
                </div>
              )}
            </div>

            {feedback && (
              <div className="rounded-2xl border border-gray-200 shadow-sm p-6 bg-white space-y-5">
                <h2 className="text-xl font-semibold text-gray-800 flex items-center gap-2">
                  <Sparkles className="w-5 h-5 text-yellow-500" />
                  Feedback Analysis
                </h2>

                <div className="space-y-4">
                  {parseFeedback(feedback).map((item, index) => (
                    <div 
                      key={index} 
                      className="bg-gray-50 rounded-xl p-4 border border-gray-100 shadow-sm space-y-2"
                    >
                      <div className="flex items-start gap-2">
                        <Lightbulb className="w-5 h-5 text-yellow-500 mt-1 flex-shrink-0" />
                        <div className="space-y-2 flex-grow">
                          <div>
                            <p className="text-gray-800">
                              <span className="font-semibold">{item.title}</span>
                              {item.summary && `: ${item.summary}`}
                            </p>
                            
                            {item.metrics && Object.keys(item.metrics).length > 0 && (
                              <div className="mt-2 grid grid-cols-2 gap-2 text-sm">
                                {item.metrics.score !== undefined && (
                                  <div className="flex items-center gap-1">
                                    <span className="text-gray-600">Score:</span>
                                    <span className="font-medium text-blue-600">{item.metrics.score}/10</span>
                                  </div>
                                )}
                                {item.metrics.level && (
                                  <div className="flex items-center gap-1">
                                    <span className="text-gray-600">Level:</span>
                                    <span className={`font-medium ${
                                      item.metrics.level === 'High' ? 'text-green-600' :
                                      item.metrics.level === 'Moderate' ? 'text-yellow-600' :
                                      'text-red-600'
                                    }`}>
                                      {item.metrics.level}
                                    </span>
                                  </div>
                                )}
                                {item.metrics.presence !== undefined && (
                                  <div className="flex items-center gap-1">
                                    <span className="text-gray-600">Presence:</span>
                                    <span className={`font-medium ${item.metrics.presence ? 'text-green-600' : 'text-red-600'}`}>
                                      {item.metrics.presence ? '✓ Present' : '✗ Missing'}
                                    </span>
                                  </div>
                                )}
                                {item.metrics.impact && (
                                  <div className="col-span-2 flex items-center gap-1">
                                    <span className="text-gray-600">Impact:</span>
                                    <span className="font-medium text-purple-600">{item.metrics.impact}</span>
                                  </div>
                                )}
                              </div>
                            )}
                          </div>
                          
                          {item.details && (
                            <div className="text-sm text-gray-600 pt-2 border-t border-gray-100">
                              <p className="font-medium text-gray-700 mb-1">What you did well:</p>
                              <p>{item.details}</p>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </Card>
        </div>
      </div>

      {/* Quick Tips */}
      <Card className="bg-blue-50 p-6 border border-blue-200">
        <h3 className="text-lg font-semibold text-blue-900 mb-3">💡 Pro Tips for Better Performance</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h4 className="font-medium text-blue-800 mb-2">Active Listening</h4>
            <p className="text-blue-700 text-sm">Pay attention to customer concerns and respond thoughtfully to build trust.</p>
          </div>
          <div>
            <h4 className="font-medium text-blue-800 mb-2">Empathy First</h4>
            <p className="text-blue-700 text-sm">Acknowledge customer emotions before presenting solutions or products.</p>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default RoleplayPractice; 