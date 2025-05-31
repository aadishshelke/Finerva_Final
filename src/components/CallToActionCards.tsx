import React from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { BookOpen, MessageSquare, Mic } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const CallToActionCards = () => {
  const navigate = useNavigate();

  const actions = [
    {
      title: "Learning Modules",
      description: "Access structured learning paths and track your progress",
      icon: <BookOpen className="h-6 w-6" />,
      buttonText: "Start Learning",
      gradient: "from-blue-500 to-indigo-500",
      onClick: () => navigate('/learning-modules')
    },
    {
      title: "Roleplay Practice",
      description: "Practice sales scenarios with AI-powered feedback",
      icon: <MessageSquare className="h-6 w-6" />,
      buttonText: "Start Practice",
      gradient: "from-purple-500 to-pink-500",
      onClick: () => navigate('/roleplay')
    },
    {
      title: "Voice Chat Coach",
      description: "Get real-time coaching during customer calls",
      icon: <Mic className="h-6 w-6" />,
      buttonText: "Start Session",
      gradient: "from-green-500 to-teal-500",
      onClick: () => navigate('/voice-chat')
    }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {actions.map((action, index) => (
        <Card
          key={index}
          className={`p-6 bg-gradient-to-r ${action.gradient} text-white hover:shadow-lg transition-all duration-200`}
        >
          <div className="flex items-center gap-3 mb-4">
            {action.icon}
            <h3 className="text-lg font-semibold">{action.title}</h3>
          </div>
          <p className="text-white/80 mb-4">{action.description}</p>
          <Button
            onClick={action.onClick}
            className="w-full bg-white/10 hover:bg-white/20 text-white border border-white/20"
          >
            {action.buttonText}
          </Button>
        </Card>
      ))}
    </div>
  );
};

export default CallToActionCards; 