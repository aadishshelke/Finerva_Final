import React from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Brain, Trophy } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const QuizCard = () => {
  const navigate = useNavigate();
  const progress = 75; // 75% complete

  const handleStartQuiz = () => {
    navigate('/learning-modules', { state: { autoStartQuiz: true } });
  };

  return (
    <Card className="p-6 bg-gradient-to-br from-purple-50 to-pink-50">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Brain className="h-5 w-5 text-purple-600" />
          <h2 className="text-lg font-semibold text-gray-800">Daily Quiz</h2>
        </div>
        <div className="flex items-center gap-1">
          <Trophy className="h-4 w-4 text-yellow-500" />
          <span className="text-sm font-medium text-gray-600">+50 XP</span>
        </div>
      </div>

      <div className="space-y-4">
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-gray-600">Progress</span>
            <span className="font-medium text-purple-600">{progress}%</span>
          </div>
          <Progress value={progress} className="h-2" />
        </div>

        <div className="space-y-2">
          <p className="text-sm text-gray-600">
            Test your knowledge on today's sales techniques and best practices.
          </p>
          <div className="flex items-center gap-2 text-sm text-gray-500">
            <span>• 5 questions</span>
            <span>• 10 minutes</span>
          </div>
        </div>

        <Button 
          className="w-full bg-purple-600 hover:bg-purple-700"
          onClick={handleStartQuiz}
        >
          Start Quiz
        </Button>
      </div>
    </Card>
  );
};

export default QuizCard;
