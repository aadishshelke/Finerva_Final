import React, { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Badge } from '@/components/ui/badge';
import { useNavigate } from 'react-router-dom';

const DailyMissionCard = () => {
  const [watchedTime, setWatchedTime] = useState(2);
  const totalTime = 5;
  const progress = (watchedTime / totalTime) * 100;
  const navigate = useNavigate();

  const handleStartRoleplay = () => {
    const scenario = {
      customerProfile: "A 39-year-old single mother who works as a school teacher",
      customerNeeds: "Wants to secure life insurance to protect her 8-year-old daughter's future",
      customerConcerns: "Concerned about affordability and wants to understand the benefits clearly",
      salesFocus: "Focus on explaining the importance of life insurance for single parents and finding a plan that fits her budget"
    };

    navigate('/roleplay', {
      state: {
        scenario: JSON.stringify(scenario),
        autoStart: true
      }
    });
  };

  return (
    <TooltipProvider>
      <Card className="p-6 bg-gradient-to-r from-blue-50 to-purple-50 border-0 shadow-lg card-hover">
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-semibold text-gray-900">🎯 Today's Roleplay</h2>
              <p className="text-sm text-gray-600">+50 XP</p>
            </div>
            <Badge variant="secondary" className="bg-blue-100 text-blue-800">
              Roleplay
            </Badge>
          </div>
          
          <div className="bg-white p-4 rounded-lg">
            <h3 className="text-lg font-medium text-gray-800 mb-2">Single Parent Insurance Case</h3>
            <ul className="space-y-2 text-sm text-gray-600">
              <li>• 39-year-old single mother</li>
              <li>• School teacher seeking life insurance</li>
              <li>• Wants to protect her 8-year-old daughter</li>
            </ul>
            <div className="mt-3 pt-3 border-t border-gray-200">
              <div className="flex justify-between text-sm text-gray-600">
                <span>Progress</span>
                <span>{watchedTime} mins of {totalTime} mins</span>
              </div>
              <Progress value={progress} className="h-2" />
            </div>
          </div>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button 
                className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 text-lg rounded-lg shadow-md"
                onClick={handleStartRoleplay}
              >
                ▶️ Start Roleplay
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Practice selling life insurance to a single parent</p>
            </TooltipContent>
          </Tooltip>
        </div>
      </Card>
    </TooltipProvider>
  );
};

export default DailyMissionCard;
