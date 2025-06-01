/*
import React from 'react';
import { Calendar, Clock, Award, CheckCircle } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

export const DailyLearning: React.FC = () => {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Daily Learning</h1>
        <p className="text-gray-600 mt-1">Stay consistent with bite-sized daily lessons and challenges.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <Card className="p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Today's Challenge</h3>
            
            <div className="p-6 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg text-white mb-4">
              <h4 className="text-xl font-bold mb-2">Insurance Product Knowledge Quiz</h4>
              <p className="text-blue-100 mb-4">Test your understanding of term life insurance benefits and features.</p>
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <Clock className="w-4 h-4 mr-2" />
                  <span>5 minutes</span>
                </div>
                <div className="flex items-center">
                  <Award className="w-4 h-4 mr-2" />
                  <span>50 XP</span>
                </div>
              </div>
            </div>
            
            <Button className="w-full">
              Start Today's Challenge
            </Button>
          </Card>
        </div>

        <Card className="p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Weekly Progress</h3>
          
          <div className="space-y-3">
            {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'].map((day, index) => (
              <div key={day} className="flex items-center justify-between">
                <span className="text-gray-600">{day}</span>
                {index < 4 ? (
                  <CheckCircle className="w-5 h-5 text-green-500" />
                ) : (
                  <div className="w-5 h-5 border-2 border-gray-300 rounded-full"></div>
                )}
              </div>
            ))}
          </div>
          
          <div className="mt-4 pt-4 border-t border-gray-200">
            <div className="text-center">
              <p className="text-2xl font-bold text-green-600">4/7</p>
              <p className="text-sm text-gray-600">Days completed</p>
            </div>
          </div>
        </Card>
      </div>

      <Card className="p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Learning Streak</h3>
        
        <div className="grid grid-cols-7 gap-2">
          {Array.from({ length: 21 }, (_, i) => (
            <div
              key={i}
              className={`aspect-square rounded-sm ${
                i < 15 ? 'bg-green-500' : i < 18 ? 'bg-green-300' : 'bg-gray-200'
              }`}
            />
          ))}
        </div>
        
        <div className="flex items-center justify-between mt-4">
          <div>
            <p className="text-2xl font-bold text-gray-900">18</p>
            <p className="text-sm text-gray-600">Day streak</p>
          </div>
          <div className="text-right">
            <p className="text-lg font-semibold text-gray-900">Keep it up!</p>
            <p className="text-sm text-gray-600">You're on fire 🔥</p>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default DailyLearning;
*/ 