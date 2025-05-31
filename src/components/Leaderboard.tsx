import React from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { useNavigate } from 'react-router-dom';

const Leaderboard = () => {
  const navigate = useNavigate();
  const topUsers = [
    { rank: 1, name: "Priya Sharma", xp: 1250, badge: "🥇" },
    { rank: 2, name: "Rajesh Kumar", xp: 1180, badge: "🥈" },
    { rank: 3, name: "Anjali Patel", xp: 1050, badge: "🥉" },
    { rank: 4, name: "Vikram Singh", xp: 980, badge: "" },
    { rank: 5, name: "Sunita Reddy", xp: 920, badge: "" },
  ];

  return (
    <Card 
      className="p-6 space-y-4 card-hover cursor-pointer transition-all duration-300 hover:shadow-lg"
      onClick={() => navigate('/achievements')}
    >
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold text-gray-800">📊 Regional Leaderboard</h2>
        <Badge variant="outline" className="text-gromo-primary border-gromo-primary">
          Pune Region
        </Badge>
      </div>

      <div className="space-y-3">
        {topUsers.map((user) => (
          <div key={user.rank} className={`flex items-center justify-between p-3 rounded-lg transition-all duration-300 ${
            user.rank <= 3 
              ? 'bg-gradient-to-r from-gromo-accent to-yellow-400 text-white' 
              : 'bg-gray-50 hover:bg-gray-100'
          }`}>
            <div className="flex items-center gap-3">
              <span className="text-2xl font-bold">{user.badge || `#${user.rank}`}</span>
              <div>
                <div className="font-semibold">{user.name}</div>
                <div className={`text-sm ${user.rank <= 3 ? 'text-white opacity-90' : 'text-gray-500'}`}>
                  Rank #{user.rank}
                </div>
              </div>
            </div>
            <div className="text-right">
              <div className="font-bold text-lg">{user.xp}</div>
              <div className={`text-sm ${user.rank <= 3 ? 'text-white opacity-90' : 'text-gray-500'}`}>
                XP
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="bg-gromo-primary text-white p-3 rounded-lg text-center">
        <div className="font-semibold">Your Rank: #12</div>
        <div className="text-sm opacity-90">360 XP • Keep learning to climb!</div>
      </div>
      
      <div className="text-center text-sm text-gray-600">
        💪 300 partners learned today
      </div>
    </Card>
  );
};

export default Leaderboard;
