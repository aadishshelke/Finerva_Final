import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { TooltipProvider } from '@/components/ui/tooltip';
import MainLayout from '@/components/MainLayout';
import Dashboard from '@/pages/Dashboard';
import Achievements from '@/pages/Achievements';
import RoleplayPractice from './components/RoleplayPractice';
import VoiceChatCoach from './components/VoiceChatCoach';
// import DailyLearning from './components/DailyLearning';
import { SavedScripts } from './components/SavedScripts';
import { Settings } from './components/Settings';
import { Analytics } from './components/Analytics';
import LearningModules from './components/LearningModules';

const App = () => {
  return (
    <TooltipProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<MainLayout />}>
            <Route index element={<Navigate to="/dashboard" replace />} />
            <Route path="dashboard" element={<Dashboard />} />
            <Route path="roleplay" element={<RoleplayPractice />} />
            <Route path="voice-chat" element={<VoiceChatCoach />} />
            {/* <Route path="daily" element={<DailyLearning />} /> */}
            <Route path="achievements" element={<Achievements />} />
            <Route path="scripts" element={<SavedScripts />} />
            <Route path="analytics" element={<Analytics />} />
            <Route path="settings" element={<Settings />} />
            <Route path="learning-modules" element={<LearningModules />} />
            <Route path="*" element={<div className="p-6">404 - Page Not Found</div>} />
          </Route>
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  );
};

export default App;
