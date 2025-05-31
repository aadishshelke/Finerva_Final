import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import SalesPath from './SalesPath';
import ModuleCard from './ModuleCard';
import LessonNode from './LessonNode';
import Checkpoint from './Checkpoint';
import XPStreakBar from './XPStreakBar';
import SurpriseBox from './SurpriseBox';
import Quiz from './Quiz';
import { useToast } from '../hooks/use-toast';
import { useLocation } from 'react-router-dom';

interface Lesson {
  id: number;
  title: string;
  stars: number;
  type: 'lesson' | 'quiz' | 'roleplay';
  locked: boolean;
}

interface Module {
  id: number;
  title: string;
  stars: number;
  completed: boolean;
  lessons: Lesson[];
}

const LearningModules: React.FC = () => {
  const { toast } = useToast();
  const location = useLocation();
  const [activeTab, setActiveTab] = useState<'modules' | 'quiz'>('modules');
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [showQuiz, setShowQuiz] = useState(false);
  const [userStats, setUserStats] = useState({
    xp: 1250,
    maxXP: 2000,
    streak: 7,
    dailyGoal: 5,
    dailyProgress: 3
  });

  useEffect(() => {
    const saved = localStorage.getItem('sidebarCollapsed');
    setIsSidebarCollapsed(saved ? JSON.parse(saved) : false);
  }, []);

  useEffect(() => {
    // Check if we should auto-start the quiz
    if (location.state?.autoStartQuiz) {
      setShowQuiz(true);
    }
  }, [location.state]);

  const [modules, setModules] = useState<Module[]>([
    {
      id: 1,
      title: "Module 1: Health Insurance 101",
      stars: 3,
      completed: true,
      lessons: [
        { id: 101, title: "What is Insurance?", stars: 2, type: "lesson", locked: false },
        { id: 102, title: "Types of Health Policies", stars: 3, type: "lesson", locked: false },
        { id: 103, title: "Match Scenarios", stars: 1, type: "quiz", locked: false },
      ],
    },
    {
      id: 2,
      title: "Module 2: Customer Personas",
      stars: 2,
      completed: true,
      lessons: [
        { id: 201, title: "Identifying Personas", stars: 3, type: "lesson", locked: false },
        { id: 202, title: "Confused Customers", stars: 2, type: "lesson", locked: false },
        { id: 203, title: "Roleplay Level", stars: 1, type: "roleplay", locked: false },
      ],
    },
    {
      id: 3,
      title: "Module 3: Sales Techniques",
      stars: 1,
      completed: false,
      lessons: [
        { id: 301, title: "Building Rapport", stars: 2, type: "lesson", locked: false },
        { id: 302, title: "Handling Objections", stars: 0, type: "lesson", locked: true },
        { id: 303, title: "Closing Techniques", stars: 0, type: "quiz", locked: true },
        { id: 304, title: "Advanced Roleplay", stars: 0, type: "roleplay", locked: true },
      ],
    },
    {
      id: 4,
      title: "Module 4: Advanced Strategies",
      stars: 0,
      completed: false,
      lessons: [
        { id: 401, title: "Cross-selling Tactics", stars: 0, type: "lesson", locked: true },
        { id: 402, title: "Digital Sales Tools", stars: 0, type: "lesson", locked: true },
        { id: 403, title: "Master Challenge", stars: 0, type: "roleplay", locked: true },
      ],
    },
  ]);

  const [checkpoints, setCheckpoints] = useState([
    { id: 1, title: "Checkpoint 1: Basics Mastery", unlocked: true, completed: true },
    { id: 2, title: "Checkpoint 2: Sales Pro", unlocked: false, completed: false },
  ]);

  const [surpriseBoxes, setSurpriseBoxes] = useState([
    { id: 1, opened: false },
    { id: 2, opened: true },
  ]);

  const handleLessonClick = (lessonId: number) => {
    toast({
      title: "Lesson Started! 🎯",
      description: `Starting lesson ${lessonId}. Good luck!`,
    });

    // Simulate lesson completion with XP gain
    setTimeout(() => {
      setUserStats(prev => ({
        ...prev,
        xp: prev.xp + 25,
        dailyProgress: Math.min(prev.dailyProgress + 1, prev.dailyGoal)
      }));
      
      toast({
        title: "Lesson Completed! ⭐",
        description: "+25 XP earned! Keep going!",
      });
    }, 2000);
  };

  const handleCheckpointClick = (checkpointId: number) => {
    toast({
      title: "Checkpoint Challenge! 🏁",
      description: "Ready for the big test? Let's see what you've learned!",
    });
  };

  const handleSurpriseBoxOpen = (boxId: number) => {
    setSurpriseBoxes(prev => 
      prev.map(box => 
        box.id === boxId ? { ...box, opened: true } : box
      )
    );

    setUserStats(prev => ({
      ...prev,
      xp: prev.xp + 50
    }));

    toast({
      title: "Surprise Reward! 🎁",
      description: "+50 XP bonus! You're on fire!",
    });
  };

  // Check if checkpoint should be unlocked
  const shouldUnlockCheckpoint = (checkpointIndex: number) => {
    const modulesNeeded = (checkpointIndex + 1) * 2;
    const completedModules = modules.slice(0, modulesNeeded).filter(m => m.completed).length;
    return completedModules >= modulesNeeded;
  };

  const handleQuizComplete = (score: number) => {
    setUserStats(prev => ({
      ...prev,
      xp: prev.xp + Math.floor(score / 10) * 25, // Award XP based on score
      dailyProgress: Math.min(prev.dailyProgress + 1, prev.dailyGoal)
    }));

    toast({
      title: "Quiz Completed! 🎯",
      description: `You earned ${Math.floor(score / 10) * 25} XP! Keep up the great work!`,
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50">
      <XPStreakBar
        xp={userStats.xp}
        maxXP={userStats.maxXP}
        streak={userStats.streak}
        dailyGoal={userStats.dailyGoal}
        dailyProgress={userStats.dailyProgress}
        isSidebarCollapsed={isSidebarCollapsed}
      />

      <div className="container mx-auto px-4 py-8">
        <div className="text-center mb-8 pt-16">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-4">
            Sales Mastery Journey
          </h1>
          <p className="text-lg text-gray-600">
            Master the art of sales through interactive lessons, quizzes, and roleplay scenarios
          </p>
        </div>

        {/* Tab Navigation */}
        <div className="flex justify-center mb-8">
          <div className="inline-flex rounded-lg border border-gray-200 bg-white p-1">
            <button
              onClick={() => setActiveTab('modules')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                activeTab === 'modules'
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              Learning Modules
            </button>
            <button
              onClick={() => setActiveTab('quiz')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                activeTab === 'quiz'
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              Knowledge Quiz
            </button>
          </div>
        </div>

        {/* Content Area */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          {activeTab === 'modules' ? (
            <div className="relative">
              <SalesPath>
                {/* Module 1 */}
                <ModuleCard
                  title={modules[0].title}
                  stars={modules[0].stars}
                  locked={false}
                  completed={modules[0].completed}
                  lessons={modules[0].lessons}
                  onLessonClick={handleLessonClick}
                />

                {/* Module 2 */}
                <ModuleCard
                  title={modules[1].title}
                  stars={modules[1].stars}
                  locked={false}
                  completed={modules[1].completed}
                  lessons={modules[1].lessons}
                  onLessonClick={handleLessonClick}
                />

                {/* Checkpoint 1 */}
                <Checkpoint
                  title={checkpoints[0].title}
                  unlocked={checkpoints[0].unlocked}
                  completed={checkpoints[0].completed}
                  onClick={() => handleCheckpointClick(1)}
                />

                {/* Surprise Box 1 */}
                <SurpriseBox
                  opened={surpriseBoxes[0].opened}
                  onOpen={() => handleSurpriseBoxOpen(1)}
                />

                {/* Module 3 */}
                <ModuleCard
                  title={modules[2].title}
                  stars={modules[2].stars}
                  locked={!modules[1].completed}
                  completed={modules[2].completed}
                  lessons={modules[2].lessons}
                  onLessonClick={handleLessonClick}
                />

                {/* Module 4 */}
                <ModuleCard
                  title={modules[3].title}
                  stars={modules[3].stars}
                  locked={!modules[2].completed}
                  completed={modules[3].completed}
                  lessons={modules[3].lessons}
                  onLessonClick={handleLessonClick}
                />

                {/* Checkpoint 2 */}
                <Checkpoint
                  title={checkpoints[1].title}
                  unlocked={shouldUnlockCheckpoint(1)}
                  completed={checkpoints[1].completed}
                  onClick={() => handleCheckpointClick(2)}
                />

                {/* Surprise Box 2 */}
                <SurpriseBox
                  opened={surpriseBoxes[1].opened}
                  onOpen={() => handleSurpriseBoxOpen(2)}
                />
              </SalesPath>
            </div>
          ) : (
            <Quiz onComplete={handleQuizComplete} />
          )}
        </motion.div>

        {/* Achievement Summary */}
        <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto">
          <div className="bg-white rounded-xl p-6 shadow-lg border border-blue-200">
            <h3 className="font-semibold text-gray-800 mb-2">Progress Summary</h3>
            <p className="text-sm text-gray-600">
              {modules.filter(m => m.completed).length}/{modules.length} modules completed
            </p>
            <p className="text-sm text-gray-600">
              {modules.reduce((acc, m) => acc + m.lessons.filter(l => l.stars > 0).length, 0)} lessons mastered
            </p>
          </div>

          <div className="bg-white rounded-xl p-6 shadow-lg border border-purple-200">
            <h3 className="font-semibold text-gray-800 mb-2">Achievements</h3>
            <p className="text-sm text-gray-600">🏆 Quick Learner</p>
            <p className="text-sm text-gray-600">🔥 {userStats.streak} Day Streak</p>
          </div>

          <div className="bg-white rounded-xl p-6 shadow-lg border border-green-200">
            <h3 className="font-semibold text-gray-800 mb-2">Next Goal</h3>
            <p className="text-sm text-gray-600">
              Complete Module 3 to unlock advanced techniques
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LearningModules; 