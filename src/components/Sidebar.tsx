import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { 
  Home, 
  BookOpen, 
  Users, 
  Mic, 
  Calendar, 
  Trophy, 
  FileText, 
  BarChart, 
  Settings,
  ChevronLeft,
  ChevronRight
} from 'lucide-react';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';

interface SidebarItem {
  icon: React.ElementType;
  label: string;
  path: string;
}

const sidebarItems: SidebarItem[] = [
  { icon: Home, label: 'Dashboard', path: '/dashboard' },
  { icon: Users, label: 'Roleplay Practice', path: '/roleplay' },
  { icon: BookOpen, label: 'Learning Modules', path: '/learning-modules' },
  { icon: Mic, label: 'Voice Chat Coach', path: '/voice-chat' },
  { icon: Calendar, label: 'Daily Learning', path: '/daily' },
  { icon: Trophy, label: 'Achievements', path: '/achievements' },
  { icon: FileText, label: 'Saved Scripts', path: '/scripts' },
  { icon: BarChart, label: 'Analytics & Reports', path: '/analytics' },
  { icon: Settings, label: 'Settings', path: '/settings' },
];

const Sidebar = () => {
  const [isCollapsed, setIsCollapsed] = useState(() => {
    const saved = localStorage.getItem('sidebarCollapsed');
    return saved ? JSON.parse(saved) : false;
  });
  
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    localStorage.setItem('sidebarCollapsed', JSON.stringify(isCollapsed));
  }, [isCollapsed]);

  const toggleSidebar = () => {
    setIsCollapsed(!isCollapsed);
  };

  return (
    <div 
      className={`fixed left-0 top-0 h-screen bg-slate-800 text-white transition-all duration-300 ${
        isCollapsed ? 'w-[70px]' : 'w-[250px]'
      }`}
    >
      <div className="flex flex-col h-full">
        {/* Logo/Brand */}
        <div className="h-16 flex items-center justify-center border-b border-slate-700">
          {!isCollapsed && (
            <span className="text-xl font-bold">
              <span style={{ color: '#00bd87' }}>Gro</span>
              <span style={{ color: '#268bf0' }}>Mo</span>
            </span>
          )}
        </div>

        {/* Navigation Items */}
        <nav className="flex-1 py-4">
          {sidebarItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;
            
            return (
              <Tooltip key={item.path}>
                <TooltipTrigger asChild>
                  <button
                    onClick={() => navigate(item.path)}
                    className={`w-full flex items-center px-4 py-3 transition-colors duration-200 ${
                      isActive 
                        ? 'bg-blue-600 text-white' 
                        : 'text-gray-300 hover:bg-slate-700 hover:text-white'
                    }`}
                  >
                    <Icon className="w-6 h-6" />
                    {!isCollapsed && (
                      <span className="ml-3">{item.label}</span>
                    )}
                  </button>
                </TooltipTrigger>
                {isCollapsed && <TooltipContent>{item.label}</TooltipContent>}
              </Tooltip>
            );
          })}
        </nav>

        {/* Collapse Toggle */}
        <button
          onClick={toggleSidebar}
          className="p-4 border-t border-slate-700 hover:bg-slate-700 transition-colors duration-200"
        >
          {isCollapsed ? (
            <ChevronRight className="w-6 h-6" />
          ) : (
            <ChevronLeft className="w-6 h-6" />
          )}
        </button>
      </div>
    </div>
  );
};

export default Sidebar; 