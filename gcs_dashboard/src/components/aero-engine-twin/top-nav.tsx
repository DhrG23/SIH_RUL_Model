import React from 'react';
import {
  Menu,
  Search,
  Plus,
  Layers,
  History,
  Bell,
  Settings,
  BookOpen,
  User,
  Activity,
  Cpu
} from 'lucide-react';

interface TopNavProps {
  assetTitle?: string;
  isLiveSimulating: boolean;
  onToggleSimulation: () => void;
}

export const MiniSidebar: React.FC = () => {
  return (
    <aside className="w-12 bg-[#202020] border-r border-[#383838] flex flex-col items-center py-3 justify-between select-none z-30 shrink-0">
      <div className="flex flex-col items-center gap-5">
        <div className="w-8 h-8 rounded bg-[#292929] flex items-center justify-center border border-[#454545] text-gray-300 font-bold text-xs tracking-wider cursor-pointer hover:border-gray-300 transition-colors">
          <Cpu className="w-4 h-4 text-gray-300" />
        </div>
        <div className="w-full border-t border-[#333333] my-1" />
        <button title="Menu" className="p-2 text-gray-400 hover:text-white hover:bg-[#373737] rounded transition-colors">
          <Menu className="w-4 h-4" />
        </button>
        <button title="Search" className="p-2 text-gray-400 hover:text-white hover:bg-[#373737] rounded transition-colors">
          <Search className="w-4 h-4" />
        </button>
        <button title="Add Widget" className="p-2 text-gray-400 hover:text-white hover:bg-[#373737] rounded transition-colors">
          <Plus className="w-4 h-4" />
        </button>
        <button title="Layers / Views" className="p-2 text-gray-400 hover:text-white hover:bg-[#373737] rounded transition-colors">
          <Layers className="w-4 h-4" />
        </button>
        <button title="History & Logs" className="p-2 text-gray-400 hover:text-white hover:bg-[#373737] rounded transition-colors">
          <History className="w-4 h-4" />
        </button>
      </div>

      <div className="flex flex-col items-center gap-3">
        <span className="w-2 h-2 rounded-full bg-gray-300 animate-pulse" title="System Connected" />
      </div>
    </aside>
  );
};

export const TopHeader: React.FC<TopNavProps> = ({
  assetTitle = 'Aero Piston Engine Real-Time Health',
  isLiveSimulating,
  onToggleSimulation
}) => {
  return (
    <header className="h-14 bg-[#252525] border-b border-[#3b3b3b] flex items-center justify-between px-4 z-20 shrink-0">
      {/* Left Title & Brand */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <h1 className="text-sm font-semibold text-gray-200 tracking-wide">
            {assetTitle}
          </h1>
          <span className="text-[10px] uppercase tracking-wider px-2 py-0.5 rounded bg-[#303030] text-gray-300 border border-[#454545] font-mono">
            DRDO MALE UAV
          </span>
        </div>
      </div>

      {/* Right Actions & Utilities */}
      <div className="flex items-center gap-4">
        {/* Search Input */}
        <div className="relative hidden md:flex items-center">
          <Search className="w-3.5 h-3.5 absolute left-3 text-gray-400" />
          <input
            type="text"
            placeholder="Search all resources"
            className="bg-[#202020] text-xs text-gray-200 pl-8 pr-4 py-1.5 rounded border border-[#3d3d3d] focus:outline-none focus:border-[#707070] w-56 placeholder-gray-500 font-sans"
          />
        </div>

        {/* Utility Icons */}
        <div className="flex items-center gap-1.5 text-gray-400">
          <button title="Notifications" className="p-1.5 hover:text-white hover:bg-[#303030] rounded transition-colors relative">
            <Bell className="w-4 h-4" />
            <span className="absolute top-1 right-1 w-1.5 h-1.5 bg-gray-300 rounded-full" />
          </button>
          <button title="Settings" className="p-1.5 hover:text-white hover:bg-[#303030] rounded transition-colors">
            <Settings className="w-4 h-4" />
          </button>
          <button title="Documentation" className="p-1.5 hover:text-white hover:bg-[#303030] rounded transition-colors">
            <BookOpen className="w-4 h-4" />
          </button>
        </div>

        {/* User Profile */}
        <div className="flex items-center gap-2 border-l border-[#3d3d3d] pl-3">
          <div className="text-right hidden xl:block">
            <div className="text-xs font-medium text-gray-200 leading-none">Kirsten Schwarzer</div>
            <div className="text-[10px] text-gray-500 font-mono leading-tight mt-0.5">L1 TFUER</div>
          </div>
          <div className="w-7 h-7 rounded-full bg-[#2c333f] border border-[#3b4454] flex items-center justify-center text-gray-300">
            <User className="w-4 h-4" />
          </div>
        </div>

        {/* Simulation Toggle / View Telemetry Action */}
        <button
          onClick={onToggleSimulation}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-semibold tracking-wide transition-all ${
            isLiveSimulating
              ? 'bg-[#5a5a5a] hover:bg-[#6a6a6a] text-white shadow-lg shadow-black/30'
              : 'bg-[#303030] hover:bg-[#3a3a3a] text-gray-300 border border-[#454545]'
          }`}
        >
          <Activity className={`w-3.5 h-3.5 ${isLiveSimulating ? 'animate-pulse' : ''}`} />
          <span>{isLiveSimulating ? 'Live Telemetry Active' : 'View Telemetry'}</span>
        </button>
      </div>
    </header>
  );
};
