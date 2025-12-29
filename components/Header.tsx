import React from 'react';
import { Radio, Waves } from 'lucide-react';

const Header: React.FC = () => {
  return (
    <header className="w-full py-6 px-4 flex flex-col items-center justify-center border-b border-gray-100 bg-white/80 backdrop-blur-xl sticky top-0 z-50 transition-all duration-300">
      <div className="flex items-center gap-3 mb-2">
        <div className="relative">
          <div className="bg-rose-500 rounded-lg p-1.5 shadow-lg shadow-rose-200">
             <Radio className="w-6 h-6 text-white" />
          </div>
          <Waves className="w-4 h-4 text-rose-400 absolute -top-2 -right-2 animate-pulse" />
        </div>
        <h1 className="text-2xl font-bold text-gray-900 tracking-tight">
          PodcastHunter & AudioWriter
        </h1>
      </div>
      <p className="text-gray-500 text-sm max-w-md text-center font-medium">
        Unlock secret PODCAST feeds. Paste a LINK to let the AI extract hidden audio.
      </p>
    </header>
  );
};

export default Header;