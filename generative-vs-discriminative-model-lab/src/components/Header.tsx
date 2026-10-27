import React from 'react';
import { Download } from 'lucide-react';

interface HeaderProps {
  activeTab: 'lab' | 'theory' | 'code' | 'checklist';
  setActiveTab: (tab: 'lab' | 'theory' | 'code' | 'checklist') => void;
  onDownloadZip: () => void;
  isDownloading: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  setActiveTab,
  onDownloadZip,
  isDownloading,
}) => {
  return (
    <header className="sticky top-0 z-40 bg-slate-950/90 backdrop-blur-md border-b border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-8">
        {/* Zone 1: Brand wordmark (single element, no subtitle clutter) */}
        <div className="flex items-center gap-3 shrink-0">
          <span className="text-xl font-bold tracking-tight text-white whitespace-nowrap">
            Generative vs Discriminative Lab
          </span>
          <span className="hidden sm:inline text-xs font-mono text-slate-400 border border-slate-800 px-2 py-0.5 rounded">
            MNIST · PyTorch · Streamlit
          </span>
        </div>

        {/* Zone 2: Navigation links */}
        <nav className="hidden md:flex items-center gap-6 text-sm font-medium">
          <button
            onClick={() => setActiveTab('lab')}
            className={`whitespace-nowrap transition-colors pb-1 border-b-2 ${
              activeTab === 'lab'
                ? 'text-sky-400 border-sky-500 font-semibold'
                : 'text-slate-400 border-transparent hover:text-slate-200'
            }`}
          >
            Interactive Lab
          </button>
          <button
            onClick={() => setActiveTab('theory')}
            className={`whitespace-nowrap transition-colors pb-1 border-b-2 ${
              activeTab === 'theory'
                ? 'text-sky-400 border-sky-500 font-semibold'
                : 'text-slate-400 border-transparent hover:text-slate-200'
            }`}
          >
            Theory & Metrics
          </button>
          <button
            onClick={() => setActiveTab('code')}
            className={`whitespace-nowrap transition-colors pb-1 border-b-2 ${
              activeTab === 'code'
                ? 'text-sky-400 border-sky-500 font-semibold'
                : 'text-slate-400 border-transparent hover:text-slate-200'
            }`}
          >
            Python Project Code
          </button>
          <button
            onClick={() => setActiveTab('checklist')}
            className={`whitespace-nowrap transition-colors pb-1 border-b-2 ${
              activeTab === 'checklist'
                ? 'text-sky-400 border-sky-500 font-semibold'
                : 'text-slate-400 border-transparent hover:text-slate-200'
            }`}
          >
            Exam Checklist
          </button>
        </nav>

        {/* Zone 3: Primary Action CTA */}
        <div className="flex items-center gap-3 shrink-0">
          <button
            onClick={onDownloadZip}
            disabled={isDownloading}
            className="flex items-center gap-2 px-4 py-2 text-xs font-semibold text-slate-900 bg-sky-400 hover:bg-sky-300 disabled:opacity-50 transition-colors rounded-lg shadow-sm whitespace-nowrap cursor-pointer"
            title="Download the full Python project as a zip archive"
          >
            <Download className="w-3.5 h-3.5" />
            <span>{isDownloading ? 'Bundling...' : 'Download Project (.zip)'}</span>
          </button>
        </div>
      </div>
    </header>
  );
};
