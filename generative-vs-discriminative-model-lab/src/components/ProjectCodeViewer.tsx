import React, { useState } from 'react';
import { PYTHON_PROJECT_FILES, ProjectFile } from '../data/projectFiles';
import { Copy, Check, Download, FileCode, FolderArchive, Terminal } from 'lucide-react';
import JSZip from 'jszip';

interface ProjectCodeViewerProps {
  onDownloadZip: () => void;
  isDownloading: boolean;
}

export const ProjectCodeViewer: React.FC<ProjectCodeViewerProps> = ({
  onDownloadZip,
  isDownloading,
}) => {
  const [selectedFile, setSelectedFile] = useState<ProjectFile>(PYTHON_PROJECT_FILES[0]);
  const [copied, setCopied] = useState<boolean>(false);

  const copyToClipboard = () => {
    navigator.clipboard.writeText(selectedFile.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const lines = selectedFile.content.split('\n');

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-5 bg-slate-900 border border-slate-800 rounded-xl">
        <div>
          <div className="flex items-center gap-2">
            <FolderArchive className="w-5 h-5 text-sky-400" />
            <h2 className="text-base font-bold text-white">
              Full Python & PyTorch Project Codebase
            </h2>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Modular architecture: <code className="text-sky-300">preprocessing.py</code>, <code className="text-sky-300">models.py</code>, <code className="text-sky-300">train.py</code>, <code className="text-sky-300">inference.py</code>, <code className="text-sky-300">app.py</code>.
          </p>
        </div>

        <button
          onClick={onDownloadZip}
          disabled={isDownloading}
          className="flex items-center justify-center gap-2 px-4 py-2.5 text-xs font-semibold text-slate-950 bg-sky-400 hover:bg-sky-300 disabled:opacity-50 transition-colors rounded-lg shadow-sm whitespace-nowrap cursor-pointer shrink-0"
        >
          <Download className="w-4 h-4" />
          <span>{isDownloading ? 'Packaging .zip...' : 'Download Project (.zip)'}</span>
        </button>
      </div>

      {/* Code Viewer Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 items-start">
        {/* File Selector Sidebar */}
        <div className="lg:col-span-1 bg-slate-900 border border-slate-800 rounded-xl p-3 space-y-1">
          <div className="text-[11px] font-semibold text-slate-400 px-3 py-1.5 uppercase tracking-wider">
            Files ({PYTHON_PROJECT_FILES.length})
          </div>

          {PYTHON_PROJECT_FILES.map((file) => {
            const isSelected = file.name === selectedFile.name;
            return (
              <button
                key={file.name}
                onClick={() => setSelectedFile(file)}
                className={`w-full flex items-center justify-between px-3 py-2 text-xs rounded-lg transition-colors text-left cursor-pointer ${
                  isSelected
                    ? 'bg-sky-950/80 text-sky-300 font-semibold border border-sky-800/60'
                    : 'text-slate-400 hover:bg-slate-800/60 hover:text-slate-200'
                }`}
              >
                <div className="flex items-center gap-2 truncate">
                  <FileCode className={`w-3.5 h-3.5 shrink-0 ${isSelected ? 'text-sky-400' : 'text-slate-500'}`} />
                  <span className="font-mono truncate">{file.name}</span>
                </div>
                <span className="text-[10px] font-mono text-slate-500 uppercase shrink-0">
                  {file.language}
                </span>
              </button>
            );
          })}

          <div className="pt-4 mt-2 border-t border-slate-800 text-[11px] text-slate-400 px-3 space-y-2">
            <div className="flex items-center gap-1.5 font-medium text-slate-300">
              <Terminal className="w-3.5 h-3.5 text-sky-400" />
              <span>Execution Steps:</span>
            </div>
            <ol className="list-decimal list-inside space-y-1 text-slate-400 pl-1 font-mono text-[10px]">
              <li>pip install -r requirements.txt</li>
              <li>python train.py</li>
              <li>streamlit run app.py</li>
            </ol>
          </div>
        </div>

        {/* Code View Area */}
        <div className="lg:col-span-3 bg-slate-950 border border-slate-800 rounded-xl overflow-hidden shadow-2xl">
          {/* File Header Bar */}
          <div className="flex items-center justify-between px-4 py-3 bg-slate-900/90 border-b border-slate-800">
            <div className="flex items-center gap-3">
              <span className="font-mono text-xs font-semibold text-slate-200">
                {selectedFile.path}
              </span>
              <span className="hidden sm:inline text-xs text-slate-500">
                {selectedFile.description}
              </span>
            </div>

            <button
              onClick={copyToClipboard}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700 rounded-md transition-colors cursor-pointer"
            >
              {copied ? (
                <>
                  <Check className="w-3.5 h-3.5 text-emerald-400" />
                  <span className="text-emerald-400 font-medium">Copied!</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5 text-slate-400" />
                  <span>Copy File</span>
                </>
              )}
            </button>
          </div>

          {/* Line by line code container */}
          <div className="p-4 overflow-x-auto max-h-[640px] text-xs font-mono leading-relaxed bg-[#060913]">
            <table className="w-full border-collapse">
              <tbody>
                {lines.map((line, idx) => (
                  <tr key={idx} className="hover:bg-slate-900/40">
                    <td className="w-10 pr-4 text-right text-slate-600 select-none align-top tabular-nums text-[11px]">
                      {idx + 1}
                    </td>
                    <td className="text-slate-200 whitespace-pre font-mono">
                      {line}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};
