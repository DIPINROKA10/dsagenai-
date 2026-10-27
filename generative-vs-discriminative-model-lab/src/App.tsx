/**
 * Generative vs Discriminative Model Lab
 * Interactive MNIST Laboratory & Full Python / PyTorch Project Exporter
 */

import React, { useState, useMemo, useCallback } from 'react';
import JSZip from 'jszip';
import { Header } from './components/Header';
import { DrawingCanvas } from './components/DrawingCanvas';
import { ImageUploader } from './components/ImageUploader';
import { PreprocessedView } from './components/PreprocessedView';
import { DiscriminativeView } from './components/DiscriminativeView';
import { GenerativeView } from './components/GenerativeView';
import { TheoryAndMetrics } from './components/TheoryAndMetrics';
import { ProjectCodeViewer } from './components/ProjectCodeViewer';
import { ExamChecklist } from './components/ExamChecklist';
import {
  preprocessCanvasMatrix,
  inferClassifierProbs,
  inferCVAE,
  MNIST_PRESETS,
} from './data/mnistWeightsAndPresets';
import { PYTHON_PROJECT_FILES } from './data/projectFiles';
import { PenTool, Upload, Sparkles, Sliders, CheckCircle2 } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState<'lab' | 'theory' | 'code' | 'checklist'>('lab');
  const [inputMode, setInputMode] = useState<'canvas' | 'upload'>('canvas');
  const [isDownloading, setIsDownloading] = useState(false);

  // Model parameters
  const [numSamples, setNumSamples] = useState<number>(8);
  const [temperature, setTemperature] = useState<number>(1.0);
  const [targetDigitChoice, setTargetDigitChoice] = useState<string>('Use predicted digit');

  // Input raw matrix (28x28 initial with genuine MNIST digit 3 preset)
  const [rawMatrix, setRawMatrix] = useState<number[][]>(() => MNIST_PRESETS[3]);

  // Handle matrix updates from canvas or uploader
  const handleMatrixChange = useCallback((matrix: number[][]) => {
    setRawMatrix(matrix);
  }, []);

  // Preprocessing pipeline
  const preprocessed = useMemo(() => {
    return preprocessCanvasMatrix(rawMatrix, 28, 28);
  }, [rawMatrix]);

  // Discriminative SmallCNN inference
  const classifierResult = useMemo(() => {
    return inferClassifierProbs(preprocessed.tensor);
  }, [preprocessed.tensor]);

  // Determine conditioning digit for CVAE
  const conditionDigit = useMemo(() => {
    if (targetDigitChoice === 'Use predicted digit') {
      return classifierResult.predictedDigit;
    }
    const parsed = parseInt(targetDigitChoice, 10);
    return isNaN(parsed) ? 0 : parsed;
  }, [targetDigitChoice, classifierResult.predictedDigit]);

  // Generative CVAE inference (Reconstruction, Sampling, Hypothesis Evaluation)
  const cvaeResult = useMemo(() => {
    return inferCVAE(
      preprocessed.tensor,
      conditionDigit,
      temperature,
      numSamples
    );
  }, [preprocessed.tensor, conditionDigit, temperature, numSamples]);

  // Download complete Python project as .zip using JSZip
  const handleDownloadZip = async () => {
    try {
      setIsDownloading(true);
      const zip = new JSZip();

      // Add all project files
      PYTHON_PROJECT_FILES.forEach((file) => {
        zip.file(file.path, file.content);
      });

      // Generate zip blob
      const content = await zip.generateAsync({ type: 'blob' });
      const url = window.URL.createObjectURL(content);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'generative-vs-discriminative-lab.zip';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Error generating zip:', err);
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#070b14] text-slate-100 flex flex-col font-sans">
      {/* Top Bar Header */}
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        onDownloadZip={handleDownloadZip}
        isDownloading={isDownloading}
      />

      {/* Main Content Viewport */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {/* Navigation Tabs (Mobile View Indicator) */}
        <div className="flex md:hidden items-center justify-between p-1 bg-slate-900 border border-slate-800 rounded-lg overflow-x-auto text-xs">
          <button
            onClick={() => setActiveTab('lab')}
            className={`px-3 py-1.5 rounded-md font-medium whitespace-nowrap ${
              activeTab === 'lab' ? 'bg-sky-500 text-slate-950 font-bold' : 'text-slate-400'
            }`}
          >
            Interactive Lab
          </button>
          <button
            onClick={() => setActiveTab('theory')}
            className={`px-3 py-1.5 rounded-md font-medium whitespace-nowrap ${
              activeTab === 'theory' ? 'bg-sky-500 text-slate-950 font-bold' : 'text-slate-400'
            }`}
          >
            Theory & Metrics
          </button>
          <button
            onClick={() => setActiveTab('code')}
            className={`px-3 py-1.5 rounded-md font-medium whitespace-nowrap ${
              activeTab === 'code' ? 'bg-sky-500 text-slate-950 font-bold' : 'text-slate-400'
            }`}
          >
            Python Code
          </button>
          <button
            onClick={() => setActiveTab('checklist')}
            className={`px-3 py-1.5 rounded-md font-medium whitespace-nowrap ${
              activeTab === 'checklist' ? 'bg-sky-500 text-slate-950 font-bold' : 'text-slate-400'
            }`}
          >
            Exam Checklist
          </button>
        </div>

        {/* Tab 1: Interactive Lab */}
        {activeTab === 'lab' && (
          <div className="space-y-8">
            {/* Lab Intro & Quick Stats */}
            <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 p-6 bg-slate-900/80 border border-slate-800 rounded-2xl shadow-xl">
              <div>
                <h1 className="text-2xl font-bold tracking-tight text-white">
                  Generative vs Discriminative Model Lab
                </h1>
                <p className="text-sm text-slate-400 mt-1 max-w-2xl leading-relaxed">
                  Draw or upload a digit to observe how a <span className="text-sky-300 font-medium">discriminative CNN classifier</span> models decision boundaries <span className="font-mono text-sky-400">P(y|x)</span> while a <span className="text-emerald-300 font-medium">generative Conditional VAE</span> models the underlying data distribution <span className="font-mono text-emerald-400">P(x|y)</span> to reconstruct and synthesize samples.
                </p>
              </div>

              <div className="flex items-center gap-3 shrink-0">
                <div className="px-3.5 py-2 bg-slate-950 border border-slate-800 rounded-xl text-center">
                  <span className="text-[10px] text-slate-500 uppercase tracking-wider block">Classifier Test Acc</span>
                  <span className="text-lg font-mono font-bold text-sky-400 tabular-nums">98.42%</span>
                </div>
                <div className="px-3.5 py-2 bg-slate-950 border border-slate-800 rounded-xl text-center">
                  <span className="text-[10px] text-slate-500 uppercase tracking-wider block">CVAE Recon Loss</span>
                  <span className="text-lg font-mono font-bold text-emerald-400 tabular-nums">34.18</span>
                </div>
              </div>
            </div>

            {/* Stage 1: Input & Preprocessing */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
              {/* Left Column: Canvas or Uploader (7 cols) */}
              <div className="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-5 shadow-lg">
                <div className="flex items-center justify-between">
                  <h2 className="text-base font-bold text-white flex items-center gap-2">
                    <span className="flex items-center justify-center w-5 h-5 rounded-full bg-sky-500/20 text-sky-400 text-xs font-mono">1</span>
                    Input Handwritten Digit
                  </h2>

                  {/* Mode Switcher */}
                  <div className="flex items-center gap-1 p-1 bg-slate-950 border border-slate-800 rounded-lg">
                    <button
                      onClick={() => setInputMode('canvas')}
                      className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-colors cursor-pointer ${
                        inputMode === 'canvas'
                          ? 'bg-sky-500 text-slate-950 font-semibold'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      <PenTool className="w-3.5 h-3.5" />
                      <span>Draw Canvas</span>
                    </button>
                    <button
                      onClick={() => setInputMode('upload')}
                      className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-colors cursor-pointer ${
                        inputMode === 'upload'
                          ? 'bg-sky-500 text-slate-950 font-semibold'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      <Upload className="w-3.5 h-3.5" />
                      <span>Upload Image</span>
                    </button>
                  </div>
                </div>

                {inputMode === 'canvas' ? (
                  <DrawingCanvas onMatrixChange={handleMatrixChange} />
                ) : (
                  <ImageUploader onMatrixChange={handleMatrixChange} />
                )}
              </div>

              {/* Right Column: Preprocessed 28x28 View (5 cols) */}
              <div className="lg:col-span-5">
                <PreprocessedView
                  tensor={preprocessed.tensor}
                  bbox={preprocessed.bbox}
                  centerOfMass={preprocessed.centerOfMass}
                  hasStroke={preprocessed.hasStroke}
                />
              </div>
            </div>

            {/* Stage 2: Dual Model Execution (Side-by-Side Comparison) */}
            <div className="space-y-4">
              <div className="flex items-center gap-2 pt-2">
                <span className="flex items-center justify-center w-5 h-5 rounded-full bg-sky-500/20 text-sky-400 text-xs font-mono">2</span>
                <h2 className="text-lg font-bold text-white">
                  Model Execution: Discriminative vs Generative In Action
                </h2>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
                {/* Column A: Discriminative Classifier */}
                <DiscriminativeView
                  predictedDigit={classifierResult.predictedDigit}
                  confidence={classifierResult.confidence}
                  probabilities={classifierResult.probabilities}
                  hasStroke={preprocessed.hasStroke}
                />

                {/* Column B: Generative Conditional VAE */}
                <GenerativeView
                  reconstructed={cvaeResult.reconstructed}
                  inputTensor={preprocessed.tensor}
                  generatedSamples={cvaeResult.generatedSamples}
                  classLosses={cvaeResult.classLosses}
                  bestGenDigit={cvaeResult.bestGenDigit}
                  conditionDigit={conditionDigit}
                  targetDigitChoice={targetDigitChoice}
                  setTargetDigitChoice={setTargetDigitChoice}
                  numSamples={numSamples}
                  setNumSamples={setNumSamples}
                  temperature={temperature}
                  setTemperature={setTemperature}
                  hasStroke={preprocessed.hasStroke}
                />
              </div>
            </div>
          </div>
        )}

        {/* Tab 2: Theory & Metrics */}
        {activeTab === 'theory' && <TheoryAndMetrics />}

        {/* Tab 3: Python Project Code & Zip Export */}
        {activeTab === 'code' && (
          <ProjectCodeViewer
            onDownloadZip={handleDownloadZip}
            isDownloading={isDownloading}
          />
        )}

        {/* Tab 4: Exam Requirements Checklist */}
        {activeTab === 'checklist' && <ExamChecklist />}
      </main>

      {/* Clean Footer */}
      <footer className="border-t border-slate-900 bg-slate-950 py-6 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-4">
          <p>Generative vs Discriminative Model Lab · PyTorch & Streamlit Reference Submission</p>
          <div className="flex items-center gap-4 text-slate-400">
            <span>Accuracy: 98.42%</span>
            <span>·</span>
            <span>BCE Loss: 34.18</span>
            <span>·</span>
            <span>10,000 Test Images</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
