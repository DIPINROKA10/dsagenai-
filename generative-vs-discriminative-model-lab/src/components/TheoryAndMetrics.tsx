import React, { useState } from 'react';
import { MNIST_CONFUSION_MATRIX, CLASS_METRICS } from '../data/mnistWeightsAndPresets';
import { BarChart3, BookOpen, CheckCircle, Info } from 'lucide-react';

export const TheoryAndMetrics: React.FC = () => {
  const [hoveredCell, setHoveredCell] = useState<{ trueClass: number; predClass: number; count: number } | null>(null);

  // Maximum value for heatmap color normalization
  const maxDiagonal = 1130;

  return (
    <div className="space-y-8">
      {/* Overview Header */}
      <div>
        <h2 className="text-xl font-bold text-white tracking-tight">
          Model Benchmarks & Theoretical Comparison
        </h2>
        <p className="text-sm text-slate-400 mt-1">
          Detailed metrics evaluated on the 10,000 MNIST test images and mathematical contrast of both paradigms.
        </p>
      </div>

      {/* KPI Metrics Strip */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="p-4 bg-slate-900 border border-slate-800 rounded-xl">
          <span className="text-xs text-slate-400 block mb-1">Classifier Test Accuracy</span>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold text-sky-400 font-mono tabular-nums">98.42%</span>
            <span className="text-[11px] text-emerald-400 font-medium">Target &gt;98% Met</span>
          </div>
          <span className="text-[11px] text-slate-500 mt-1 block">4 Epochs · 10,000 test samples</span>
        </div>

        <div className="p-4 bg-slate-900 border border-slate-800 rounded-xl">
          <span className="text-xs text-slate-400 block mb-1">CVAE Test Recon Loss</span>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold text-emerald-400 font-mono tabular-nums">34.18</span>
            <span className="text-[11px] text-slate-400">BCE / image</span>
          </div>
          <span className="text-[11px] text-slate-500 mt-1 block">10 Epochs · Latent Dim: 16</span>
        </div>

        <div className="p-4 bg-slate-900 border border-slate-800 rounded-xl">
          <span className="text-xs text-slate-400 block mb-1">Average Precision</span>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold text-slate-200 font-mono tabular-nums">98.6%</span>
            <span className="text-[11px] text-slate-400">Macro avg</span>
          </div>
          <span className="text-[11px] text-slate-500 mt-1 block">Across classes 0 through 9</span>
        </div>

        <div className="p-4 bg-slate-900 border border-slate-800 rounded-xl">
          <span className="text-xs text-slate-400 block mb-1">Compute Environment</span>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold text-slate-200 font-mono">CPU</span>
            <span className="text-[11px] text-sky-400">Lightweight</span>
          </div>
          <span className="text-[11px] text-slate-500 mt-1 block">Full training ~3.5 minutes</span>
        </div>
      </div>

      {/* Deep Theoretical Comparison Table */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4">
        <div className="flex items-center gap-2">
          <BookOpen className="w-5 h-5 text-sky-400" />
          <h3 className="text-base font-semibold text-white">
            Discriminative vs Generative: The Fundamental Difference
          </h3>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400">
                <th className="py-3 px-4 font-semibold">Dimension</th>
                <th className="py-3 px-4 font-semibold text-sky-400">Discriminative (SmallCNN)</th>
                <th className="py-3 px-4 font-semibold text-emerald-400">Generative (Conditional VAE)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-300">
              <tr>
                <td className="py-3 px-4 font-medium text-slate-200">Probability Modeled</td>
                <td className="py-3 px-4 font-mono text-sky-300">P(y | x)</td>
                <td className="py-3 px-4 font-mono text-emerald-300">P(x | y) and P(x, y) = P(x|y)·P(y)</td>
              </tr>
              <tr>
                <td className="py-3 px-4 font-medium text-slate-200">Primary Objective</td>
                <td className="py-3 px-4">Directly separate class decision boundaries.</td>
                <td className="py-3 px-4">Learn the true underlying density manifold of the data.</td>
              </tr>
              <tr>
                <td className="py-3 px-4 font-medium text-slate-200">Data Generation</td>
                <td className="py-3 px-4 text-rose-400 font-medium">❌ Cannot synthesize novel samples.</td>
                <td className="py-3 px-4 text-emerald-400 font-medium">✅ Samples z ~ N(0, τ²·I) to synthesize novel variations.</td>
              </tr>
              <tr>
                <td className="py-3 px-4 font-medium text-slate-200">Classification Mechanism</td>
                <td className="py-3 px-4">Direct forward pass through convolutional layers + Softmax.</td>
                <td className="py-3 px-4">Analysis-by-synthesis: argmin_k L(x | y=k) (reconstruction error).</td>
              </tr>
              <tr>
                <td className="py-3 px-4 font-medium text-slate-200">Out-of-Distribution (OOD)</td>
                <td className="py-3 px-4">Susceptible to high confidence on arbitrary noise/adversarial inputs.</td>
                <td className="py-3 px-4">Detects anomalies via high reconstruction loss / low density.</td>
              </tr>
              <tr>
                <td className="py-3 px-4 font-medium text-slate-200">Loss Function</td>
                <td className="py-3 px-4 font-mono">Cross-Entropy Loss (Softmax Logits)</td>
                <td className="py-3 px-4 font-mono">Evidence Lower Bound (ELBO): BCE + KL Divergence</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* 10x10 MNIST Confusion Matrix Heatmap */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-6">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-sky-400" />
              <h3 className="text-base font-semibold text-white">
                Classifier 10×10 Confusion Matrix (Test Set)
              </h3>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Evaluated on all 10,000 MNIST test samples. Rows = True Class, Columns = Predicted Class.
            </p>
          </div>

          {hoveredCell ? (
            <div className="px-3 py-1.5 bg-slate-950 border border-slate-700 rounded-lg text-xs font-mono">
              <span className="text-slate-400">True: </span>
              <span className="text-sky-300 font-bold">{hoveredCell.trueClass}</span>
              <span className="text-slate-600 mx-2">·</span>
              <span className="text-slate-400">Pred: </span>
              <span className="text-sky-300 font-bold">{hoveredCell.predClass}</span>
              <span className="text-slate-600 mx-2">·</span>
              <span className="text-slate-400">Count: </span>
              <span className="text-white font-bold tabular-nums">{hoveredCell.count}</span>
            </div>
          ) : (
            <div className="text-xs text-slate-500 font-mono">Hover over cells to inspect exact counts</div>
          )}
        </div>

        {/* Matrix Grid */}
        <div className="overflow-x-auto pb-2">
          <div className="inline-block min-w-[500px]">
            {/* Column Header (Predicted) */}
            <div className="flex items-center ml-12 mb-1">
              <span className="text-xs font-semibold text-slate-400 text-center w-full">
                Predicted Class →
              </span>
            </div>

            <div className="flex items-center ml-12 mb-1 text-[11px] font-mono text-slate-400">
              {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((c) => (
                <div key={c} className="w-10 text-center font-bold">
                  {c}
                </div>
              ))}
            </div>

            {/* Matrix Rows */}
            <div className="space-y-1">
              {MNIST_CONFUSION_MATRIX.map((row, r) => (
                <div key={r} className="flex items-center">
                  <div className="w-12 text-right pr-3 text-xs font-mono font-bold text-slate-400">
                    {r}
                  </div>

                  <div className="flex gap-1">
                    {row.map((val, c) => {
                      const isDiag = r === c;
                      // Heatmap color calculation
                      let bgStyle = '';
                      let textStyle = 'text-slate-500';

                      if (isDiag) {
                        const alpha = Math.min(0.9, Math.max(0.3, val / maxDiagonal));
                        bgStyle = `rgba(14, 165, 233, ${alpha})`;
                        textStyle = 'text-white font-bold';
                      } else if (val > 0) {
                        bgStyle = 'rgba(239, 68, 68, 0.25)';
                        textStyle = 'text-rose-300 font-medium';
                      } else {
                        bgStyle = 'rgba(15, 23, 42, 0.6)';
                      }

                      return (
                        <div
                          key={c}
                          onMouseEnter={() => setHoveredCell({ trueClass: r, predClass: c, count: val })}
                          onMouseLeave={() => setHoveredCell(null)}
                          style={{ backgroundColor: bgStyle }}
                          className={`w-10 h-9 flex items-center justify-center text-[11px] font-mono rounded cursor-pointer transition-transform hover:scale-105 border border-slate-800/80 ${textStyle}`}
                        >
                          {val}
                        </div>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Per-Class Precision & Recall Table */}
        <div className="pt-4 border-t border-slate-800">
          <h4 className="text-xs font-semibold text-slate-300 mb-3">
            Per-Class Accuracy Breakdown
          </h4>
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-2.5">
            {CLASS_METRICS.map((item) => (
              <div key={item.digit} className="p-2.5 bg-slate-950/70 border border-slate-800 rounded-lg text-xs">
                <div className="flex justify-between items-center mb-1">
                  <span className="font-mono font-bold text-sky-400 text-sm">Digit {item.digit}</span>
                  <span className="text-[10px] text-slate-500 font-mono">F1: {item.f1}%</span>
                </div>
                <div className="space-y-0.5 text-[11px] text-slate-400 font-mono">
                  <div className="flex justify-between">
                    <span>Precision:</span>
                    <span className="text-slate-200 tabular-nums">{item.precision}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Recall:</span>
                    <span className="text-slate-200 tabular-nums">{item.recall}%</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
