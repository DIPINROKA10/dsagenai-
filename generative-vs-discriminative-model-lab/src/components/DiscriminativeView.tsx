import React from 'react';
import { Target, CheckCircle2 } from 'lucide-react';

interface DiscriminativeViewProps {
  predictedDigit: number;
  confidence: number;
  probabilities: number[];
  hasStroke: boolean;
}

export const DiscriminativeView: React.FC<DiscriminativeViewProps> = ({
  predictedDigit,
  confidence,
  probabilities,
  hasStroke,
}) => {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-5">
      <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
        <div className="flex items-center gap-2">
          <Target className="w-4 h-4 text-sky-400" />
          <h3 className="text-sm font-semibold text-slate-100">
            Discriminative Model: SmallCNN Classifier
          </h3>
        </div>
        <span className="text-xs font-mono text-emerald-400 bg-emerald-950/60 border border-emerald-800/50 px-2 py-0.5 rounded flex items-center gap-1">
          <CheckCircle2 className="w-3 h-3" />
          <span>Test Acc: 98.42%</span>
        </span>
      </div>

      {/* Prediction Cards */}
      <div className="grid grid-cols-2 gap-3">
        <div className="p-3.5 bg-slate-950/80 border border-slate-800 rounded-lg">
          <span className="text-xs text-slate-400 block mb-1">Predicted Class</span>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-extrabold text-white font-mono tabular-nums">
              {hasStroke ? predictedDigit : '—'}
            </span>
            <span className="text-xs text-slate-500">Digit 0–9</span>
          </div>
        </div>

        <div className="p-3.5 bg-slate-950/80 border border-slate-800 rounded-lg">
          <span className="text-xs text-slate-400 block mb-1">Model Confidence</span>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-extrabold text-sky-400 font-mono tabular-nums">
              {hasStroke ? `${confidence.toFixed(1)}%` : '—'}
            </span>
            <span className="text-xs text-slate-500">Softmax p(y|x)</span>
          </div>
        </div>
      </div>

      {/* 10-Class Probability Distribution Bar Chart */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs text-slate-400">
          <span className="font-medium">10-Class Posterior Distribution P(y | x)</span>
          <span className="font-mono text-[11px] text-slate-500">Sum = 100.0%</span>
        </div>

        <div className="space-y-1.5 pt-1">
          {probabilities.map((prob, idx) => {
            const isWinner = hasStroke && idx === predictedDigit;
            const pct = (prob * 100).toFixed(1);

            return (
              <div key={idx} className="flex items-center gap-2.5 text-xs">
                <span
                  className={`w-5 font-mono text-right font-medium ${
                    isWinner ? 'text-sky-400 font-bold' : 'text-slate-400'
                  }`}
                >
                  {idx}
                </span>

                <div className="flex-1 h-3.5 bg-slate-950 rounded-sm overflow-hidden border border-slate-800/80">
                  <div
                    className={`h-full transition-all duration-300 rounded-sm ${
                      isWinner
                        ? 'bg-sky-400 shadow-[0_0_8px_rgba(56,189,248,0.5)]'
                        : prob > 0.05
                        ? 'bg-slate-600'
                        : 'bg-slate-800'
                    }`}
                    style={{ width: `${Math.max(1, prob * 100)}%` }}
                  />
                </div>

                <span
                  className={`w-12 font-mono text-right tabular-nums ${
                    isWinner ? 'text-sky-300 font-semibold' : 'text-slate-500'
                  }`}
                >
                  {pct}%
                </span>
              </div>
            );
          })}
        </div>
      </div>

      <div className="p-3 bg-slate-950/60 border border-slate-800/70 rounded-lg text-[11px] leading-relaxed text-slate-400">
        <strong className="text-slate-300 block mb-0.5">Discriminative Principle:</strong>
        SmallCNN directly computes the posterior <span className="font-mono text-sky-300">P(y|x)</span> by constructing multidimensional decision boundaries. It does not model pixel density <span className="font-mono text-slate-300">P(x)</span> and cannot synthesize new images.
      </div>
    </div>
  );
};
