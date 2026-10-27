import React from 'react';
import { Palette, Sparkles, SlidersHorizontal } from 'lucide-react';

interface GenerativeViewProps {
  reconstructed: number[][];
  inputTensor: number[][];
  generatedSamples: number[][][];
  classLosses: number[];
  bestGenDigit: number;
  conditionDigit: number;
  targetDigitChoice: string;
  setTargetDigitChoice: (val: string) => void;
  numSamples: number;
  setNumSamples: (val: number) => void;
  temperature: number;
  setTemperature: (val: number) => void;
  hasStroke: boolean;
}

export const GenerativeView: React.FC<GenerativeViewProps> = ({
  reconstructed,
  inputTensor,
  generatedSamples,
  classLosses,
  bestGenDigit,
  conditionDigit,
  targetDigitChoice,
  setTargetDigitChoice,
  numSamples,
  setNumSamples,
  temperature,
  setTemperature,
  hasStroke,
}) => {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-6">
      <div className="flex flex-wrap items-center justify-between border-b border-slate-800/80 pb-3 gap-2">
        <div className="flex items-center gap-2">
          <Palette className="w-4 h-4 text-emerald-400" />
          <h3 className="text-sm font-semibold text-slate-100">
            Generative Model: Conditional VAE (CVAE)
          </h3>
        </div>
        <span className="text-xs font-mono text-emerald-400 bg-emerald-950/60 border border-emerald-800/50 px-2 py-0.5 rounded">
          Latent Dim: 16 · Test BCE: 34.18
        </span>
      </div>

      {/* Sampling & Conditioning Controls */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 p-3.5 bg-slate-950/80 border border-slate-800 rounded-lg">
        {/* Sample Count Slider */}
        <div className="space-y-1.5">
          <div className="flex justify-between text-xs text-slate-400 font-medium">
            <span>Sample Count</span>
            <span className="font-mono text-slate-200 tabular-nums">{numSamples} samples</span>
          </div>
          <input
            type="range"
            min={1}
            max={16}
            value={numSamples}
            onChange={(e) => setNumSamples(Number(e.target.value))}
            className="w-full h-1.5 bg-slate-800 rounded appearance-none cursor-pointer accent-emerald-400"
          />
        </div>

        {/* Temperature / Latent Variability Slider */}
        <div className="space-y-1.5">
          <div className="flex justify-between text-xs text-slate-400 font-medium">
            <span>Variability (Temp τ)</span>
            <span className="font-mono text-slate-200 tabular-nums">τ = {temperature.toFixed(1)}</span>
          </div>
          <input
            type="range"
            min={0.1}
            max={2.5}
            step={0.1}
            value={temperature}
            onChange={(e) => setTemperature(Number(e.target.value))}
            className="w-full h-1.5 bg-slate-800 rounded appearance-none cursor-pointer accent-emerald-400"
          />
          <div className="flex justify-between text-[10px] text-slate-500">
            <span>Crisp</span>
            <span>Standard (1.0)</span>
            <span>Diverse</span>
          </div>
        </div>

        {/* Target Digit Dropdown */}
        <div className="space-y-1.5">
          <div className="text-xs text-slate-400 font-medium">Conditioning Label</div>
          <select
            value={targetDigitChoice}
            onChange={(e) => setTargetDigitChoice(e.target.value)}
            className="w-full px-2.5 py-1 text-xs bg-slate-900 border border-slate-700 text-slate-200 rounded focus:outline-none focus:border-emerald-500 cursor-pointer"
          >
            <option value="Use predicted digit">Use predicted digit ({conditionDigit})</option>
            {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((d) => (
              <option key={d} value={String(d)}>
                Force Digit {d}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Side-by-Side Reconstruction */}
      <div className="space-y-3">
        <div className="flex items-center justify-between text-xs text-slate-400">
          <span className="font-medium text-slate-300">
            Autoencoding Reconstruction (x → z → x̂) Conditioned on y={conditionDigit}
          </span>
          <span className="text-emerald-400 font-mono text-[11px]">Loss: {classLosses[conditionDigit]} BCE</span>
        </div>

        <div className="grid grid-cols-2 gap-4 p-3 bg-slate-950/60 border border-slate-800/80 rounded-lg">
          <div className="flex flex-col items-center gap-2">
            <span className="text-[11px] text-slate-400">Original Drawn/Uploaded Input</span>
            <div className="w-[100px] h-[100px] bg-black p-1 border border-slate-800 rounded">
              <CanvasGrid matrix={inputTensor} />
            </div>
          </div>

          <div className="flex flex-col items-center gap-2">
            <span className="text-[11px] text-emerald-400 font-medium">
              CVAE Reconstructed Output (x̂)
            </span>
            <div className="w-[100px] h-[100px] bg-black p-1 border border-emerald-900/40 rounded">
              <CanvasGrid matrix={reconstructed} />
            </div>
          </div>
        </div>
      </div>

      {/* Generated Digit Samples Grid */}
      <div className="space-y-3">
        <div className="flex items-center justify-between text-xs text-slate-400">
          <div className="flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
            <span className="font-medium text-slate-200">
              {numSamples} Generated Samples (Prior z ~ N(0, τ²·I), Condition y={conditionDigit})
            </span>
          </div>
          <span className="text-slate-500 font-mono text-[11px]">τ = {temperature.toFixed(1)}</span>
        </div>

        <div className="grid grid-cols-4 sm:grid-cols-8 gap-2 p-3 bg-slate-950/60 border border-slate-800/80 rounded-lg">
          {generatedSamples.slice(0, numSamples).map((sample, idx) => (
            <div
              key={idx}
              className="flex flex-col items-center p-1 bg-black border border-slate-800 hover:border-emerald-500/50 rounded transition-colors"
            >
              <div className="w-12 h-12">
                <CanvasGrid matrix={sample} />
              </div>
              <span className="text-[9px] font-mono text-slate-500 mt-1">#{idx + 1}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Generative Classification (Hypothesis Testing) */}
      <div className="space-y-3 pt-2 border-t border-slate-800/80">
        <div className="flex items-center justify-between text-xs">
          <span className="font-medium text-slate-200">
            Generative Classification: Hypothesis Loss L(x | y=k)
          </span>
          <span className="text-emerald-400 font-mono text-[11px]">
            Lowest Loss = Digit {bestGenDigit}
          </span>
        </div>

        <p className="text-[11px] text-slate-400 leading-relaxed">
          The generative model tests the input against each candidate class condition <span className="font-mono text-slate-300">0–9</span>. The class condition that yields the minimum reconstruction error represents the most likely data generator:
        </p>

        <div className="space-y-1 pt-1">
          {classLosses.map((loss, k) => {
            const isBest = k === bestGenDigit;
            // Normalize loss for bar display (lower loss is better)
            const minL = Math.min(...classLosses);
            const maxL = Math.max(...classLosses);
            const relativeWidth = Math.max(10, 100 - ((loss - minL) / (maxL - minL + 1e-5)) * 80);

            return (
              <div key={k} className="flex items-center gap-2 text-xs">
                <span className={`w-5 font-mono text-right ${isBest ? 'text-emerald-400 font-bold' : 'text-slate-400'}`}>
                  {k}
                </span>

                <div className="flex-1 h-3 bg-slate-950 rounded-sm overflow-hidden border border-slate-800">
                  <div
                    className={`h-full transition-all duration-300 rounded-sm ${
                      isBest ? 'bg-emerald-400' : 'bg-slate-700'
                    }`}
                    style={{ width: `${relativeWidth}%` }}
                  />
                </div>

                <span className={`w-14 font-mono text-right tabular-nums text-[11px] ${isBest ? 'text-emerald-300 font-semibold' : 'text-slate-500'}`}>
                  {loss.toFixed(1)} BCE
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

// Canvas grid helper to render 28x28 matrix cleanly
const CanvasGrid: React.FC<{ matrix: number[][] }> = ({ matrix }) => {
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = '#05070e';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const cellW = canvas.width / 28;
    const cellH = canvas.height / 28;

    for (let r = 0; r < 28; r++) {
      for (let c = 0; c < 28; c++) {
        const val = matrix[r] && matrix[r][c] !== undefined ? matrix[r][c] : 0;
        if (val > 0.01) {
          const intensity = Math.round(val * 255);
          ctx.fillStyle = `rgb(${intensity}, ${intensity}, ${intensity})`;
          ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
        }
      }
    }
  }, [matrix]);

  return (
    <canvas
      ref={canvasRef}
      width={56}
      height={56}
      className="w-full h-full rounded"
      style={{ imageRendering: 'pixelated' }}
    />
  );
};
