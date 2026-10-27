import React, { useRef, useEffect } from 'react';

interface PreprocessedViewProps {
  tensor: number[][];
  bbox: { minX: number; minY: number; maxX: number; maxY: number };
  centerOfMass: { cx: number; cy: number };
  hasStroke: boolean;
}

export const PreprocessedView: React.FC<PreprocessedViewProps> = ({
  tensor,
  bbox,
  centerOfMass,
  hasStroke,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
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
        const val = tensor[r] && tensor[r][c] !== undefined ? tensor[r][c] : 0;
        if (val > 0.005) {
          const intensity = Math.round(val * 255);
          ctx.fillStyle = `rgb(${intensity}, ${intensity}, ${intensity})`;
          ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
        }
      }
    }

    // Draw center-of-mass crosshair indicator if stroke exists
    if (hasStroke) {
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.4)';
      ctx.lineWidth = 1;
      const cmX = centerOfMass.cx * cellW;
      const cmY = centerOfMass.cy * cellH;
      ctx.beginPath();
      ctx.arc(cmX, cmY, 4, 0, Math.PI * 2);
      ctx.stroke();
    }
  }, [tensor, hasStroke, centerOfMass]);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-200">
          Preprocessed 28×28 Tensor
        </h3>
        <span className="text-xs font-mono text-sky-400 bg-sky-950/60 border border-sky-800/50 px-2 py-0.5 rounded">
          Shape: [1, 1, 28, 28]
        </span>
      </div>

      <div className="flex flex-col sm:flex-row items-center gap-5">
        <div className="relative p-1 bg-black border border-slate-800 rounded-lg shadow-inner shrink-0">
          <canvas
            ref={canvasRef}
            width={140}
            height={140}
            className="w-[140px] h-[140px] rounded"
            style={{ imageRendering: 'pixelated' }}
          />
        </div>

        <div className="space-y-2 text-xs text-slate-400 w-full">
          <div className="grid grid-cols-2 gap-2 p-2.5 bg-slate-950/60 rounded-lg border border-slate-800/60">
            <div>
              <span className="text-slate-500 block">Center of Mass:</span>
              <span className="font-mono text-slate-200 font-semibold tabular-nums">
                ({centerOfMass.cx.toFixed(1)}, {centerOfMass.cy.toFixed(1)})
              </span>
            </div>
            <div>
              <span className="text-slate-500 block">Active Status:</span>
              <span className={`font-medium ${hasStroke ? 'text-emerald-400' : 'text-amber-400'}`}>
                {hasStroke ? 'Valid Digit Stroke' : 'Canvas Blank'}
              </span>
            </div>
          </div>

          <p className="text-[11px] leading-relaxed text-slate-400">
            Processed via <code className="text-sky-300 font-mono">preprocessing.py</code>:
            Bounding box crop → Aspect-preserved 20×20 scaling → Mass-centered inside 28×28 canvas → Normalized [0.0, 1.0].
          </p>
        </div>
      </div>
    </div>
  );
};
