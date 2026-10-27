import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Eraser, RotateCcw, Sparkles } from 'lucide-react';
import { MNIST_PRESETS } from '../data/mnistWeightsAndPresets';

interface DrawingCanvasProps {
  onMatrixChange: (matrix: number[][]) => void;
}

export const DrawingCanvas: React.FC<DrawingCanvasProps> = ({ onMatrixChange }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [strokeWidth, setStrokeWidth] = useState<number>(18);
  const [hasDrawn, setHasDrawn] = useState(false);
  const lastPos = useRef<{ x: number; y: number } | null>(null);

  // Extract 280x280 pixel intensities to a normalized 280x280 matrix (0 to 1)
  const emitMatrix = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imgData.data;
    const matrix: number[][] = [];

    // Subsample from 280x280 to 28x28 for fast response
    const scaleFactor = canvas.width / 28;
    for (let r = 0; r < 28; r++) {
      const row: number[] = [];
      for (let c = 0; c < 28; c++) {
        // Average the corresponding block
        let sum = 0;
        let count = 0;
        const startY = Math.floor(r * scaleFactor);
        const endY = Math.floor((r + 1) * scaleFactor);
        const startX = Math.floor(c * scaleFactor);
        const endX = Math.floor((c + 1) * scaleFactor);

        for (let py = startY; py < endY; py++) {
          for (let px = startX; px < endX; px++) {
            const idx = (py * canvas.width + px) * 4;
            // Canvas has white strokes (255, 255, 255) on black background (0, 0, 0)
            const intensity = (data[idx] + data[idx + 1] + data[idx + 2]) / 3 / 255;
            sum += intensity;
            count++;
          }
        }
        row.push(count > 0 ? sum / count : 0);
      }
      matrix.push(row);
    }

    onMatrixChange(matrix);
  }, [onMatrixChange]);

  // Initialize canvas with black background
  const initCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.fillStyle = '#05070e';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setHasDrawn(false);
    emitMatrix();
  }, [emitMatrix]);

  useEffect(() => {
    initCanvas();
  }, [initCanvas]);

  // Load a genuine MNIST preset digit onto the canvas
  const loadPreset = (digit: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const preset = MNIST_PRESETS[digit];
    ctx.fillStyle = '#05070e';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const cellW = canvas.width / 28;
    const cellH = canvas.height / 28;

    for (let r = 0; r < 28; r++) {
      for (let c = 0; c < 28; c++) {
        const val = preset[r][c];
        if (val > 0.01) {
          const intensity = Math.round(val * 255);
          ctx.fillStyle = `rgb(${intensity}, ${intensity}, ${intensity})`;
          ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
        }
      }
    }
    setHasDrawn(true);
    emitMatrix();
  };

  const getCanvasCoordinates = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
    const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY,
    };
  };

  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    setIsDrawing(true);
    setHasDrawn(true);
    const coords = getCanvasCoordinates(e);
    lastPos.current = coords;

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(coords.x, coords.y, strokeWidth / 2, 0, Math.PI * 2);
    ctx.fill();
  };

  const draw = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    if (!isDrawing || !lastPos.current) return;
    e.preventDefault();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const currentPos = getCanvasCoordinates(e);

    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = strokeWidth;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    ctx.beginPath();
    ctx.moveTo(lastPos.current.x, lastPos.current.y);
    ctx.lineTo(currentPos.x, currentPos.y);
    ctx.stroke();

    lastPos.current = currentPos;
  };

  const stopDrawing = () => {
    if (isDrawing) {
      setIsDrawing(false);
      lastPos.current = null;
      emitMatrix();
    }
  };

  return (
    <div className="space-y-4">
      {/* Canvas container */}
      <div className="flex flex-col sm:flex-row items-center gap-6">
        <div className="relative group p-2 bg-slate-900 border border-slate-800 rounded-xl shadow-inner">
          <canvas
            ref={canvasRef}
            width={280}
            height={280}
            onMouseDown={startDrawing}
            onMouseMove={draw}
            onMouseUp={stopDrawing}
            onMouseLeave={stopDrawing}
            onTouchStart={startDrawing}
            onTouchMove={draw}
            onTouchEnd={stopDrawing}
            className="w-[260px] h-[260px] sm:w-[280px] sm:h-[280px] cursor-crosshair rounded-lg touch-none"
            style={{ imageRendering: 'pixelated' }}
          />

          {/* Guide grid watermark lines */}
          <div className="absolute inset-2 pointer-events-none border border-slate-800/40 rounded-lg flex items-center justify-center">
            <div className="w-full h-px bg-slate-800/30"></div>
            <div className="absolute h-full w-px bg-slate-800/30"></div>
          </div>
        </div>

        {/* Canvas Controls */}
        <div className="flex-1 space-y-4 w-full">
          <div>
            <div className="flex justify-between text-xs text-slate-400 mb-1.5 font-medium">
              <span>Brush Thickness</span>
              <span className="font-mono tabular-nums text-slate-200">{strokeWidth}px</span>
            </div>
            <input
              type="range"
              min={10}
              max={26}
              value={strokeWidth}
              onChange={(e) => setStrokeWidth(Number(e.target.value))}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-sky-400"
            />
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={initCanvas}
              className="flex-1 flex items-center justify-center gap-2 px-3 py-2 text-xs font-medium text-slate-300 bg-slate-800 hover:bg-slate-700 hover:text-white rounded-lg transition-colors cursor-pointer"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Clear Canvas</span>
            </button>
          </div>

          {/* Preset MNIST Digit Loader */}
          <div className="pt-2 border-t border-slate-800/80">
            <div className="flex items-center gap-1.5 text-xs text-slate-400 font-medium mb-2">
              <Sparkles className="w-3.5 h-3.5 text-sky-400" />
              <span>Load MNIST Test Preset (0–9):</span>
            </div>
            <div className="grid grid-cols-5 gap-1.5">
              {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((digit) => (
                <button
                  key={digit}
                  onClick={() => loadPreset(digit)}
                  className="px-2 py-1.5 text-xs font-mono font-medium text-slate-300 bg-slate-900 border border-slate-800 hover:border-sky-500 hover:text-sky-300 hover:bg-slate-800 rounded transition-colors text-center cursor-pointer"
                >
                  {digit}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
