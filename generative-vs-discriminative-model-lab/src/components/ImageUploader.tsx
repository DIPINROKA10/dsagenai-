import React, { useState, useRef } from 'react';
import { Upload, Image as ImageIcon, ArrowLeftRight } from 'lucide-react';

interface ImageUploaderProps {
  onMatrixChange: (matrix: number[][]) => void;
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({ onMatrixChange }) => {
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [invertColors, setInvertColors] = useState<boolean>(true);
  const [threshold, setThreshold] = useState<number>(30);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const processFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const result = e.target?.result as string;
      setImagePreview(result);
      processImageBitmap(result, invertColors, threshold);
    };
    reader.readAsDataURL(file);
  };

  const processImageBitmap = (dataUrl: string, invert: boolean, thresh: number) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      canvas.width = 28;
      canvas.height = 28;

      ctx.fillStyle = invert ? '#000000' : '#ffffff';
      ctx.fillRect(0, 0, 28, 28);
      ctx.drawImage(img, 0, 0, 28, 28);

      const imgData = ctx.getImageData(0, 0, 28, 28);
      const data = imgData.data;
      const matrix: number[][] = [];

      for (let r = 0; r < 28; r++) {
        const row: number[] = [];
        for (let c = 0; c < 28; c++) {
          const idx = (r * 28 + c) * 4;
          let lum = (data[idx] * 0.299 + data[idx + 1] * 0.587 + data[idx + 2] * 0.114);

          // If invert is enabled: black stroke (0) becomes white (255) on black canvas
          if (invert) {
            lum = 255 - lum;
          }

          if (lum < thresh) lum = 0;
          row.push(Math.min(1.0, Math.max(0, lum / 255)));
        }
        matrix.push(row);
      }

      onMatrixChange(matrix);
    };
    img.src = dataUrl;
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleInvertToggle = () => {
    const newInvert = !invertColors;
    setInvertColors(newInvert);
    if (imagePreview) {
      processImageBitmap(imagePreview, newInvert, threshold);
    }
  };

  const handleThresholdChange = (val: number) => {
    setThreshold(val);
    if (imagePreview) {
      processImageBitmap(imagePreview, invertColors, val);
    }
  };

  return (
    <div className="space-y-4">
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className="border-2 border-dashed border-slate-800 hover:border-sky-500/60 bg-slate-900/60 rounded-xl p-6 text-center cursor-pointer transition-colors"
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/png, image/jpeg, image/jpg"
          className="hidden"
          onChange={(e) => {
            if (e.target.files && e.target.files[0]) {
              processFile(e.target.files[0]);
            }
          }}
        />

        <div className="flex flex-col items-center justify-center gap-2">
          <div className="p-3 bg-slate-800 rounded-full text-slate-300">
            <Upload className="w-5 h-5 text-sky-400" />
          </div>
          <p className="text-sm font-medium text-slate-200">
            Click to upload or drag & drop digit image
          </p>
          <p className="text-xs text-slate-500">
            PNG, JPG or JPEG (handwritten digit photo or sketch)
          </p>
        </div>
      </div>

      {imagePreview && (
        <div className="flex items-center gap-4 p-3 bg-slate-900 border border-slate-800 rounded-lg">
          <img
            src={imagePreview}
            alt="Uploaded digit"
            className="w-16 h-16 object-cover rounded bg-black border border-slate-800"
          />
          <div className="flex-1 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs text-slate-400 font-medium">Auto Inversion (Dark on Light)</span>
              <button
                type="button"
                onClick={handleInvertToggle}
                className={`flex items-center gap-1.5 px-2.5 py-1 text-xs rounded transition-colors ${
                  invertColors ? 'bg-sky-500/20 text-sky-300 border border-sky-500/30' : 'bg-slate-800 text-slate-400'
                }`}
              >
                <ArrowLeftRight className="w-3 h-3" />
                <span>{invertColors ? 'Inverted (White on Black)' : 'Standard'}</span>
              </button>
            </div>
            <div>
              <div className="flex justify-between text-xs text-slate-400 mb-1">
                <span>Noise Threshold</span>
                <span className="font-mono tabular-nums text-slate-300">{threshold}</span>
              </div>
              <input
                type="range"
                min={0}
                max={100}
                value={threshold}
                onChange={(e) => handleThresholdChange(Number(e.target.value))}
                className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-sky-400"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
