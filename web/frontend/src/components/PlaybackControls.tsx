import { useEffect, useRef, useState } from "react";

const SPEED_OPTIONS = [0.5, 1, 1.5, 2, 3, 4];

interface Props {
  playing: boolean;
  stepIndex: number;
  totalSteps: number;
  speed: number;
  onPlay: () => void;
  onPause: () => void;
  onStepForward: () => void;
  onStepBack: () => void;
  onSeek: (step: number) => void;
  onSpeedChange: (speed: number) => void;
}

export default function PlaybackControls({
  playing,
  stepIndex,
  totalSteps,
  speed,
  onPlay,
  onPause,
  onStepForward,
  onStepBack,
  onSeek,
  onSpeedChange,
}: Props) {
  const [speedOpen, setSpeedOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setSpeedOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return (
    <div className="bg-c-subtle rounded-lg px-3 py-2 space-y-2 transition-colors">
      {/* Row 1: transport · step counter · speed */}
      <div className="flex items-center gap-1">
        <button
          onClick={onStepBack}
          disabled={stepIndex <= 0}
          className="p-1 text-c-text-2 hover:text-c-text-1 disabled:opacity-40 transition-colors"
          aria-label="Step back"
        >
          <StepBackIcon />
        </button>

        <button
          onClick={playing ? onPause : onPlay}
          className="p-1 text-c-text-2 hover:text-c-text-1 transition-colors"
          aria-label={playing ? "Pause" : "Play"}
        >
          {playing ? <PauseIcon /> : <PlayIcon />}
        </button>

        <button
          onClick={onStepForward}
          disabled={stepIndex >= totalSteps}
          className="p-1 text-c-text-2 hover:text-c-text-1 disabled:opacity-40 transition-colors"
          aria-label="Step forward"
        >
          <StepForwardIcon />
        </button>

        <span className="flex-1 text-xs text-c-text-2 text-right tabular-nums">
          {stepIndex} / {totalSteps}
        </span>

        {/* Speed dropdown */}
        <div className="relative ml-1" ref={dropdownRef}>
          <button
            onClick={() => setSpeedOpen((o) => !o)}
            className="flex items-center gap-1 px-1.5 py-0.5 rounded text-[11px]
                       text-c-text-2 hover:bg-c-border transition-colors"
            aria-label="Playback speed"
            title="Playback speed"
          >
            <SpeedIcon />
            <span className="tabular-nums">{speed}×</span>
          </button>

          {speedOpen && (
            <div
              className="absolute right-0 bottom-full mb-1 bg-c-surface border border-c-border
                         rounded-lg shadow-lg py-1 z-10 min-w-[72px] transition-colors"
            >
              {SPEED_OPTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => { onSpeedChange(s); setSpeedOpen(false); }}
                  className={`w-full text-left px-3 py-1 text-xs transition-colors ${
                    s === speed
                      ? "bg-c-subtle text-c-text-1 font-medium"
                      : "text-c-text-2 hover:bg-c-base"
                  }`}
                >
                  {s}×
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Row 2: scrub bar */}
      <input
        type="range"
        min={0}
        max={totalSteps}
        step={1}
        value={stepIndex}
        onChange={(e) => onSeek(Number(e.target.value))}
        className="w-full block"
      />
    </div>
  );
}

function PlayIcon() {
  return <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>;
}
function PauseIcon() {
  return <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M6 5h4v14H6zM14 5h4v14h-4z"/></svg>;
}
function StepBackIcon() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M6 6h2v12H6zM9.5 12l8.5 6V6z"/></svg>;
}
function StepForwardIcon() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M16 6h2v12h-2zM6 18l8.5-6L6 6z"/></svg>;
}
function SpeedIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" className="opacity-60">
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2"/>
      <line x1="12" y1="12" x2="12" y2="7"  stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
      <line x1="12" y1="12" x2="16" y2="14" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
    </svg>
  );
}
