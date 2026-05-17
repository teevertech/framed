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
  return (
    <div className="flex items-center gap-2 bg-gray-100 rounded-lg px-3 py-2">
      <button
        onClick={onStepBack}
        disabled={stepIndex <= 0}
        className="p-1 text-gray-600 hover:text-gray-900 disabled:text-gray-300 transition-colors"
        aria-label="Step back"
      >
        <StepBackIcon />
      </button>

      <button
        onClick={playing ? onPause : onPlay}
        className="p-1 text-gray-600 hover:text-gray-900 transition-colors"
        aria-label={playing ? "Pause" : "Play"}
      >
        {playing ? <PauseIcon /> : <PlayIcon />}
      </button>

      <button
        onClick={onStepForward}
        disabled={stepIndex >= totalSteps}
        className="p-1 text-gray-600 hover:text-gray-900 disabled:text-gray-300 transition-colors"
        aria-label="Step forward"
      >
        <StepForwardIcon />
      </button>

      <input
        type="range"
        min={0}
        max={totalSteps}
        step={1}
        value={stepIndex}
        onChange={(e) => onSeek(Number(e.target.value))}
        className="flex-1"
      />

      <span className="text-xs text-gray-500 min-w-[48px] text-right tabular-nums">
        {stepIndex} / {totalSteps}
      </span>

      <div className="border-l border-gray-300 h-4 mx-1" />

      <span className="text-[11px] text-gray-400">Speed</span>
      <input
        type="range"
        min={0.5}
        max={4}
        step={0.5}
        value={speed}
        onChange={(e) => onSpeedChange(Number(e.target.value))}
        className="w-14"
      />
      <span className="text-[11px] text-gray-500 min-w-[24px]">{speed}×</span>
    </div>
  );
}

function PlayIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
      <path d="M8 5v14l11-7z" />
    </svg>
  );
}

function PauseIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
      <path d="M6 5h4v14H6zM14 5h4v14h-4z" />
    </svg>
  );
}

function StepBackIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
      <path d="M6 6h2v12H6zM9.5 12l8.5 6V6z" />
    </svg>
  );
}

function StepForwardIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
      <path d="M16 6h2v12h-2zM6 18l8.5-6L6 6z" />
    </svg>
  );
}
