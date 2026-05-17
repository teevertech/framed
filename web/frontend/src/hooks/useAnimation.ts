import { useCallback, useEffect, useRef, useState } from "react";
import type { SequenceStep } from "@/types/panel";

export interface AnimationFrame {
  stepIndex: number;
  progress: number; // 0–1 within current step
  placedIds: Set<string>;
  robotXY: [number, number];
  targetId: string | null;
  completedPaths: { from: [number, number]; to: [number, number]; collided: boolean }[];
  partialPath: { from: [number, number]; to: [number, number] } | null;
  collidedThis: boolean;
  cumulativeReward: number;
}

function buildFrame(steps: SequenceStep[], stepIndex: number, progress: number): AnimationFrame {
  const placed = new Set<string>();
  const paths: AnimationFrame["completedPaths"] = [];

  for (let i = 0; i < stepIndex; i++) {
    const s = steps[i]!;
    placed.add(s.member_id);
    paths.push({ from: s.from_xy, to: s.to_xy, collided: s.collided });
  }

  const current = steps[stepIndex];
  const isComplete = stepIndex >= steps.length;

  let robotXY: [number, number];
  let targetId: string | null = null;
  let partialPath: AnimationFrame["partialPath"] = null;
  let collidedThis = false;
  let cumReward = stepIndex > 0 ? steps[stepIndex - 1]!.cumulative_reward : 0;

  if (isComplete || !current) {
    robotXY = steps.length > 0 ? steps[steps.length - 1]!.to_xy : [0, 0];
    cumReward = steps.length > 0 ? steps[steps.length - 1]!.cumulative_reward : 0;
    // Add last step to paths if we just finished
    if (isComplete && steps.length > 0) {
      const last = steps[steps.length - 1]!;
      if (!placed.has(last.member_id)) {
        placed.add(last.member_id);
        paths.push({ from: last.from_xy, to: last.to_xy, collided: last.collided });
      }
    }
  } else {
    targetId = current.member_id;
    collidedThis = current.collided;
    const [fx, fy] = current.from_xy;
    const [tx, ty] = current.to_xy;
    const x = fx + (tx - fx) * progress;
    const y = fy + (ty - fy) * progress;
    robotXY = [x, y];
    partialPath = { from: current.from_xy, to: [x, y] };
    cumReward += current.reward * progress;
  }

  return {
    stepIndex,
    progress,
    placedIds: placed,
    robotXY,
    targetId,
    completedPaths: paths,
    partialPath,
    collidedThis,
    cumulativeReward: cumReward,
  };
}

export function useAnimation(steps: SequenceStep[] | null, speed: number = 1) {
  const [playing, setPlaying] = useState(false);
  const [stepIndex, setStepIndex] = useState(0);
  const [progress, setProgress] = useState(0);
  const lastTimeRef = useRef<number | null>(null);
  const totalSteps = steps?.length ?? 0;

  // Reset when steps change
  useEffect(() => {
    setStepIndex(0);
    setProgress(0);
    setPlaying(false);
    lastTimeRef.current = null;
  }, [steps]);

  // Animation loop
  useEffect(() => {
    if (!playing || !steps || totalSteps === 0) {
      lastTimeRef.current = null;
      return;
    }

    let animId: number;
    const tick = (time: number) => {
      if (lastTimeRef.current === null) {
        lastTimeRef.current = time;
        animId = requestAnimationFrame(tick);
        return;
      }

      const dt = (time - lastTimeRef.current) / 1000;
      lastTimeRef.current = time;

      // Each step takes ~0.6s at speed=1
      const stepDuration = 0.6 / speed;

      setProgress((prev) => {
        let next = prev + dt / stepDuration;
        if (next >= 1) {
          setStepIndex((prevStep) => {
            const nextStep = prevStep + 1;
            if (nextStep >= totalSteps) {
              setPlaying(false);
              return totalSteps;
            }
            return nextStep;
          });
          next = 0;
        }
        return next >= 1 ? 0 : next;
      });

      animId = requestAnimationFrame(tick);
    };

    animId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animId);
  }, [playing, steps, totalSteps, speed]);

  const play = useCallback(() => {
    if (stepIndex >= totalSteps) {
      setStepIndex(0);
      setProgress(0);
    }
    setPlaying(true);
  }, [stepIndex, totalSteps]);

  const pause = useCallback(() => {
    setPlaying(false);
    lastTimeRef.current = null;
  }, []);

  const stepForward = useCallback(() => {
    setPlaying(false);
    lastTimeRef.current = null;
    setProgress(0);
    setStepIndex((prev) => Math.min(prev + 1, totalSteps));
  }, [totalSteps]);

  const stepBack = useCallback(() => {
    setPlaying(false);
    lastTimeRef.current = null;
    setProgress(0);
    setStepIndex((prev) => Math.max(prev - 1, 0));
  }, []);

  const seekTo = useCallback(
    (step: number) => {
      setPlaying(false);
      lastTimeRef.current = null;
      setStepIndex(Math.max(0, Math.min(step, totalSteps)));
      setProgress(0);
    },
    [totalSteps],
  );

  const frame = steps ? buildFrame(steps, Math.min(stepIndex, totalSteps), progress) : null;

  return {
    frame,
    playing,
    stepIndex,
    totalSteps,
    play,
    pause,
    stepForward,
    stepBack,
    seekTo,
  };
}
