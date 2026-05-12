import { useEffect } from "react";
import { useShallow } from "zustand/react/shallow";
import { useTTSStore } from "../stores/ttsStore";

function clamp(min: number, value: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

export default function TTSPanel() {
  const { displayText, segments, segmentIndex, speakerPlaying, nextSegment } =
    useTTSStore(
      useShallow((s) => ({
        displayText: s.displayText,
        segments: s.segments,
        segmentIndex: s.segmentIndex,
        speakerPlaying: s.speakerPlaying,
        nextSegment: s.nextSegment,
      })),
    );

  useEffect(() => {
    if (!displayText) return;
    if (!speakerPlaying) return;
    if (segments.length <= 1) return;
    if (segmentIndex >= segments.length - 1) return;

    const fixedDelayRaw = Number(import.meta.env.VITE_TTS_SEGMENT_DELAY_MS);
    const hasFixedDelay = Number.isFinite(fixedDelayRaw) && fixedDelayRaw > 0;

	    const delayMs = hasFixedDelay
	      ? fixedDelayRaw
	      : (() => {
	          const perCharMs = Number(import.meta.env.VITE_TTS_SEGMENT_MS_PER_CHAR ?? 200);
	          const minMs = Number(import.meta.env.VITE_TTS_SEGMENT_MIN_MS ?? 900);
	          const maxMs = Number(import.meta.env.VITE_TTS_SEGMENT_MAX_MS ?? 10000);
	          const current = segments[segmentIndex] ?? displayText;
	          return clamp(minMs, current.length * perCharMs, maxMs);
	        })();

	    if (import.meta.env.DEV) {
	      const current = segments[segmentIndex] ?? displayText;
	      console.log("[TTSPanel] segment delay", {
	        segmentIndex,
	        segmentsCount: segments.length,
	        currentLen: current.length,
	        hasFixedDelay,
	        delayMs,
	      });
	    }

	    const id = window.setTimeout(() => {
	      nextSegment();
	    }, delayMs);

    return () => {
      window.clearTimeout(id);
    };
  }, [displayText, nextSegment, segmentIndex, segments, speakerPlaying]);

  if (!displayText) {
    return null;
  }

  return (
    <div
      style={{
        position: "fixed",
        top: "50%",
        left: "50%",
        transform: "translate(-50%, -50%)",
        width: "100vw",
        textAlign: "center",
        color: "#ffffff",
        fontSize: "4rem",
        fontWeight: "bold",
        textShadow: "0 0 18px rgba(0,0,0,0.9)",
        padding: "0 40px",
        boxSizing: "border-box",
        zIndex: 20,
        pointerEvents: "none",
        lineHeight: 1.2,
        whiteSpace: "pre-wrap",
      }}
    >
      {displayText}
    </div>
  );
}
