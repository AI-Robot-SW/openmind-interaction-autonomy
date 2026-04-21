import { useEffect, useRef, useState } from "react";
import { useAudioStore } from "../stores/audioStore";
import { useTTSStore } from "../stores/ttsStore";
import type { TTSStateValue } from "../stores/ttsStore";

export type RobotState = "idle" | "hearing" | "thinking" | "talking";

export type SpeechUiState = {
  ttsPipelineState: TTSStateValue;
  speakerPlaying: boolean;
  voiceActive: boolean;
  robotState: RobotState;
  isSpeaking: boolean;
  isListening: boolean;
  isThinking: boolean;
};

// UI-facing "effective" speech state derived from raw WS telemetry.
export function useSpeechUiState(): SpeechUiState {
  const ttsPipelineState = useTTSStore((s) => s.state);
  const speakerPlaying = useTTSStore((s) => s.speakerPlaying);
  const voiceActive = useAudioStore((s) => s.voiceActive);

  // Latch "thinking" when hearing ends, so we can cover the STT/LLM gap before TTS begins.
  const prevVoiceActiveRef = useRef(voiceActive);
  const [thinkingLatched, setThinkingLatched] = useState(false);

  useEffect(() => {
    const prevVoiceActive = prevVoiceActiveRef.current;

    if (speakerPlaying || voiceActive) {
      setThinkingLatched(false);
    } else if (prevVoiceActive && !voiceActive) {
      setThinkingLatched(true);
    }

    prevVoiceActiveRef.current = voiceActive;
  }, [speakerPlaying, voiceActive]);

  // Priority order (strongest signals first):
  // 1) talking: actual speaker playback
  // 2) hearing: voice activity (when not talking)
  // 3) thinking: latched after hearing ends and/or TTS pipeline is active but speaker isn't playing yet
  // 4) idle
  const isTtsActive = ttsPipelineState === "processing" || ttsPipelineState === "speaking";
  const shouldThink =
    thinkingLatched || (isTtsActive && !speakerPlaying && !voiceActive);
  const robotState: RobotState = speakerPlaying
    ? "talking"
    : voiceActive
      ? "hearing"
      : shouldThink
        ? "thinking"
        : "idle";

  const isSpeaking = robotState === "talking";
  const isListening = robotState === "hearing";
  const isThinking = robotState === "thinking";

  return {
    ttsPipelineState,
    speakerPlaying,
    voiceActive,
    robotState,
    isSpeaking,
    isListening,
    isThinking,
  };
}
