import { create } from "zustand";

export type TTSStateValue = "idle" | "processing" | "speaking" | "error";

function splitTtsText(raw: string): string[] {
  const text = raw.trim();
  if (!text) return [];
  return text
    .split(/[,.!?]+/)
    .map((s) => s.trim())
    .filter(Boolean);
}

type TTSStore = {
  // Full raw text received from backend (trimmed).
  fullText: string;
  // Text currently shown on screen (may be a segment of fullText).
  displayText: string;
  // Split segments derived from fullText.
  segments: string[];
  segmentIndex: number;
  state: TTSStateValue;
  // Whether the speaker is actually playing audio right now (may be sent by backend).
  speakerPlaying: boolean;
  setTTSState: (value: {
    // Prefer passing `text` (raw backend text). `displayText` is kept for backward compatibility.
    text?: string;
    displayText?: string;
    state: TTSStateValue;
    speakerPlaying?: boolean;
  }) => void;
  nextSegment: () => void;
  resetSegments: () => void;
};

export const useTTSStore = create<TTSStore>((set, get) => ({
  fullText: "",
  displayText: "",
  segments: [],
  segmentIndex: 0,
  state: "idle",
  speakerPlaying: false,
  setTTSState: (value) => {
    const prev = get();
    const nextTextRaw = value.text ?? value.displayText ?? "";
    const nextText = nextTextRaw.trim();
    const speakerPlaying = value.speakerPlaying ?? prev.speakerPlaying ?? false;

    if (nextText === prev.fullText) {
      set({
        state: value.state,
        speakerPlaying,
      });
      return;
    }

    const segments = splitTtsText(nextText);
    const displayText = segments[0] ?? "";
    set({
      fullText: nextText,
      displayText,
      segments,
      segmentIndex: 0,
      state: value.state,
      speakerPlaying,
    });
  },
  nextSegment: () =>
    set((s) => {
      if (s.segmentIndex >= s.segments.length - 1) return {};
      const nextIndex = s.segmentIndex + 1;
      return {
        segmentIndex: nextIndex,
        displayText: s.segments[nextIndex] ?? "",
      };
    }),
  resetSegments: () =>
    set({
      fullText: "",
      displayText: "",
      segments: [],
      segmentIndex: 0,
    }),
}));
