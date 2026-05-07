import { create } from "zustand";

export type TTSStateValue = "idle" | "processing" | "speaking" | "error";

type TTSStore = {
  displayText: string;
  state: TTSStateValue;
  // Whether the speaker is actually playing audio right now (may be sent by backend).
  speakerPlaying: boolean;
  setTTSState: (value: {
    displayText: string;
    state: TTSStateValue;
    speakerPlaying?: boolean;
  }) => void;
};

export const useTTSStore = create<TTSStore>((set) => ({
  displayText: "",
  state: "idle",
  speakerPlaying: false,
  setTTSState: (value) =>
    set({
      displayText: value.displayText,
      state: value.state,
      speakerPlaying: value.speakerPlaying ?? false,
    }),
}));
