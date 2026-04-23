import { create } from "zustand";

type AudioStore = {
  audioLevel: number;
  voiceActive: boolean;
  setAudioState: (value: { audioLevel: number; voiceActive: boolean }) => void;
};

export const useAudioStore = create<AudioStore>((set) => ({
  audioLevel: 0,
  voiceActive: false,
  setAudioState: (value) => set(value),
}));
