import { create } from "zustand";

type AudioStore = {
  audioLevel: number;
  setAudioLevel: (value: number) => void;
};

export const useAudioStore = create<AudioStore>((set) => ({
  audioLevel: 0,
  setAudioLevel: (value) => set({ audioLevel: value }),
}));