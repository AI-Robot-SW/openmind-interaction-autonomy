import { create } from "zustand";

type TTSStore = {
  displayText: string;
  setDisplayText: (value: string) => void;
  clearDisplayText: () => void;
};

export const useTTSStore = create<TTSStore>((set) => ({
  displayText: "",
  setDisplayText: (value) => set({ displayText: value }),
  clearDisplayText: () => set({ displayText: "" }),
}));
