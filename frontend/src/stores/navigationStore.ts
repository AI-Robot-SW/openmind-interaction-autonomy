import { create } from "zustand";

type NavigationState = {
  activeGoal: string | null;
  reached: boolean;
  lastReachedGoal: string | null;

  setActiveGoal: (goal: string | null) => void;
  setReached: (reached: boolean) => void;
  setLastReachedGoal: (goal: string | null) => void;
};

export const useNavigationStore = create<NavigationState>((set) => ({
  activeGoal: null,
  reached: false,
  lastReachedGoal: null,

  setActiveGoal: (goal) => set({ activeGoal: goal }),
  setReached: (reached) => set({ reached }),
  setLastReachedGoal: (goal) => set({ lastReachedGoal: goal }),
}));