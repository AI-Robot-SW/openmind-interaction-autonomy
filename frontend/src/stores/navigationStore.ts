import { create } from "zustand";

type NavigationState = {
  activeGoal: string | null;
  reached: boolean;
  lastReachedGoal: string | null;

  setActiveGoal: (goal: string | null) => void;
  setReached: (reached: boolean) => void;
  setLastReachedGoal: (goal: string | null) => void;
  getDisplayName: (goal: string | null) => string | null;
};

// Mapping from backend place_id to frontend display name
const destinationMapping: Record<string, string> = {
  "a0": "본관",
  "l8": "L8",
  "north_gate": "북문",
};

export const useNavigationStore = create<NavigationState>((set) => ({
  activeGoal: null,
  reached: false,
  lastReachedGoal: null,

  setActiveGoal: (goal) => set({ activeGoal: goal }),
  setReached: (reached) => set({ reached }),
  setLastReachedGoal: (goal) => set({ lastReachedGoal: goal }),
  getDisplayName: (goal) => {
    if (!goal) return null;
    return destinationMapping[goal] || goal; // Fallback to original if not mapped
  },
}));
