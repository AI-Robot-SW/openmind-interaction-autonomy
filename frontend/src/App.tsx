import AudioPanel from "./components/AudioPanel";
import { useVoiceSpectrum } from "./hooks/useVoiceSpectrum";

export default function App() {
  useVoiceSpectrum();
  return <AudioPanel />;
}