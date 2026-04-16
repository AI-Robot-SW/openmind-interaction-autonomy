import AudioPanel from "./components/AudioPanel";
import DestinationPanel from "./components/DestinationPanel";
import { useVoiceSpectrum } from "./hooks/useVoiceSpectrum";
import { useNavigation } from "./hooks/useNavigation";

export default function App() {
  useVoiceSpectrum();
  useNavigation();

  return (
    <>
      <AudioPanel />
      <DestinationPanel />
    </>
  );
}