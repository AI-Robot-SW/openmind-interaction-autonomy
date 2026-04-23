import AudioPanel from "./components/AudioPanel";
import DestinationPanel from "./components/DestinationPanel";
import StandbyPanel from "./components/StandbyPanel";
import TTSPanel from "./components/TTSPanel";
import { useVoiceSpectrum } from "./hooks/useVoiceSpectrum";
import { useNavigation } from "./hooks/useNavigation";
import { useTTS } from "./hooks/useTTS";
import { useSpeechUiState } from "./hooks/useSpeechUiState";

export default function App() {
  useVoiceSpectrum();
  useNavigation();
  useTTS();

  const { robotState } = useSpeechUiState();

  return (
    <>
      <DestinationPanel />
      {robotState === "talking" ? <TTSPanel /> : null}
      {robotState === "thinking" ? <StandbyPanel /> : null}
      {robotState === "hearing" ? <AudioPanel /> : null}
    </>
  );
}
