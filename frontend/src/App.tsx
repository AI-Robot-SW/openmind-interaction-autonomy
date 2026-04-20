import AudioPanel from "./components/AudioPanel";
import DestinationPanel from "./components/DestinationPanel";
import TTSPanel from "./components/TTSPanel";
import { useVoiceSpectrum } from "./hooks/useVoiceSpectrum";
import { useNavigation } from "./hooks/useNavigation";
import { useTTS } from "./hooks/useTTS";
import { useAudioStore } from "./stores/audioStore";
import { useTTSStore } from "./stores/ttsStore";

export default function App() {
  useVoiceSpectrum();
  useNavigation();
  useTTS();

  const ttsText = useTTSStore((s) => s.displayText);
  const audioLevel = useAudioStore((s) => s.audioLevel);

  const isSpeaking = Boolean(ttsText && ttsText.trim().length > 0);
  const isListening = audioLevel > 0.01;

  return (
    <>
      <DestinationPanel />
      {isSpeaking ? <TTSPanel /> : null}
      {!isSpeaking && isListening ? <AudioPanel /> : null}
    </>
  );
}
