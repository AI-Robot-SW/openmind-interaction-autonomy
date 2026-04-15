import { useAudioStore } from "../stores/audioStore";
import "../App.css";

export default function AudioPanel() {
  const audioLevel = useAudioStore((state) => state.audioLevel);

  const normalized = Math.max(0, Math.min(1, audioLevel * 6));
  const scale = 0.7 + normalized * 1.8;
  const glow = 10 + normalized;

  return (
    <main className="screen">
      <div className="orb-stage">
        <div
          className="orb-core"
          style={{
            transform: `scale(${scale})`,
            boxShadow: `0 0 ${glow}px rgba(255,255,255,0.9), 0 0 ${glow * 2.4}px rgba(255,255,255,0.35)`,
          }}
        />
      </div>
    </main>
  );
}