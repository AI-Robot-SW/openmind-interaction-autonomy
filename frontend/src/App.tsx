import { useEffect, useMemo, useState } from "react";
import "./App.css";
import { createWebSocketClient } from "./wire/websocket";

export default function App() {
  const [audioLevel, setAudioLevel] = useState(0);

  const wsUrl = useMemo(() => {
    const scheme = window.location.protocol === "https:" ? "wss" : "ws";
    const base =
      import.meta.env.VITE_GUI_WS_BASE ??
      `${scheme}://${window.location.hostname}:8767`;
    return `${base}/voice_spectrum`;
  }, []);

  useEffect(() => {
    const client = createWebSocketClient<number>(
      wsUrl,
      {
        onMessage: (value) => {
          if (!Number.isFinite(value)) return;
          setAudioLevel(Math.abs(value));
        },
      },
      (raw) => JSON.parse(raw) as number, // GUIBg: json.dumps(float)
    );

    client.connect();
    return () => client.disconnect();
  }, [wsUrl]);

  const normalized = Math.max(0, Math.min(1, audioLevel * 6));
  const scale = 0.7 + normalized * 1.8;
  const glow = 10 + normalized ;

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
