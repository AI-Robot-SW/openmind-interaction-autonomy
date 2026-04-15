import { useEffect, useMemo } from "react";
import { createWebSocketClient } from "../wire/websocket";
import { useAudioStore } from "../stores/audioStore";

export function useVoiceSpectrum() {
  const setAudioLevel = useAudioStore((state) => state.setAudioLevel);

  const wsUrl = useMemo(() => {
    const scheme = window.location.protocol === "https:" ? "wss" : "ws";
    const base =
      import.meta.env.VITE_GUI_WS_BASE ??
      `${scheme}://${window.location.hostname}:8767`;
    
    console.log("VITE_GUI_WS_BASE:", import.meta.env.VITE_GUI_WS_BASE);
    console.log("WebSocket URL:", `${base}/voice_spectrum`);
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
      (raw) => JSON.parse(raw) as number,
    );

    client.connect();

    return () => {
      client.disconnect();
    };
  }, [wsUrl, setAudioLevel]);
}