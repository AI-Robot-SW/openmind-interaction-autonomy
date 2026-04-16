import { useEffect, useMemo } from "react";
import { createWebSocketClient } from "../wire/websocket";
import { useAudioStore } from "../stores/audioStore";
import { getWsBase } from "../wire/getWsBase";

export function useVoiceSpectrum() {
  const setAudioLevel = useAudioStore((state) => state.setAudioLevel);

    const wsUrl = useMemo(() => {
    return `${getWsBase()}/voice_spectrum`;
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