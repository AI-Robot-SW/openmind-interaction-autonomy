import { useEffect, useMemo, useRef } from "react";
import { useTTSStore } from "../stores/ttsStore";
import { getWsBase } from "../wire/getWsBase";
import { createWebSocketClient } from "../wire/websocket";

type TTSMessage = {
  text?: string;
};

export function useTTS() {
  const setDisplayText = useTTSStore((state) => state.setDisplayText);
  const clearDisplayText = useTTSStore((state) => state.clearDisplayText);
  const clearTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const wsUrl = useMemo(() => {
    return `${getWsBase()}/tts_text`;
  }, []);

  useEffect(() => {
    const client = createWebSocketClient<TTSMessage>(
      wsUrl,
      {
        onMessage: (msg) => {
          const text = msg.text?.trim() ?? "";

          if (clearTimerRef.current) {
            clearTimeout(clearTimerRef.current);
            clearTimerRef.current = null;
          }

          if (text) {
            setDisplayText(text);
            return;
          }

          clearTimerRef.current = setTimeout(() => {
            clearDisplayText();
            clearTimerRef.current = null;
          }, 3000);
        },
      },
      (raw) => JSON.parse(raw) as TTSMessage,
    );

    client.connect();

    return () => {
      if (clearTimerRef.current) {
        clearTimeout(clearTimerRef.current);
        clearTimerRef.current = null;
      }
      client.disconnect();
    };
  }, [clearDisplayText, setDisplayText, wsUrl]);
}
