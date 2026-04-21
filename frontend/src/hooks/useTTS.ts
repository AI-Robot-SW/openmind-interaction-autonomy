import { useEffect, useMemo, useRef } from "react";
import { useTTSStore } from "../stores/ttsStore";
import type { TTSStateValue } from "../stores/ttsStore";
import { getWsBase } from "../wire/getWsBase";
import { createWebSocketClient } from "../wire/websocket";

type TTSMessage = {
  text?: string;
  // Preferred key (matches current GUI_bg.py)
  state?: TTSStateValue;
  // Compatibility keys (in case an older server uses different naming)
  tts_state?: TTSStateValue;
  ttsState?: TTSStateValue;
  // Optional playback info (GUI_bg.py may include this)
  speaker_playing?: boolean;
  speakerPlaying?: boolean;
};

export function useTTS() {
  const setTTSState = useTTSStore((state) => state.setTTSState);
  const warnedMissingStateRef = useRef(false);

  const wsUrl = useMemo(() => {
    return `${getWsBase()}/tts_text`;
  }, []);

  useEffect(() => {
    const client = createWebSocketClient<TTSMessage>(
      wsUrl,
      {
        onMessage: (msg) => {
          const nextState = msg.state ?? msg.tts_state ?? msg.ttsState;
          const speakerPlaying = msg.speaker_playing ?? msg.speakerPlaying;
          if (import.meta.env.DEV && nextState == null && !warnedMissingStateRef.current) {
            warnedMissingStateRef.current = true;
            console.warn("[useTTS] Missing `state` in WS message; defaulting to `idle`.", msg);
          }
          setTTSState({
            displayText: msg.text?.trim() ?? "",
            state: nextState ?? "idle",
            speakerPlaying: Boolean(speakerPlaying),
          });
        },
        onStatus: (s) => {
          if (!import.meta.env.DEV) return;
          console.log("[useTTS] ws status", { url: wsUrl, status: s });
        },
        onError: (e) => {
          if (!import.meta.env.DEV) return;
          console.warn("[useTTS] ws error", { url: wsUrl, event: e });
        },
      },
      (raw) => JSON.parse(raw) as TTSMessage,
    );

    client.connect();

    return () => {
      client.disconnect();
    };
  }, [setTTSState, wsUrl]);
}
