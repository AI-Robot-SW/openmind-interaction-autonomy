import { useEffect, useMemo, useRef } from "react";
import { createWebSocketClient } from "../wire/websocket";
import { useAudioStore } from "../stores/audioStore";
import { getWsBase } from "../wire/getWsBase";

type AudioMessage = {
  level?: number;
  // Preferred key (matches current GUI_bg.py)
  voice_active?: boolean;
  // Compatibility key (camelCase)
  voiceActive?: boolean;
};

export function useVoiceSpectrum() {
  const setAudioState = useAudioStore((state) => state.setAudioState);
  const warnedMissingVadKeyRef = useRef(false);

  const wsUrl = useMemo(() => {
    return `${getWsBase()}/voice_spectrum`;
  }, []);

  useEffect(() => {
    const client = createWebSocketClient<AudioMessage>(
      wsUrl,
      {
        onMessage: (msg) => {
          const level = Number(msg.level);
          const voiceActive = msg.voice_active ?? msg.voiceActive;
          if (
            import.meta.env.DEV &&
            voiceActive == null &&
            !warnedMissingVadKeyRef.current
          ) {
            warnedMissingVadKeyRef.current = true;
            console.warn(
              "[useVoiceSpectrum] Missing `voice_active` in WS message; defaulting to `false`.",
              msg,
            );
          }
          setAudioState({
            audioLevel: Number.isFinite(level) ? Math.abs(level) : 0,
            voiceActive: Boolean(voiceActive),
          });
        },
        onStatus: (s) => {
          if (!import.meta.env.DEV) return;
          console.log("[useVoiceSpectrum] ws status", { url: wsUrl, status: s });
        },
        onError: (e) => {
          if (!import.meta.env.DEV) return;
          console.warn("[useVoiceSpectrum] ws error", { url: wsUrl, event: e });
        },
      },
      (raw) => JSON.parse(raw) as AudioMessage,
    );

    client.connect();

    return () => {
      client.disconnect();
    };
  }, [wsUrl, setAudioState]);
}
