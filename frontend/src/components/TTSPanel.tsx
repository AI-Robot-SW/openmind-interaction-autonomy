import { useTTSStore } from "../stores/ttsStore";

export default function TTSPanel() {
  const displayText = useTTSStore((state) => state.displayText);

  if (!displayText) {
    return null;
  }

  return (
    <div
      style={{
        position: "fixed",
        top: "50%",
        left: "50%",
        transform: "translate(-50%, -50%)",
        width: "100vw",
        textAlign: "center",
        color: "#ffffff",
        fontSize: "4rem",
        fontWeight: "bold",
        textShadow: "0 0 18px rgba(0,0,0,0.9)",
        padding: "0 40px",
        boxSizing: "border-box",
        zIndex: 20,
        pointerEvents: "none",
        lineHeight: 1.2,
        whiteSpace: "pre-wrap",
      }}
    >
      {displayText}
    </div>
  );
}
