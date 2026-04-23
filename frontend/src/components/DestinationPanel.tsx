import { useNavigationStore } from "../stores/navigationStore";
function formatDestination(
  activeGoal: string | null,
  reached: boolean,
  lastReachedGoal: string | null,
) {
  if (activeGoal && !reached) return activeGoal;
  if (reached && lastReachedGoal) return `${lastReachedGoal} 도착`;
  return " 없음";
}

export default function DestinationPanel() {
  const activeGoal = useNavigationStore((s) => s.activeGoal);
  const reached = useNavigationStore((s) => s.reached);
  const lastReachedGoal = useNavigationStore((s) => s.lastReachedGoal);

  const destination = formatDestination(activeGoal, reached, lastReachedGoal);

  return (
    <div
      style={{
        position: "fixed",
        top: "4%",
        left: "50%",
        transform: "translateX(-50%)",
        width: "100vw",
        textAlign: "center",
        fontSize: "2.8rem",
        fontWeight: "bold",
        color: "#ffffff",
        textShadow: "0 0 8px rgba(0,0,0,0.7)",
        zIndex: 30,
        pointerEvents: "none",
      }}
    >
      <span>목적지 : </span>
      <span style={{ color: "#FFD700" }}>{destination}</span>
    </div>
  );
}
