import { useEffect, useMemo } from "react"; 
import { createWebSocketClient } from "../wire/websocket"; 
import { useNavigationStore } from "../stores/navigationStore"; 
import { getWsBase } 
from "../wire/getWsBase"; 

type NavigationMessage = 
  { active_goal?: string | null; reached_goal?: boolean; };

export function useNavigation() {
  const setActiveGoal = useNavigationStore((s) => s.setActiveGoal);
  const setReached = useNavigationStore((s) => s.setReached);
  const setLastReachedGoal = useNavigationStore((s) => s.setLastReachedGoal);

  const wsUrl = useMemo(() => {
    return `${getWsBase()}/navigation`;
  }, []);

  useEffect(() => {
    const client = createWebSocketClient<NavigationMessage>(
      wsUrl,
      {
        onMessage: (msg) => {
          // ⭐ 최신 active 가져오기
          const currentActive = useNavigationStore.getState().activeGoal;
          const nextActive = msg.active_goal ?? currentActive;

          // ⭐ reached 처리 (msg 기준)
          if (msg.reached_goal === true && nextActive) {
            setLastReachedGoal(nextActive);
          }

          // active 업데이트
          if (msg.active_goal !== undefined) {
            setActiveGoal(msg.active_goal);
          }

          // reached 상태
          setReached(msg.reached_goal ?? false);
        },
      },
      (raw) => JSON.parse(raw)
    );

    client.connect();
    return () => client.disconnect();
  }, [wsUrl]); // ❗ activeGoal 제거
}