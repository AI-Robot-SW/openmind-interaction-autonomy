export function getWsBase() {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";

  return (
    import.meta.env.VITE_GUI_WS_BASE ??
    `${scheme}://${window.location.hostname}:8767`
  );
}