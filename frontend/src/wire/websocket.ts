export type WsStatus = 'connecting' | 'connected' | 'closed' | 'error'

type Handlers<T> = {
  onMessage: (msg: T) => void
  onStatus?: (s: WsStatus) => void
  onError?: (e: Event) => void
}

export function createWebSocketClient<T = string>(
  url: string,
  handlers: Handlers<T>,
  parse: (raw: string) => T,
) {
  let ws: WebSocket | null = null
  let status: WsStatus = 'closed'

  const setStatus = (next: WsStatus) => {
    status = next
    handlers.onStatus?.(next)
  }

  const connect = () => {
    setStatus('connecting')
    ws = new WebSocket(url)

    ws.onopen = () => setStatus('connected')
    ws.onmessage = (event) => handlers.onMessage(parse(String(event.data)))
    ws.onerror = (event) => {
      setStatus('error')
      handlers.onError?.(event)
    }
    ws.onclose = () => setStatus('closed')
  }

  const disconnect = () => ws?.close()
  const send = (data: string) => ws?.send(data)
  const getStatus = () => status

  return { connect, disconnect, send, getStatus }
}
