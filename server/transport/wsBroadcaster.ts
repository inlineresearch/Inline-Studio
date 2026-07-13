/** The WebSocket broadcaster: fans server->client events out to every connected browser socket. */
import { WebSocket } from 'ws'
import type { Broadcaster } from '@main/events/broadcaster'

export class WsBroadcaster implements Broadcaster {
  private readonly clients = new Set<WebSocket>()

  add(socket: WebSocket): void {
    this.clients.add(socket)
    socket.on('close', () => this.clients.delete(socket))
  }

  send(channel: string, payload?: unknown): void {
    const message = JSON.stringify({ channel, payload })
    for (const socket of this.clients) {
      if (socket.readyState === WebSocket.OPEN) socket.send(message)
    }
  }
}
