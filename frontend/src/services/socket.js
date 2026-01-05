import { useEffect, useState } from 'react'
import { io } from 'socket.io-client'

let socket = null

export function getSocket() {
  if (!socket) {
    socket = io({
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionAttempts: 5,
    })
  }
  return socket
}

export function useSocket() {
  const [socketInstance, setSocketInstance] = useState(null)

  useEffect(() => {
    const sock = getSocket()
    setSocketInstance(sock)

    return () => {
      // Don't disconnect on unmount, keep socket alive
    }
  }, [])

  return socketInstance
}

export function disconnectSocket() {
  if (socket) {
    socket.disconnect()
    socket = null
  }
}
