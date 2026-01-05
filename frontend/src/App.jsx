import React, { useState, useEffect } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import DetectionFeed from './components/DetectionFeed'
import DetectionHistory from './components/DetectionHistory'
import StatsPanel from './components/StatsPanel'
import BinFullnessCard from './components/BinFullnessCard'
import SummaryCard from './components/SummaryCard'
import Login from './components/Login'
import ThemeToggle from './components/ThemeToggle'
import LiveCameraFeed from './components/LiveCameraFeed'
import ClassificationCameraFeed from './components/ClassificationCameraFeed'
import NavigationSummary from './components/NavigationSummary'
import { fetchLatestDetection, fetchDetections, fetchStats } from './services/api'
import { useSocket } from './services/socket'
import { LogOut, User } from 'lucide-react'

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [user, setUser] = useState(null)
  const [selectedDetection, setSelectedDetection] = useState(null)
  const [isConnected, setIsConnected] = useState(false)
  const [currentPage, setCurrentPage] = useState(1)
  const [theme, setTheme] = useState(() => {
    // Initialize from localStorage or system preference
    const saved = localStorage.getItem('theme')
    if (saved) return saved
    return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark'
  })
  const queryClient = useQueryClient()

  // Apply theme to document
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('theme', theme)
  }, [theme])

  // Toggle theme
  const toggleTheme = () => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark')
  }

  // Check for existing auth on mount
  useEffect(() => {
    const token = localStorage.getItem('auth_token')
    const savedUser = localStorage.getItem('user')
    if (token && savedUser) {
      // Verify token is still valid
      fetch('/api/auth/verify', {
        headers: { 'Authorization': `Bearer ${token}` }
      })
        .then(res => res.ok ? res.json() : Promise.reject())
        .then(() => {
          setIsAuthenticated(true)
          setUser(JSON.parse(savedUser))
        })
        .catch(() => {
          // Token invalid, clear storage
          localStorage.removeItem('auth_token')
          localStorage.removeItem('hmac_secret')
          localStorage.removeItem('user')
        })
    }
  }, [])

  // Handle login
  const handleLogin = (data) => {
    setIsAuthenticated(true)
    setUser(data.user)
  }

  // Handle logout
  const handleLogout = () => {
    const token = localStorage.getItem('auth_token')
    fetch('/api/auth/logout', {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${token}` }
    }).finally(() => {
      localStorage.removeItem('auth_token')
      localStorage.removeItem('hmac_secret')
      localStorage.removeItem('user')
      setIsAuthenticated(false)
      setUser(null)
      setSelectedDetection(null)
      queryClient.clear()
    })
  }

  // Socket connection for real-time updates
  const socket = useSocket()

  useEffect(() => {
    if (socket && isAuthenticated) {
      socket.on('connect', () => setIsConnected(true))
      socket.on('disconnect', () => setIsConnected(false))
      socket.on('new_detection', (data) => {
        setSelectedDetection(data)
        queryClient.invalidateQueries({ queryKey: ['detections'] })
        queryClient.invalidateQueries({ queryKey: ['stats'] })
        queryClient.invalidateQueries({ queryKey: ['latestDetection'] })
      })

      return () => {
        socket.off('connect')
        socket.off('disconnect')
        socket.off('new_detection')
      }
    }
  }, [socket, queryClient, isAuthenticated])

  // Fetch latest detection
  const { data: latestDetection } = useQuery({
    queryKey: ['latestDetection'],
    queryFn: fetchLatestDetection,
    refetchInterval: 2000,
    enabled: isAuthenticated,
  })

  // Update selected detection when latest changes
  useEffect(() => {
    if (latestDetection && !selectedDetection) {
      setSelectedDetection(latestDetection)
    }
  }, [latestDetection, selectedDetection])

  // Fetch detection history with pagination (5 items per page)
  const { data: detectionHistory, isLoading: historyLoading } = useQuery({
    queryKey: ['detections', currentPage],
    queryFn: () => fetchDetections({ page: currentPage, per_page: 5 }),
    refetchInterval: 5000,
    enabled: isAuthenticated,
  })

  // Handle page change
  const handlePageChange = (newPage) => {
    setCurrentPage(newPage)
  }

  // Fetch stats
  const { data: stats } = useQuery({
    queryKey: ['stats'],
    queryFn: fetchStats,
    refetchInterval: 10000,
    enabled: isAuthenticated,
  })

  // Use selected detection or latest
  const displayDetection = selectedDetection || latestDetection

  // Extract pagination info
  const pagination = detectionHistory ? {
    page: detectionHistory.page,
    pages: detectionHistory.pages,
    total: detectionHistory.total,
    has_next: detectionHistory.has_next,
    has_prev: detectionHistory.has_prev,
  } : null

  // Show login if not authenticated
  if (!isAuthenticated) {
    return <Login onLogin={handleLogin} />
  }

  return (
    <motion.div
      className="dashboard"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <motion.header
        className="header"
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.1, duration: 0.4 }}
      >
        <h1>
          <span>Waste Bin Monitor</span>
        </h1>

        <div className="header-right">
          <div className="status-indicator">
            <span className={`status-dot ${isConnected ? 'connected' : 'disconnected'}`}></span>
            <span>{isConnected ? 'LIVE' : 'POLLING'}</span>
          </div>

          <ThemeToggle theme={theme} onToggle={toggleTheme} />

          <div className="user-menu">
            <div className="user-info">
              <User size={14} />
              <span>{user?.username}</span>
            </div>
            <button className="logout-btn" onClick={handleLogout} title="Sign out">
              <LogOut size={16} />
            </button>
          </div>
        </div>
      </motion.header>

      <motion.main
        className="main-content"
        initial={{ x: -20, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ delay: 0.2, duration: 0.4 }}
      >
        {/* Robot Vision Panel - 2 column layout */}
        <div className="vision-panel-grid dual-camera">
          <LiveCameraFeed />
          <ClassificationCameraFeed />
        </div>
        <NavigationSummary />

        {/* Detection Feed with dual image display */}
        <DetectionFeed detection={displayDetection} />

        {/* Bottom section: Fullness + Summary */}
        <div className="bottom-panel">
          <BinFullnessCard detection={displayDetection} />
          <SummaryCard detection={displayDetection} />
        </div>
      </motion.main>

      <motion.aside
        className="sidebar"
        initial={{ x: 20, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ delay: 0.3, duration: 0.4 }}
      >
        <StatsPanel stats={stats} />
        <DetectionHistory
          detections={detectionHistory?.detections || []}
          loading={historyLoading}
          selectedId={selectedDetection?.id}
          onSelect={setSelectedDetection}
          pagination={pagination}
          onPageChange={handlePageChange}
        />
      </motion.aside>
    </motion.div>
  )
}

export default App
