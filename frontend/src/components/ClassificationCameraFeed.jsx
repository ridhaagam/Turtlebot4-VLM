import React, { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { Camera, CameraOff, Wifi, WifiOff, Trash2, Recycle, Leaf, Target } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'

// Fetch classification camera status
async function fetchClassificationStatus() {
  const response = await fetch('/api/camera/classification/status')
  if (!response.ok) throw new Error('Failed to fetch classification camera status')
  return response.json()
}

// Fullness level colors
const FULLNESS_COLORS = {
  'EMPTY': '#22c55e',
  '0-25%': '#22c55e',
  '25-75%': '#84cc16',
  'PARTIALLY_FULL': '#84cc16',
  '75-90%': '#f59e0b',
  '90-100%': '#ef4444',
  'FULL': '#ef4444',
  'unknown': '#6b7280',
}

// Waste type icons
const WASTE_ICONS = {
  'RECYCLABLE': Recycle,
  'ORGANIC': Leaf,
  'GENERAL': Trash2,
  'MIXED': Trash2,
}

function ClassificationCameraFeed() {
  const [imageUrl, setImageUrl] = useState(null)
  const [lastUpdate, setLastUpdate] = useState(null)
  const [isLive, setIsLive] = useState(false)
  const containerRef = useRef(null)

  // Poll classification camera status
  const { data: cameraStatus } = useQuery({
    queryKey: ['classificationCameraStatus'],
    queryFn: fetchClassificationStatus,
    refetchInterval: 2000,
    retry: false,
  })

  // Update image every 2 seconds
  useEffect(() => {
    let isMounted = true
    let currentUrl = null

    const fetchImage = async () => {
      try {
        const response = await fetch('/api/camera/classification/latest', {
          cache: 'no-store',
        })

        if (response.ok && isMounted) {
          const blob = await response.blob()
          const url = URL.createObjectURL(blob)

          // Revoke old URL to prevent memory leak
          if (currentUrl) {
            URL.revokeObjectURL(currentUrl)
          }
          currentUrl = url

          setImageUrl(url)
          setLastUpdate(new Date())
          setIsLive(true)
        } else if (isMounted) {
          setIsLive(false)
        }
      } catch (error) {
        if (isMounted) setIsLive(false)
      }
    }

    // Initial fetch
    fetchImage()

    // Poll every 2 seconds
    const interval = setInterval(fetchImage, 2000)

    return () => {
      isMounted = false
      clearInterval(interval)
      if (currentUrl) {
        URL.revokeObjectURL(currentUrl)
      }
    }
  }, [])

  // Format timestamp
  const formatTime = (date) => {
    if (!date) return '--:--:--'
    return date.toLocaleTimeString()
  }

  // Get classification result from status
  const classResult = cameraStatus?.classification_result

  // Get fullness color
  const fullnessLevel = classResult?.fullness_level || 'unknown'
  const fullnessColor = FULLNESS_COLORS[fullnessLevel] || FULLNESS_COLORS['unknown']

  // Get waste type icon
  const wasteType = classResult?.waste_type || 'GENERAL'
  const WasteIcon = WASTE_ICONS[wasteType] || Trash2

  return (
    <motion.div
      className="classification-camera-feed card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div className="card-header">
        <div className="header-title">
          <Trash2 size={18} />
          <h3>Classification Camera</h3>
        </div>
        <div className={`live-indicator ${isLive ? 'live' : 'offline'}`}>
          {isLive ? (
            <>
              <Wifi size={14} />
              <span>LIVE</span>
            </>
          ) : (
            <>
              <WifiOff size={14} />
              <span>OFFLINE</span>
            </>
          )}
        </div>
      </div>

      <div className="camera-content">
        {imageUrl ? (
          <div className="camera-image-container" ref={containerRef}>
            <img
              src={imageUrl}
              alt="Classification Camera Feed"
              className="camera-image"
            />
            <div className="camera-overlay">
              <span className="timestamp">{formatTime(lastUpdate)}</span>
            </div>
            {/* Classification Result Overlay */}
            {classResult && (
              <div
                className="classification-overlay"
                style={{ '--fullness-color': fullnessColor }}
              >
                <WasteIcon size={20} />
                <span className="fullness-level">{fullnessLevel}</span>
              </div>
            )}
          </div>
        ) : (
          <div className="no-camera">
            <CameraOff size={48} className="no-camera-icon" />
            <span>No classification camera feed</span>
            <span className="hint">Waiting for robot connection...</span>
          </div>
        )}
      </div>

      {/* Camera stats - match LiveCameraFeed style */}
      <div className="camera-stats">
        {cameraStatus && (
          <>
            <div className="stat">
              <span className="label">Resolution</span>
              <span className="value">
                {cameraStatus.width || '--'} x {cameraStatus.height || '--'}
              </span>
            </div>
            <div className="stat">
              <span className="label">Age</span>
              <span className="value">
                {cameraStatus.age_seconds
                  ? `${cameraStatus.age_seconds.toFixed(1)}s`
                  : '--'}
              </span>
            </div>
          </>
        )}
        {/* Classification status indicator */}
        <div className="stat detection-stat">
          <span className="label">
            <Target size={12} />
            Status
          </span>
          <span className="value detection-count" style={{ color: classResult ? fullnessColor : 'var(--text-tertiary)' }}>
            {classResult ? 'CLASSIFIED' : 'WAITING'}
          </span>
        </div>
      </div>

      {/* Classification result bar - shown when classification is available */}
      {classResult && (
        <div className="nav-status-bar" style={{ '--cmd-color': fullnessColor }}>
          <div className="nav-status-item">
            <span className="nav-label">Fullness</span>
            <span className="nav-value" style={{ color: fullnessColor }}>
              <WasteIcon size={14} />
              {classResult.fullness_level}
            </span>
          </div>
          {classResult.fullness_percent !== undefined && (
            <div className="nav-status-item">
              <span className="nav-label">Level</span>
              <span className="nav-value">
                {classResult.fullness_percent}%
              </span>
            </div>
          )}
          <div className="nav-status-item">
            <span className="nav-label">Waste Type</span>
            <span className="nav-value">
              {classResult.waste_type || 'UNKNOWN'}
            </span>
          </div>
          {classResult.confidence !== undefined && (
            <div className="nav-status-item">
              <span className="nav-label">Confidence</span>
              <span className="nav-value">
                {(classResult.confidence * 100).toFixed(0)}%
              </span>
            </div>
          )}
        </div>
      )}
    </motion.div>
  )
}

export default ClassificationCameraFeed
