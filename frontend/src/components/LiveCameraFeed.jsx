import React, { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { Camera, CameraOff, Wifi, WifiOff, Box, Target, Navigation, ArrowUp, ArrowLeft, ArrowRight, CheckCircle, XCircle, RotateCcw, RotateCw } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import BoundingBoxOverlay from './BoundingBoxOverlay'

// Fetch camera status
async function fetchCameraStatus() {
  const response = await fetch('/api/camera/status')
  if (!response.ok) throw new Error('Failed to fetch camera status')
  return response.json()
}

// Fetch detection info from camera
async function fetchDetections() {
  const response = await fetch('/api/camera/detections')
  if (!response.ok) return { detections: [], has_detections: false }
  const result = await response.json()

  // Log detection info for debugging
  if (result.has_detections && result.detections?.length > 0) {
    console.log(`[DETECT] ${result.detections.length} detection(s):`, result.detections.map(d => d.label).join(', '))
  }

  return result
}

// Fetch navigation inference from vision API
async function fetchNavigationResult() {
  const response = await fetch('/api/vision/navigation/latest')
  if (!response.ok) return null
  const result = await response.json()

  // Log navigation result for debugging
  if (result && !result.is_stale) {
    console.log(`[NAV] Command: ${result.command}, Bin: ${result.bin_detected}, Pos: ${result.bin_position}, Size: ${result.bin_size}, Conf: ${(result.confidence * 100).toFixed(0)}%`)
    if (result.bboxes?.length > 0) {
      console.log(`[NAV] Bboxes: ${result.bboxes.length}`, result.bboxes)
    }
  }

  return result
}

// Navigation command icons
const COMMAND_ICONS = {
  forward: ArrowUp,
  left: ArrowLeft,
  right: ArrowRight,
  arrived: CheckCircle,
  not_found: XCircle,
  search_left: RotateCcw,
  search_right: RotateCw,
}

// Navigation command colors
const COMMAND_COLORS = {
  forward: '#22c55e',
  left: '#3b82f6',
  right: '#f59e0b',
  arrived: '#8b5cf6',
  not_found: '#6b7280',
  search_left: '#ec4899',  // Pink for search mode
  search_right: '#ec4899', // Pink for search mode
}

function LiveCameraFeed() {
  const [imageUrl, setImageUrl] = useState(null)
  const [lastUpdate, setLastUpdate] = useState(null)
  const [isLive, setIsLive] = useState(false)
  const containerRef = useRef(null)

  // Poll camera status
  const { data: cameraStatus } = useQuery({
    queryKey: ['cameraStatus'],
    queryFn: fetchCameraStatus,
    refetchInterval: 2000,
    retry: false,
  })

  // Poll detection info from camera
  const { data: detectionInfo } = useQuery({
    queryKey: ['cameraDetections'],
    queryFn: fetchDetections,
    refetchInterval: 2000,
    retry: false,
  })

  // Poll navigation inference result
  const { data: navResult } = useQuery({
    queryKey: ['navigationLatest'],
    queryFn: fetchNavigationResult,
    refetchInterval: 2000,
    retry: false,
  })

  // Update image every 2 seconds
  useEffect(() => {
    const fetchImage = async () => {
      try {
        const response = await fetch('/api/camera/latest', {
          cache: 'no-store',
        })

        if (response.ok) {
          const blob = await response.blob()
          const url = URL.createObjectURL(blob)

          // Revoke old URL to prevent memory leak
          if (imageUrl) {
            URL.revokeObjectURL(imageUrl)
          }

          setImageUrl(url)
          setLastUpdate(new Date())
          setIsLive(true)
        } else {
          setIsLive(false)
        }
      } catch (error) {
        setIsLive(false)
      }
    }

    // Initial fetch
    fetchImage()

    // Poll every 2 seconds
    const interval = setInterval(fetchImage, 2000)

    return () => {
      clearInterval(interval)
      if (imageUrl) {
        URL.revokeObjectURL(imageUrl)
      }
    }
  }, [])

  // Format timestamp
  const formatTime = (date) => {
    if (!date) return '--:--:--'
    return date.toLocaleTimeString()
  }

  // Get detections for overlay - prefer navigation bboxes, fallback to camera detections
  // If data is stale, don't show overlays
  const isNavStale = navResult?.is_stale === true
  const overlayDetections = (!isNavStale && navResult?.bboxes?.length > 0)
    ? navResult.bboxes.map(bbox => ({
        label: bbox.label || 'bin',
        confidence: bbox.confidence || 0.9,
        bbox: { x: bbox.x, y: bbox.y, width: bbox.w || bbox.width, height: bbox.h || bbox.height },
      }))
    : (detectionInfo?.detections || [])

  // Get image dimensions
  const imageWidth = detectionInfo?.width || cameraStatus?.width || 640
  const imageHeight = detectionInfo?.height || cameraStatus?.height || 480

  // Get navigation command info
  const navCommand = navResult?.command || 'not_found'
  const CommandIcon = COMMAND_ICONS[navCommand] || XCircle
  const commandColor = COMMAND_COLORS[navCommand] || '#6b7280'

  return (
    <motion.div
      className="live-camera-feed card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div className="card-header">
        <div className="header-title">
          <Camera size={18} />
          <h3>Robot Camera</h3>
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
              alt="Robot Camera Feed"
              className="camera-image"
            />
            {/* Bounding Box Overlay */}
            {overlayDetections.length > 0 && (
              <BoundingBoxOverlay
                detections={overlayDetections}
                imageWidth={imageWidth}
                imageHeight={imageHeight}
                containerRef={containerRef}
              />
            )}
            <div className="camera-overlay">
              <span className="timestamp">{formatTime(lastUpdate)}</span>
            </div>
            {/* Navigation Command Indicator */}
            {navResult && !isNavStale && navResult.command !== 'not_found' && (
              <div
                className="nav-command-indicator"
                style={{ '--cmd-color': commandColor }}
              >
                <CommandIcon size={24} />
                <span className="nav-command-text">
                  {navCommand.replace('_', ' ').toUpperCase()}
                </span>
              </div>
            )}
          </div>
        ) : (
          <div className="no-camera">
            <CameraOff size={48} className="no-camera-icon" />
            <span>No camera feed available</span>
            <span className="hint">Waiting for robot connection...</span>
          </div>
        )}
      </div>

      {/* Camera and navigation stats */}
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
        {/* Detection count */}
        <div className="stat detection-stat">
          <span className="label">
            <Target size={12} />
            Detections
          </span>
          <span className="value detection-count">
            {overlayDetections.length || 0}
          </span>
        </div>
        {/* Navigation inference time */}
        {navResult?.inference_time_ms && (
          <div className="stat">
            <span className="label">Inference</span>
            <span className="value">
              {navResult.inference_time_ms.toFixed(0)}ms
            </span>
          </div>
        )}
      </div>

      {/* Navigation status bar */}
      {navResult && !isNavStale && (
        <div className="nav-status-bar" style={{ '--cmd-color': commandColor }}>
          <div className="nav-status-item">
            <span className="nav-label">Command</span>
            <span className="nav-value" style={{ color: commandColor }}>
              <CommandIcon size={14} />
              {navCommand.replace('_', ' ').toUpperCase()}
            </span>
          </div>
          <div className="nav-status-item">
            <span className="nav-label">Bin</span>
            <span className="nav-value" style={{ color: navResult.bin_detected ? '#22c55e' : '#6b7280' }}>
              {navResult.bin_detected ? 'DETECTED' : 'SEARCHING...'}
            </span>
          </div>
          {navResult.bin_position && (
            <div className="nav-status-item">
              <span className="nav-label">Position</span>
              <span className="nav-value">{navResult.bin_position.toUpperCase()}</span>
            </div>
          )}
          {navResult.bin_size && (
            <div className="nav-status-item">
              <span className="nav-label">Size</span>
              <span className="nav-value">{navResult.bin_size.toUpperCase()}</span>
            </div>
          )}
          {navResult.confidence && (
            <div className="nav-status-item">
              <span className="nav-label">Confidence</span>
              <span className="nav-value">{(navResult.confidence * 100).toFixed(0)}%</span>
            </div>
          )}
        </div>
      )}

      {/* Detection labels overlay */}
      {overlayDetections.length > 0 && (
        <div className="detection-labels">
          {overlayDetections.slice(0, 4).map((det, idx) => (
            <div key={idx} className="detection-label">
              <Box size={12} />
              <span>{det.label}</span>
              <span className="confidence">{(det.confidence * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
      )}
    </motion.div>
  )
}

export default LiveCameraFeed
