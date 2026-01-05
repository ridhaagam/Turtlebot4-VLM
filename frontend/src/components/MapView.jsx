import React, { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { Map, MapPin, Navigation, Compass, RotateCw } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'

// Fetch map status
async function fetchMapStatus() {
  const response = await fetch('/api/map/status')
  if (!response.ok) throw new Error('Failed to fetch map status')
  return response.json()
}

function MapView() {
  const [imageUrl, setImageUrl] = useState(null)
  const [lastUpdate, setLastUpdate] = useState(null)
  const [isLive, setIsLive] = useState(false)
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 })
  const containerRef = useRef(null)
  const imageRef = useRef(null)

  // Poll map status
  const { data: mapStatus } = useQuery({
    queryKey: ['mapStatus'],
    queryFn: fetchMapStatus,
    refetchInterval: 5000,
    retry: false,
  })

  // Update map every 5 seconds
  useEffect(() => {
    const fetchMap = async () => {
      try {
        const response = await fetch('/api/map/latest', {
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
    fetchMap()

    // Poll every 5 seconds
    const interval = setInterval(fetchMap, 5000)

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

  // Handle image load to get actual dimensions
  const handleImageLoad = (e) => {
    const img = e.target
    setImageDimensions({
      width: img.naturalWidth,
      height: img.naturalHeight,
    })
  }

  // Calculate robot position on the map
  const calculateRobotPosition = () => {
    if (!mapStatus?.robot_x || !mapStatus?.robot_y || !mapStatus?.width || !mapStatus?.height) {
      return null
    }

    // Map dimensions in meters
    const mapWidthMeters = mapStatus.width * mapStatus.resolution
    const mapHeightMeters = mapStatus.height * mapStatus.resolution

    // Robot position relative to origin
    const robotRelX = mapStatus.robot_x - (mapStatus.origin_x || 0)
    const robotRelY = mapStatus.robot_y - (mapStatus.origin_y || 0)

    // Convert to percentage (0-100%)
    // Note: Y is inverted in image coordinates
    const percentX = (robotRelX / mapWidthMeters) * 100
    const percentY = (1 - robotRelY / mapHeightMeters) * 100

    // Clamp to valid range
    return {
      left: Math.max(0, Math.min(100, percentX)),
      top: Math.max(0, Math.min(100, percentY)),
      rotation: (mapStatus.robot_theta || 0) * (180 / Math.PI),
    }
  }

  const robotPos = calculateRobotPosition()

  return (
    <motion.div
      className="map-view card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: 0.1 }}
    >
      <div className="card-header">
        <div className="header-title">
          <Map size={18} />
          <h3>SLAM Map</h3>
        </div>
        <div className={`live-indicator ${isLive ? 'live' : 'offline'}`}>
          {isLive ? (
            <>
              <Navigation size={14} />
              <span>MAPPING</span>
            </>
          ) : (
            <>
              <Compass size={14} />
              <span>NO MAP</span>
            </>
          )}
        </div>
      </div>

      <div className="map-content" ref={containerRef}>
        {imageUrl ? (
          <div className="map-image-container">
            <img
              ref={imageRef}
              src={imageUrl}
              alt="SLAM Occupancy Grid"
              className="map-image"
              onLoad={handleImageLoad}
            />
            {/* Robot marker with corrected positioning */}
            {robotPos && (
              <div
                className="robot-marker"
                style={{
                  left: `${robotPos.left}%`,
                  top: `${robotPos.top}%`,
                  transform: `translate(-50%, -50%) rotate(${robotPos.rotation}deg)`,
                }}
              >
                <Navigation size={20} className="robot-icon" />
              </div>
            )}
            <div className="map-overlay">
              <span className="timestamp">{formatTime(lastUpdate)}</span>
            </div>
            {/* Map coordinate info */}
            {mapStatus?.robot_x != null && (
              <div className="map-coordinates">
                <span>X: {mapStatus.robot_x.toFixed(2)}m</span>
                <span>Y: {mapStatus.robot_y.toFixed(2)}m</span>
                {mapStatus.robot_theta != null && (
                  <span>θ: {(mapStatus.robot_theta * (180 / Math.PI)).toFixed(0)}°</span>
                )}
              </div>
            )}
          </div>
        ) : (
          <div className="no-map">
            <Map size={48} className="no-map-icon" />
            <span>SLAM Map Unavailable</span>
            <span className="hint">TurtleBot4 Lite requires LiDAR add-on</span>
            <span className="hint">Launch with enable_slam:=true if LiDAR is connected</span>
          </div>
        )}
      </div>

      {mapStatus && mapStatus.has_map && (
        <div className="map-stats">
          <div className="stat">
            <span className="label">Size</span>
            <span className="value">
              {mapStatus.width || '--'} x {mapStatus.height || '--'} px
            </span>
          </div>
          <div className="stat">
            <span className="label">Resolution</span>
            <span className="value">
              {mapStatus.resolution ? `${(mapStatus.resolution * 100).toFixed(1)} cm/px` : '--'}
            </span>
          </div>
          {mapStatus.robot_x != null && (
            <div className="stat">
              <span className="label">Robot Pos</span>
              <span className="value">
                ({mapStatus.robot_x.toFixed(2)}, {mapStatus.robot_y.toFixed(2)})
              </span>
            </div>
          )}
        </div>
      )}

      <div className="map-legend">
        <div className="legend-item">
          <span className="legend-color" style={{ background: '#000' }}></span>
          <span>Obstacle</span>
        </div>
        <div className="legend-item">
          <span className="legend-color" style={{ background: '#fff' }}></span>
          <span>Free</span>
        </div>
        <div className="legend-item">
          <span className="legend-color" style={{ background: '#888' }}></span>
          <span>Unknown</span>
        </div>
      </div>
    </motion.div>
  )
}

export default MapView
