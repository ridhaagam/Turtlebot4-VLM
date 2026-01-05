import React, { useRef, useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Camera, Box, Clock, Cpu, ImageOff } from 'lucide-react'
import BoundingBoxOverlay from './BoundingBoxOverlay'
import { format } from 'date-fns'

function DetectionFeed({ detection }) {
  const bboxContainerRef = useRef(null)

  if (!detection) {
    return (
      <motion.div
        className="detection-feed"
        initial={{ opacity: 0, scale: 0.98 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.3 }}
      >
        <div className="feed-header">
          <h2>Live Detection Feed</h2>
        </div>
        <div className="image-display-grid">
          <div className="image-panel">
            <span className="image-panel-label">Original</span>
            <div className="image-container">
              <div className="no-image">
                <ImageOff size={48} className="no-image-icon" />
                <span>Waiting for detections...</span>
              </div>
            </div>
          </div>
          <div className="image-panel">
            <span className="image-panel-label">Detection</span>
            <div className="image-container">
              <div className="no-image">
                <Box size={48} className="no-image-icon" />
                <span>No bounding boxes</span>
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    )
  }

  const {
    id,
    timestamp,
    frame_id,
    detections = [],
    inference_time_ms,
    image_width,
    image_height,
    image_path,
    bin_detected,
    bin_count,
    overall_fullness,
    overall_fullness_percent,
  } = detection

  // Generate image URLs from detection ID
  const imageUrl = id ? `/api/detections/${id}/image` : null
  const bboxImageUrl = id ? `/api/detections/${id}/image/bbox` : null

  const formattedTime = timestamp
    ? format(new Date(timestamp), 'HH:mm:ss')
    : '--:--:--'

  const formattedDate = timestamp
    ? format(new Date(timestamp), 'MMM dd, yyyy')
    : '--'

  const objectCount = detections.length

  // Get fullness color class
  const getFullnessColorClass = (fullness) => {
    if (!fullness) return ''
    if (fullness.includes('0-25')) return 'empty'
    if (fullness.includes('25-75')) return 'partial'
    if (fullness.includes('75-90')) return 'mostly-full'
    if (fullness.includes('90-100')) return 'full'
    return ''
  }

  return (
    <motion.div
      className="detection-feed"
      key={id}
      initial={{ opacity: 0.8 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.2 }}
    >
      <div className="feed-header">
        <h2>Live Detection Feed</h2>
        <div className="feed-stats">
          <span><Box size={12} style={{ marginRight: 4 }} />{objectCount} objects</span>
          <span><Cpu size={12} style={{ marginRight: 4 }} />{inference_time_ms?.toFixed(1) || '--'} ms</span>
          <span><Clock size={12} style={{ marginRight: 4 }} />{formattedTime}</span>
        </div>
      </div>

      {/* Dual image display: Original + Bounding Boxes */}
      <div className="image-display-grid">
        {/* Panel 1: Original Image */}
        <div className="image-panel">
          <span className="image-panel-label">
            <Camera size={10} style={{ marginRight: 4 }} />
            Original
          </span>
          <div className="image-container">
            {imageUrl ? (
              <motion.img
                key={imageUrl}
                src={imageUrl}
                alt="Original frame"
                className="detection-image"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.3 }}
                onError={(e) => {
                  e.target.style.display = 'none'
                }}
              />
            ) : (
              <div className="no-image">
                <ImageOff size={48} className="no-image-icon" />
                <span>No image available</span>
              </div>
            )}
          </div>
        </div>

        {/* Panel 2: Pre-annotated Image with Bounding Boxes */}
        <div className="image-panel" ref={bboxContainerRef}>
          <span className="image-panel-label">
            <Box size={10} style={{ marginRight: 4 }} />
            Detection
          </span>
          <div className="image-container">
            {bboxImageUrl ? (
              <motion.img
                key={`${bboxImageUrl}`}
                src={bboxImageUrl}
                alt="Detection frame with bboxes"
                className="detection-image"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.3 }}
                onError={(e) => {
                  // Fallback to original image if bbox image fails
                  e.target.src = imageUrl
                }}
              />
            ) : (
              <div className="no-image">
                <Box size={48} className="no-image-icon" />
                <span>{objectCount > 0 ? `${objectCount} objects detected` : 'No detections'}</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Detection metadata */}
      <div className="detection-info">
        <motion.div
          className="info-card"
          whileHover={{ scale: 1.02 }}
          transition={{ type: 'spring', stiffness: 400 }}
        >
          <label>Timestamp</label>
          <span>{formattedTime}</span>
        </motion.div>
        <motion.div
          className="info-card"
          whileHover={{ scale: 1.02 }}
          transition={{ type: 'spring', stiffness: 400 }}
        >
          <label>Date</label>
          <span>{formattedDate}</span>
        </motion.div>
        <motion.div
          className="info-card"
          whileHover={{ scale: 1.02 }}
          transition={{ type: 'spring', stiffness: 400 }}
        >
          <label>Resolution</label>
          <span>{image_width || '--'}x{image_height || '--'}</span>
        </motion.div>
        <motion.div
          className="info-card"
          whileHover={{ scale: 1.02 }}
          transition={{ type: 'spring', stiffness: 400 }}
        >
          <label>Bins Found</label>
          <span style={{ color: bin_count > 0 ? 'var(--accent-primary)' : undefined }}>
            {bin_count || 0}
          </span>
        </motion.div>
      </div>

      {/* Detected objects list - Separate containers from content */}
      <AnimatePresence mode="wait">
        {detections.length > 0 && (
          <motion.div
            className="detection-list"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.2 }}
          >
            {/* Container/Bin objects */}
            <h3>Containers</h3>
            <div className="object-tags">
              {detections.filter(det => !det.is_content).map((det, idx) => (
                <motion.div
                  key={`container-${det.label}-${idx}`}
                  className="object-tag"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: idx * 0.05 }}
                  whileHover={{ scale: 1.05 }}
                >
                  <span className="object-tag-label">{det.label}</span>
                  <span className="object-tag-confidence">
                    {typeof det.confidence === 'number' ? (det.confidence * 100).toFixed(0) : '90'}%
                  </span>
                  {det.bin_fullness && (
                    <span
                      className={`object-tag-fullness fullness-badge ${getFullnessColorClass(det.bin_fullness)}`}
                    >
                      {det.bin_fullness}
                    </span>
                  )}
                </motion.div>
              ))}
            </div>

            {/* Content objects (inside bins) */}
            {detections.some(det => det.is_content) && (
              <>
                <h3 style={{ marginTop: 'var(--space-md)', color: 'var(--accent-info)' }}>
                  Contents Inside Bins
                </h3>
                <div className="object-tags">
                  {detections.filter(det => det.is_content).map((det, idx) => (
                    <motion.div
                      key={`content-${det.label}-${idx}`}
                      className="object-tag content-tag"
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: idx * 0.05 }}
                      whileHover={{ scale: 1.05 }}
                      style={{
                        borderColor: 'var(--accent-info)',
                        background: 'rgba(59, 130, 246, 0.05)'
                      }}
                    >
                      <span className="object-tag-label">{det.label}</span>
                      <span className="object-tag-confidence">
                        {typeof det.confidence === 'number' ? (det.confidence * 100).toFixed(0) : '85'}%
                      </span>
                      {det.parent_bin_id !== undefined && (
                        <span className="content-parent-badge" style={{
                          fontSize: '0.65rem',
                          padding: '1px 6px',
                          background: 'rgba(59, 130, 246, 0.15)',
                          borderRadius: '4px',
                          color: 'var(--accent-info)'
                        }}>
                          Bin #{det.parent_bin_id + 1}
                        </span>
                      )}
                    </motion.div>
                  ))}
                </div>
              </>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

export default DetectionFeed
