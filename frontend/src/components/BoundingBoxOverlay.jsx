import React, { useRef, useEffect, useState } from 'react'

// Color palette matching the dashboard theme
const COLORS = {
  bin: '#22c55e',      // Green for bins
  content: '#3b82f6',  // Blue for content objects inside bins
  person: '#3b82f6',   // Blue
  vehicle: '#f59e0b',  // Amber
  animal: '#8b5cf6',   // Purple
  default: '#ef4444',  // Red
}

const LABEL_COLORS = [
  '#22c55e',  // Green
  '#3b82f6',  // Blue
  '#f59e0b',  // Amber
  '#8b5cf6',  // Purple
  '#ec4899',  // Pink
  '#14b8a6',  // Teal
  '#f97316',  // Orange
  '#06b6d4',  // Cyan
]

// Get fullness color
const getFullnessColor = (fullness) => {
  if (!fullness) return null
  if (fullness.includes('0-25')) return '#22c55e'
  if (fullness.includes('25-75')) return '#84cc16'
  if (fullness.includes('75-90')) return '#f59e0b'
  if (fullness.includes('90-100')) return '#ef4444'
  return null
}

function BoundingBoxOverlay({ detections, imageWidth, imageHeight, containerRef }) {
  const canvasRef = useRef(null)
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 })

  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef?.current) {
        const rect = containerRef.current.getBoundingClientRect()
        setDimensions({ width: rect.width, height: rect.height })
      }
    }

    updateDimensions()
    window.addEventListener('resize', updateDimensions)

    // Also observe container size changes
    const observer = new ResizeObserver(updateDimensions)
    if (containerRef?.current) {
      observer.observe(containerRef.current)
    }

    return () => {
      window.removeEventListener('resize', updateDimensions)
      observer.disconnect()
    }
  }, [containerRef])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !detections?.length) return

    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Calculate scale factors
    const scaleX = dimensions.width / imageWidth
    const scaleY = dimensions.height / imageHeight

    // Use the smaller scale to maintain aspect ratio
    const scale = Math.min(scaleX, scaleY)

    // Calculate offset to center the image
    const offsetX = (dimensions.width - imageWidth * scale) / 2
    const offsetY = (dimensions.height - imageHeight * scale) / 2

    // Create label to color mapping
    const labelColors = {}
    const uniqueLabels = [...new Set(detections.map((d) => d.label))]
    uniqueLabels.forEach((label, idx) => {
      // Check for specific categories
      const labelLower = label.toLowerCase()
      if (labelLower.includes('bin') || labelLower.includes('trash') || labelLower.includes('waste')) {
        labelColors[label] = COLORS.bin
      } else if (labelLower.includes('person') || labelLower.includes('human')) {
        labelColors[label] = COLORS.person
      } else if (labelLower.includes('car') || labelLower.includes('vehicle') || labelLower.includes('truck')) {
        labelColors[label] = COLORS.vehicle
      } else if (labelLower.includes('dog') || labelLower.includes('cat') || labelLower.includes('animal')) {
        labelColors[label] = COLORS.animal
      } else {
        labelColors[label] = LABEL_COLORS[idx % LABEL_COLORS.length]
      }
    })

    detections.forEach((detection) => {
      const { label, confidence, bbox, bin_fullness, bin_fullness_percent, is_content, parent_bin_id } = detection
      let color = labelColors[label]

      // Use content color for content objects (items inside bins)
      if (is_content) {
        color = COLORS.content
      } else {
        // Override with fullness color for bins
        const fullnessColor = getFullnessColor(bin_fullness)
        if (fullnessColor) {
          color = fullnessColor
        }
      }

      // Scale and offset bounding box
      const x = offsetX + bbox.x * scale
      const y = offsetY + bbox.y * scale
      const width = bbox.width * scale
      const height = bbox.height * scale

      // Draw bounding box with rounded corners effect
      ctx.strokeStyle = color
      ctx.lineWidth = is_content ? 2 : 2.5
      ctx.lineCap = 'round'
      ctx.lineJoin = 'round'

      // Use dashed line for content objects
      if (is_content) {
        ctx.setLineDash([6, 4])
      } else {
        ctx.setLineDash([])
      }

      // Draw main rectangle
      ctx.strokeRect(x, y, width, height)

      // Draw corner accents (only for non-content objects)
      if (!is_content) {
        const cornerLen = Math.min(width, height) * 0.15
        ctx.lineWidth = 4
        ctx.setLineDash([])
        ctx.beginPath()
        // Top-left
        ctx.moveTo(x, y + cornerLen)
        ctx.lineTo(x, y)
        ctx.lineTo(x + cornerLen, y)
        // Top-right
        ctx.moveTo(x + width - cornerLen, y)
        ctx.lineTo(x + width, y)
        ctx.lineTo(x + width, y + cornerLen)
        // Bottom-right
        ctx.moveTo(x + width, y + height - cornerLen)
        ctx.lineTo(x + width, y + height)
        ctx.lineTo(x + width - cornerLen, y + height)
        // Bottom-left
        ctx.moveTo(x + cornerLen, y + height)
        ctx.lineTo(x, y + height)
        ctx.lineTo(x, y + height - cornerLen)
        ctx.stroke()
      }

      // Reset line dash
      ctx.setLineDash([])

      // Draw semi-transparent fill
      ctx.fillStyle = color + (is_content ? '10' : '15')
      ctx.fillRect(x, y, width, height)

      // Draw label background with gradient effect
      let labelText
      if (is_content) {
        labelText = `${label} (content)`
      } else if (bin_fullness) {
        labelText = `${label} ${bin_fullness}`
      } else {
        labelText = `${label} ${(confidence * 100).toFixed(0)}%`
      }

      ctx.font = is_content ? '10px "JetBrains Mono", "SF Mono", monospace' : 'bold 11px "JetBrains Mono", "SF Mono", monospace'
      const textMetrics = ctx.measureText(labelText)
      const labelHeight = is_content ? 16 : 20
      const labelPadding = is_content ? 4 : 6

      // Label background
      const gradient = ctx.createLinearGradient(x, y - labelHeight, x, y)
      gradient.addColorStop(0, color)
      gradient.addColorStop(1, color + 'dd')
      ctx.fillStyle = gradient

      const labelWidth = textMetrics.width + labelPadding * 2
      ctx.beginPath()
      ctx.roundRect(x, y - labelHeight, labelWidth, labelHeight, [4, 4, 0, 0])
      ctx.fill()

      // Draw label text
      ctx.fillStyle = '#fff'
      ctx.fillText(labelText, x + labelPadding, y - (is_content ? 4 : 6))

      // Draw fullness percentage badge for bins (not for content)
      if (!is_content && bin_fullness_percent !== undefined && bin_fullness_percent !== null) {
        const pctText = `${bin_fullness_percent}%`
        const pctWidth = ctx.measureText(pctText).width + labelPadding * 2

        ctx.fillStyle = getFullnessColor(bin_fullness) || color
        ctx.beginPath()
        ctx.roundRect(x + labelWidth + 4, y - labelHeight, pctWidth, labelHeight, [4, 4, 4, 4])
        ctx.fill()

        ctx.fillStyle = '#fff'
        ctx.fillText(pctText, x + labelWidth + 4 + labelPadding, y - 6)
      }
    })
  }, [detections, dimensions, imageWidth, imageHeight])

  if (!detections?.length) return null

  return (
    <canvas
      ref={canvasRef}
      className="bbox-overlay"
      width={dimensions.width}
      height={dimensions.height}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
      }}
    />
  )
}

export default BoundingBoxOverlay
