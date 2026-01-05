import React from 'react'
import { motion } from 'framer-motion'
import { FileText, Activity, Trash2, Clock } from 'lucide-react'
import { format } from 'date-fns'

function SummaryCard({ detection }) {
  const {
    timestamp,
    frame_id,
    detections = [],
    inference_time_ms,
    bin_detected = false,
    bin_count = 0,
    overall_fullness,
    overall_fullness_percent,
    status_summary,
    device_id,
  } = detection || {}

  // Generate summary text
  const generateSummary = () => {
    if (!detection) {
      return 'Waiting for detection data from edge device...'
    }

    const objectCount = detections.length
    const binObjects = detections.filter(d =>
      d.label?.toLowerCase().includes('bin') ||
      d.label?.toLowerCase().includes('trash') ||
      d.label?.toLowerCase().includes('waste')
    )

    let summary = ''

    if (objectCount === 0) {
      summary = 'No objects detected in the current frame.'
    } else {
      // Get unique labels
      const labels = [...new Set(detections.map(d => d.label))]
      const labelCounts = labels.map(label => ({
        label,
        count: detections.filter(d => d.label === label).length,
      }))

      summary = `Detected ${objectCount} object${objectCount !== 1 ? 's' : ''}: `
      summary += labelCounts
        .map(({ label, count }) => `${count} ${label}${count !== 1 ? 's' : ''}`)
        .join(', ')
      summary += '.'
    }

    if (bin_detected && overall_fullness) {
      summary += ` Waste bin${bin_count > 1 ? 's' : ''} detected with overall fullness at ${overall_fullness} (${overall_fullness_percent || 0}%).`

      if (overall_fullness_percent >= 90) {
        summary += ' Immediate attention required!'
      } else if (overall_fullness_percent >= 75) {
        summary += ' Consider scheduling pickup soon.'
      }
    }

    return status_summary || summary
  }

  const formattedTime = timestamp
    ? format(new Date(timestamp), 'HH:mm:ss')
    : '--:--:--'

  const formattedDate = timestamp
    ? format(new Date(timestamp), 'MMM dd, yyyy')
    : '--'

  return (
    <motion.div
      className="summary-card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2, duration: 0.4 }}
    >
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 'var(--space-sm)',
        marginBottom: 'var(--space-md)',
      }}>
        <FileText size={18} style={{ color: 'var(--accent-info)' }} />
        <h3 style={{ margin: 0 }}>Situation Summary</h3>
      </div>

      <motion.div
        className="summary-text"
        key={detection?.id}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.3 }}
        style={{ whiteSpace: 'pre-line' }}
      >
        {generateSummary().split('\n').map((line, idx) => (
          <div key={idx} style={{
            padding: '4px 0',
            borderBottom: idx < generateSummary().split('\n').length - 1 ? '1px solid var(--border-subtle)' : 'none',
          }}>
            {line.startsWith('Action:') ? (
              <span style={{ color: line.includes('CRITICAL') || line.includes('Immediate') ? 'var(--accent-danger)' : line.includes('Schedule') ? 'var(--accent-warning)' : 'var(--text-secondary)' }}>
                {line}
              </span>
            ) : line.startsWith('Fill Level:') ? (
              <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                {line}
              </span>
            ) : line.startsWith('Contents:') ? (
              <span style={{ color: 'var(--accent-info)' }}>
                {line}
              </span>
            ) : line.startsWith('Scene:') ? (
              <span style={{ color: 'var(--text-tertiary)', fontSize: '0.85rem', fontStyle: 'italic' }}>
                {line}
              </span>
            ) : (
              <span>{line}</span>
            )}
          </div>
        ))}
      </motion.div>

      <div className="summary-meta">
        <div className="summary-meta-item">
          <span className="summary-meta-label">
            <Clock size={10} style={{ marginRight: 4 }} />
            Time
          </span>
          <span className="summary-meta-value">{formattedTime}</span>
        </div>

        <div className="summary-meta-item">
          <span className="summary-meta-label">
            <Activity size={10} style={{ marginRight: 4 }} />
            Inference
          </span>
          <span className="summary-meta-value">
            {inference_time_ms?.toFixed(1) || '--'} ms
          </span>
        </div>

        <div className="summary-meta-item">
          <span className="summary-meta-label">
            <Trash2 size={10} style={{ marginRight: 4 }} />
            Bins
          </span>
          <span className="summary-meta-value" style={{
            color: bin_count > 0 ? 'var(--accent-primary)' : undefined
          }}>
            {bin_count || 0}
          </span>
        </div>

        {device_id && (
          <div className="summary-meta-item">
            <span className="summary-meta-label">Device</span>
            <span className="summary-meta-value">
              {device_id.substring(0, 8)}
            </span>
          </div>
        )}
      </div>

      {/* Quick status indicators */}
      {detection && (
        <div style={{
          display: 'flex',
          gap: 'var(--space-sm)',
          marginTop: 'var(--space-md)',
          flexWrap: 'wrap',
        }}>
          <StatusBadge
            label="Detection"
            status={detections.length > 0 ? 'success' : 'idle'}
            value={`${detections.length} obj`}
          />
          <StatusBadge
            label="Bins"
            status={bin_detected ? (overall_fullness_percent >= 75 ? 'warning' : 'success') : 'idle'}
            value={bin_detected ? overall_fullness : 'None'}
          />
          <StatusBadge
            label="Performance"
            status={inference_time_ms && inference_time_ms < 100 ? 'success' : 'warning'}
            value={typeof inference_time_ms === 'number' ? `${inference_time_ms.toFixed(0)}ms` : '--'}
          />
        </div>
      )}
    </motion.div>
  )
}

function StatusBadge({ label, status, value }) {
  const statusColors = {
    success: 'var(--accent-primary)',
    warning: 'var(--accent-warning)',
    error: 'var(--accent-danger)',
    idle: 'var(--text-tertiary)',
  }

  const bgColors = {
    success: 'rgba(34, 197, 94, 0.1)',
    warning: 'rgba(245, 158, 11, 0.1)',
    error: 'rgba(239, 68, 68, 0.1)',
    idle: 'rgba(110, 118, 129, 0.1)',
  }

  return (
    <motion.div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 'var(--space-xs)',
        padding: 'var(--space-xs) var(--space-sm)',
        background: bgColors[status],
        borderRadius: 'var(--radius-sm)',
        fontSize: '0.75rem',
      }}
      whileHover={{ scale: 1.02 }}
    >
      <span style={{
        width: 6,
        height: 6,
        borderRadius: '50%',
        background: statusColors[status],
      }} />
      <span style={{ color: 'var(--text-tertiary)' }}>{label}:</span>
      <span style={{
        color: statusColors[status],
        fontFamily: 'var(--font-mono)',
        fontWeight: 600,
      }}>
        {value}
      </span>
    </motion.div>
  )
}

export default SummaryCard
