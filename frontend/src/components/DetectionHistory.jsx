import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { History, Clock, Trash2, ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from 'lucide-react'
import { format, formatDistanceToNow } from 'date-fns'

function DetectionHistory({ detections, loading, selectedId, onSelect, pagination, onPageChange }) {
  // Get fullness color
  const getFullnessColor = (fullness, percent) => {
    if (!fullness && !percent) return 'var(--text-tertiary)'
    const pct = percent || 0
    if (fullness?.includes('0-25') || pct < 25) return 'var(--fullness-empty)'
    if (fullness?.includes('25-75') || pct < 75) return 'var(--fullness-partial)'
    if (fullness?.includes('75-90') || pct < 90) return 'var(--fullness-mostly)'
    return 'var(--fullness-full)'
  }

  if (loading) {
    return (
      <motion.div
        className="history-panel"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        <h2>
          <History size={16} style={{ marginRight: 8 }} />
          Detection History
        </h2>
        <div className="loading">
          <div className="spinner" />
          <span className="loading-text">Loading history...</span>
        </div>
      </motion.div>
    )
  }

  const { page = 1, pages = 1, total = 0, has_next = false, has_prev = false } = pagination || {}

  return (
    <motion.div
      className="history-panel"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2, duration: 0.4 }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-md)' }}>
        <h2 style={{ margin: 0 }}>
          <History size={16} style={{ marginRight: 8 }} />
          Detection History
        </h2>
        {total > 0 && (
          <span style={{
            fontSize: '0.75rem',
            color: 'var(--text-tertiary)',
            fontFamily: 'var(--font-mono)',
          }}>
            {total} total
          </span>
        )}
      </div>

      <div className="history-list">
        <AnimatePresence mode="popLayout">
          {detections.length === 0 ? (
            <motion.div
              className="empty-state"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              style={{ padding: 'var(--space-xl)' }}
            >
              <Clock size={40} className="empty-state-icon" />
              <span className="empty-state-title">No detections yet</span>
              <span className="empty-state-text">
                Waiting for data from edge device
              </span>
            </motion.div>
          ) : (
            detections.map((detection, idx) => (
              <HistoryItem
                key={detection.id}
                detection={detection}
                isActive={detection.id === selectedId}
                onClick={() => onSelect(detection)}
                index={idx}
                getFullnessColor={getFullnessColor}
              />
            ))
          )}
        </AnimatePresence>
      </div>

      {/* Pagination controls */}
      {pages > 1 && (
        <div className="pagination">
          <button
            className="pagination-btn"
            onClick={() => onPageChange?.(1)}
            disabled={page === 1}
            title="First page"
          >
            <ChevronsLeft size={14} />
          </button>
          <button
            className="pagination-btn"
            onClick={() => onPageChange?.(page - 1)}
            disabled={!has_prev}
            title="Previous page"
          >
            <ChevronLeft size={14} />
          </button>

          <span className="pagination-info">
            {page} / {pages}
          </span>

          <button
            className="pagination-btn"
            onClick={() => onPageChange?.(page + 1)}
            disabled={!has_next}
            title="Next page"
          >
            <ChevronRight size={14} />
          </button>
          <button
            className="pagination-btn"
            onClick={() => onPageChange?.(pages)}
            disabled={page === pages}
            title="Last page"
          >
            <ChevronsRight size={14} />
          </button>
        </div>
      )}
    </motion.div>
  )
}

function HistoryItem({ detection, isActive, onClick, index, getFullnessColor }) {
  const {
    timestamp,
    detections: objects = [],
    bin_detected,
    bin_count,
    overall_fullness,
    overall_fullness_percent,
    status_summary,
  } = detection

  const formattedTime = timestamp
    ? format(new Date(timestamp), 'HH:mm:ss')
    : '--:--:--'

  const relativeTime = timestamp
    ? formatDistanceToNow(new Date(timestamp), { addSuffix: true })
    : ''

  // Get unique labels
  const labels = [...new Set(objects.map(o => o.label))]

  // Check if label is bin-related
  const isBinLabel = (label) => {
    const binKeywords = ['bin', 'trash', 'waste', 'container', 'garbage', 'bag', 'bucket']
    return binKeywords.some(k => label.toLowerCase().includes(k))
  }

  return (
    <motion.div
      className={`history-item ${isActive ? 'active' : ''}`}
      onClick={onClick}
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 20 }}
      transition={{ delay: index * 0.03 }}
      whileHover={{ x: 4 }}
      layout
    >
      <div className="history-time">
        <Clock size={10} style={{ marginRight: 4 }} />
        {formattedTime}
        <span style={{ marginLeft: 8, opacity: 0.6 }}>
          {relativeTime}
        </span>
      </div>

      {/* Summary text */}
      {status_summary && (
        <div className="history-summary">
          {status_summary.length > 60
            ? status_summary.substring(0, 60) + '...'
            : status_summary
          }
        </div>
      )}

      {/* Labels */}
      {labels.length > 0 && (
        <div className="history-labels">
          {labels.slice(0, 4).map((label, idx) => (
            <span
              key={idx}
              className={`label-tag ${isBinLabel(label) ? 'bin' : 'default'}`}
            >
              {label}
            </span>
          ))}
          {labels.length > 4 && (
            <span className="label-tag default">+{labels.length - 4}</span>
          )}
        </div>
      )}

      {/* Bin fullness indicator */}
      {bin_detected && overall_fullness && (
        <div className="history-fullness">
          <Trash2
            size={12}
            style={{ color: getFullnessColor(overall_fullness, overall_fullness_percent) }}
          />
          <div className="history-fullness-bar">
            <motion.div
              className="history-fullness-fill"
              style={{
                width: `${overall_fullness_percent || 0}%`,
                background: getFullnessColor(overall_fullness, overall_fullness_percent),
              }}
              initial={{ width: 0 }}
              animate={{ width: `${overall_fullness_percent || 0}%` }}
              transition={{ delay: index * 0.05, duration: 0.4 }}
            />
          </div>
          <span className="history-fullness-text">
            {overall_fullness_percent || 0}%
          </span>
        </div>
      )}
    </motion.div>
  )
}

export default DetectionHistory
