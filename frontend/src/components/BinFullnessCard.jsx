import React from 'react'
import { motion } from 'framer-motion'
import { Trash2, AlertTriangle, CheckCircle, AlertCircle } from 'lucide-react'

function BinFullnessCard({ detection }) {
  const {
    bin_detected = false,
    bin_count = 0,
    overall_fullness,
    overall_fullness_percent,
    detections = [],
  } = detection || {}

  // Get fullness level info
  const getFullnessInfo = (fullness, percent) => {
    if (!fullness && percent === undefined) {
      return {
        level: 'unknown',
        label: 'No Data',
        color: 'var(--text-tertiary)',
        bgClass: '',
        icon: AlertCircle,
        message: 'No bin detected in current frame',
      }
    }

    const pct = percent || 0
    if (fullness?.includes('0-25') || pct < 25) {
      return {
        level: 'empty',
        label: '0-25%',
        color: 'var(--fullness-empty)',
        bgClass: 'empty',
        icon: CheckCircle,
        message: 'Bin is mostly empty - good capacity available',
      }
    }
    if (fullness?.includes('25-75') || pct < 75) {
      return {
        level: 'partial',
        label: '25-75%',
        color: 'var(--fullness-partial)',
        bgClass: 'partial',
        icon: Trash2,
        message: 'Bin is partially full - moderate capacity remaining',
      }
    }
    if (fullness?.includes('75-90') || pct < 90) {
      return {
        level: 'mostly-full',
        label: '75-90%',
        color: 'var(--fullness-mostly)',
        bgClass: 'mostly-full',
        icon: AlertTriangle,
        message: 'Bin is almost full - consider emptying soon',
      }
    }
    return {
      level: 'full',
      label: '90-100%',
      color: 'var(--fullness-full)',
      bgClass: 'full',
      icon: AlertTriangle,
      message: 'Bin is full - needs immediate attention!',
    }
  }

  const fullnessInfo = getFullnessInfo(overall_fullness, overall_fullness_percent)
  const Icon = fullnessInfo.icon
  const displayPercent = overall_fullness_percent ?? 0

  // Get bins from detections
  const binObjects = detections.filter(d =>
    d.label?.toLowerCase().includes('bin') ||
    d.label?.toLowerCase().includes('trash') ||
    d.label?.toLowerCase().includes('waste') ||
    d.label?.toLowerCase().includes('container')
  )

  return (
    <motion.div
      className="fullness-card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.1, duration: 0.4 }}
    >
      <div className="fullness-header">
        <h3>Bin Fullness Classification</h3>
        {bin_detected && (
          <motion.span
            className={`fullness-badge ${fullnessInfo.bgClass}`}
            initial={{ scale: 0.8 }}
            animate={{ scale: 1 }}
            transition={{ type: 'spring', stiffness: 300 }}
          >
            {fullnessInfo.label}
          </motion.span>
        )}
      </div>

      {/* Large percentage display */}
      <motion.div
        className="fullness-percent"
        style={{ color: fullnessInfo.color }}
        key={displayPercent}
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ type: 'spring', stiffness: 200 }}
      >
        {bin_detected ? (
          <>
            {displayPercent}
            <span>%</span>
          </>
        ) : (
          <span style={{ fontSize: '1.5rem', color: 'var(--text-tertiary)' }}>
            --
          </span>
        )}
      </motion.div>

      {/* Fullness gauge bar */}
      <div className="fullness-gauge">
        <motion.div
          className={`fullness-gauge-fill ${fullnessInfo.bgClass}`}
          initial={{ width: 0 }}
          animate={{ width: `${displayPercent}%` }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
        />
      </div>

      {/* Status message */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 'var(--space-sm)',
        padding: 'var(--space-sm) var(--space-md)',
        background: 'var(--bg-elevated)',
        borderRadius: 'var(--radius-md)',
        marginTop: 'var(--space-sm)',
      }}>
        <Icon
          size={18}
          style={{ color: fullnessInfo.color, flexShrink: 0 }}
        />
        <span style={{
          fontSize: '0.85rem',
          color: 'var(--text-secondary)',
        }}>
          {fullnessInfo.message}
        </span>
      </div>

      {/* Individual bin indicators */}
      {binObjects.length > 0 && (
        <div className="fullness-bins" style={{ marginTop: 'var(--space-md)' }}>
          {binObjects.map((bin, idx) => {
            const binInfo = getFullnessInfo(bin.bin_fullness, bin.bin_fullness_percent)
            return (
              <motion.div
                key={idx}
                className="bin-indicator"
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.1 }}
              >
                <span
                  className="bin-indicator-dot"
                  style={{ background: binInfo.color }}
                />
                <span>Bin {idx + 1}</span>
                <span style={{ color: binInfo.color, fontWeight: 600 }}>
                  {bin.bin_fullness_percent ?? '?'}%
                </span>
              </motion.div>
            )
          })}
        </div>
      )}
    </motion.div>
  )
}

export default BinFullnessCard
