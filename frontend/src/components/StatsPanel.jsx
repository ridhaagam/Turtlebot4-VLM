import React from 'react'
import { motion } from 'framer-motion'
import { BarChart3, Camera, Box, Trash2 } from 'lucide-react'

function StatsPanel({ stats }) {
  const {
    total_detections = 0,
    total_objects = 0,
    bins_detected = 0,
    average_fullness_percent = 0,
    fullness_distribution = [],
    label_counts = [],
  } = stats || {}

  // Get color for fullness level
  const getFullnessColor = (level) => {
    if (level.includes('0-25')) return 'var(--fullness-empty)'
    if (level.includes('25-75')) return 'var(--fullness-partial)'
    if (level.includes('75-90')) return 'var(--fullness-mostly)'
    if (level.includes('90-100')) return 'var(--fullness-full)'
    return 'var(--text-tertiary)'
  }

  // Calculate max for distribution bars
  const maxDistribution = Math.max(...fullness_distribution.map(d => d.count), 1)

  return (
    <motion.div
      className="stats-panel"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.1, duration: 0.4 }}
    >
      <h2>
        <BarChart3 size={16} style={{ marginRight: 8 }} />
        Statistics
      </h2>

      <div className="stats-grid">
        <motion.div
          className="stat-item"
          whileHover={{ scale: 1.02 }}
          transition={{ type: 'spring', stiffness: 400 }}
        >
          <motion.div
            className="stat-value"
            key={total_detections}
            initial={{ scale: 0.8 }}
            animate={{ scale: 1 }}
          >
            {total_detections.toLocaleString()}
          </motion.div>
          <div className="stat-label">
            <Camera size={10} style={{ marginRight: 4 }} />
            Frames
          </div>
        </motion.div>

        <motion.div
          className="stat-item"
          whileHover={{ scale: 1.02 }}
          transition={{ type: 'spring', stiffness: 400 }}
        >
          <motion.div
            className="stat-value"
            key={total_objects}
            initial={{ scale: 0.8 }}
            animate={{ scale: 1 }}
          >
            {total_objects.toLocaleString()}
          </motion.div>
          <div className="stat-label">
            <Box size={10} style={{ marginRight: 4 }} />
            Objects
          </div>
        </motion.div>

        <motion.div
          className="stat-item"
          whileHover={{ scale: 1.02 }}
          transition={{ type: 'spring', stiffness: 400 }}
        >
          <motion.div
            className="stat-value"
            key={bins_detected}
            initial={{ scale: 0.8 }}
            animate={{ scale: 1 }}
            style={{ color: 'var(--accent-primary)' }}
          >
            {bins_detected.toLocaleString()}
          </motion.div>
          <div className="stat-label">
            <Trash2 size={10} style={{ marginRight: 4 }} />
            Bins Found
          </div>
        </motion.div>

        <motion.div
          className="stat-item"
          whileHover={{ scale: 1.02 }}
          transition={{ type: 'spring', stiffness: 400 }}
        >
          <motion.div
            className="stat-value"
            key={average_fullness_percent}
            initial={{ scale: 0.8 }}
            animate={{ scale: 1 }}
            style={{
              color: (average_fullness_percent || 0) >= 75
                ? 'var(--accent-warning)'
                : 'var(--accent-primary)'
            }}
          >
            {typeof average_fullness_percent === 'number' ? average_fullness_percent.toFixed(0) : '0'}%
          </motion.div>
          <div className="stat-label">Avg Fullness</div>
        </motion.div>
      </div>

      {/* Fullness distribution */}
      {fullness_distribution.length > 0 && (
        <div className="fullness-stats">
          <h3>Fullness Distribution</h3>
          <div className="fullness-distribution">
            {fullness_distribution.map(({ level, count }, idx) => (
              <motion.div
                key={level}
                className="distribution-row"
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.1 }}
              >
                <span className="distribution-label">{level}</span>
                <div className="distribution-bar">
                  <motion.div
                    className="distribution-bar-fill"
                    style={{ background: getFullnessColor(level) }}
                    initial={{ width: 0 }}
                    animate={{ width: `${(count / maxDistribution) * 100}%` }}
                    transition={{ delay: idx * 0.1 + 0.2, duration: 0.5 }}
                  />
                </div>
                <span className="distribution-count">{count}</span>
              </motion.div>
            ))}
          </div>
        </div>
      )}

      {/* Label counts */}
      {label_counts.length > 0 && (
        <div className="label-stats">
          <h3>Top Detected Labels</h3>
          <div className="label-list">
            {label_counts.slice(0, 5).map(({ label, count }, idx) => (
              <motion.div
                key={label}
                className="label-item"
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.05 }}
                whileHover={{ x: 4 }}
              >
                <span className="label-name">{label}</span>
                <span className="label-count">{count.toLocaleString()}</span>
              </motion.div>
            ))}
          </div>
        </div>
      )}
    </motion.div>
  )
}

export default StatsPanel
