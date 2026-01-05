import React from 'react'
import { motion } from 'framer-motion'
import { useQuery } from '@tanstack/react-query'
import {
  Navigation,
  ArrowUp,
  ArrowLeft,
  ArrowRight,
  CheckCircle,
  XCircle,
  Target,
  Clock,
  Activity,
  Box,
  Cpu,
} from 'lucide-react'

// Fetch navigation stats
async function fetchNavigationStats() {
  const response = await fetch('/api/vision/navigation/stats')
  if (!response.ok) return null
  return response.json()
}

// Fetch latest navigation result
async function fetchNavigationLatest() {
  const response = await fetch('/api/vision/navigation/latest')
  if (!response.ok) return null
  return response.json()
}

// Navigation command icons
const COMMAND_ICONS = {
  forward: ArrowUp,
  left: ArrowLeft,
  right: ArrowRight,
  arrived: CheckCircle,
  not_found: XCircle,
}

// Navigation command colors
const COMMAND_COLORS = {
  forward: '#22c55e',
  left: '#3b82f6',
  right: '#f59e0b',
  arrived: '#8b5cf6',
  not_found: '#6b7280',
}

function NavigationSummary() {
  // Fetch navigation stats
  const { data: stats } = useQuery({
    queryKey: ['navigationStats'],
    queryFn: fetchNavigationStats,
    refetchInterval: 5000,
    retry: false,
  })

  // Fetch latest navigation
  const { data: latest } = useQuery({
    queryKey: ['navigationLatest'],
    queryFn: fetchNavigationLatest,
    refetchInterval: 2000,
    retry: false,
  })

  const navCommand = latest?.command || 'not_found'
  const CommandIcon = COMMAND_ICONS[navCommand] || XCircle
  const commandColor = COMMAND_COLORS[navCommand] || '#6b7280'

  // Generate situation text
  const generateSituationText = () => {
    if (!latest) {
      return 'Waiting for navigation inference data from vision model...'
    }

    let text = ''

    if (latest.bin_detected) {
      text += `Bin detected at ${latest.bin_position || 'unknown'} position`
      if (latest.bin_size) {
        text += ` (${latest.bin_size} size)`
      }
      text += '. '

      if (navCommand === 'forward') {
        text += 'Moving forward toward the bin.'
      } else if (navCommand === 'left') {
        text += 'Turning left to align with bin.'
      } else if (navCommand === 'right') {
        text += 'Turning right to align with bin.'
      } else if (navCommand === 'arrived') {
        text += 'Arrived at bin location!'
      }
    } else {
      text += 'No bin detected in current frame. Searching...'
    }

    return text
  }

  return (
    <motion.div
      className="navigation-summary card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: 0.15 }}
    >
      <div className="card-header">
        <div className="header-title">
          <Navigation size={18} />
          <h3>Navigation Status</h3>
        </div>
        <div
          className="nav-command-badge"
          style={{ '--cmd-color': commandColor }}
        >
          <CommandIcon size={14} />
          <span>{navCommand.toUpperCase()}</span>
        </div>
      </div>

      {/* Current command display */}
      <div className="nav-command-display" style={{ '--cmd-color': commandColor }}>
        <motion.div
          className="command-icon-wrapper"
          key={navCommand}
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ type: 'spring', stiffness: 300, damping: 20 }}
        >
          <CommandIcon size={48} />
        </motion.div>
        <div className="command-text">
          <span className="command-label">Current Command</span>
          <span className="command-value">{navCommand.toUpperCase()}</span>
        </div>
      </div>

      {/* Situation summary */}
      <div className="situation-text">
        {generateSituationText()}
      </div>

      {/* Detection info */}
      {latest && (
        <div className="nav-info-grid">
          <div className="nav-info-item">
            <Target size={14} />
            <span className="info-label">Bin</span>
            <span
              className="info-value"
              style={{ color: latest.bin_detected ? '#22c55e' : '#6b7280' }}
            >
              {latest.bin_detected ? 'DETECTED' : 'NOT FOUND'}
            </span>
          </div>

          {latest.bin_position && (
            <div className="nav-info-item">
              <Box size={14} />
              <span className="info-label">Position</span>
              <span className="info-value">{latest.bin_position.toUpperCase()}</span>
            </div>
          )}

          {latest.bin_size && (
            <div className="nav-info-item">
              <Box size={14} />
              <span className="info-label">Size</span>
              <span className="info-value">{latest.bin_size.toUpperCase()}</span>
            </div>
          )}

          <div className="nav-info-item">
            <Activity size={14} />
            <span className="info-label">Confidence</span>
            <span className="info-value">
              {latest.confidence ? `${(latest.confidence * 100).toFixed(0)}%` : '--'}
            </span>
          </div>

          <div className="nav-info-item">
            <Clock size={14} />
            <span className="info-label">Inference</span>
            <span className="info-value">
              {latest.inference_time_ms ? `${latest.inference_time_ms.toFixed(0)}ms` : '--'}
            </span>
          </div>

          {latest.detection_count > 0 && (
            <div className="nav-info-item">
              <Target size={14} />
              <span className="info-label">Objects</span>
              <span className="info-value">{latest.detection_count}</span>
            </div>
          )}
        </div>
      )}

      {/* Stats summary */}
      {stats && (
        <div className="nav-stats-summary">
          <div className="stat-item">
            <span className="stat-label">Total Inferences</span>
            <span className="stat-value">{stats.total_inferences || 0}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Bins Detected</span>
            <span className="stat-value" style={{ color: '#22c55e' }}>
              {stats.bins_detected || 0}
            </span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Avg Time</span>
            <span className="stat-value">
              {stats.avg_inference_time_ms ? `${stats.avg_inference_time_ms.toFixed(0)}ms` : '--'}
            </span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Avg Confidence</span>
            <span className="stat-value">
              {stats.avg_confidence ? `${(stats.avg_confidence * 100).toFixed(0)}%` : '--'}
            </span>
          </div>
        </div>
      )}

      {/* Command distribution */}
      {stats?.commands && Object.keys(stats.commands).length > 0 && (
        <div className="command-distribution">
          <span className="dist-label">Command Distribution</span>
          <div className="dist-bars">
            {Object.entries(stats.commands).map(([cmd, count]) => {
              const total = stats.total_inferences || 1
              const percent = (count / total) * 100
              const CmdIcon = COMMAND_ICONS[cmd] || XCircle
              const color = COMMAND_COLORS[cmd] || '#6b7280'
              return (
                <div key={cmd} className="dist-bar-item">
                  <div className="dist-bar-label">
                    <CmdIcon size={12} style={{ color }} />
                    <span>{cmd}</span>
                  </div>
                  <div className="dist-bar-track">
                    <motion.div
                      className="dist-bar-fill"
                      style={{ background: color }}
                      initial={{ width: 0 }}
                      animate={{ width: `${percent}%` }}
                      transition={{ duration: 0.5, ease: 'easeOut' }}
                    />
                  </div>
                  <span className="dist-bar-value">{count}</span>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Model info */}
      {latest?.model_used && (
        <div className="model-info">
          <Cpu size={12} />
          <span>Model: {latest.model_used}</span>
        </div>
      )}
    </motion.div>
  )
}

export default NavigationSummary
