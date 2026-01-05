import React from 'react'
import { motion } from 'framer-motion'
import { Sun, Moon } from 'lucide-react'

function ThemeToggle({ theme, onToggle }) {
  const isDark = theme === 'dark'

  return (
    <motion.button
      className="theme-toggle"
      onClick={onToggle}
      title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
    >
      <motion.div
        className="theme-toggle-track"
        animate={{
          backgroundColor: isDark ? 'var(--bg-elevated)' : 'var(--accent-primary)',
        }}
      >
        <motion.div
          className="theme-toggle-thumb"
          animate={{
            x: isDark ? 0 : 22,
          }}
          transition={{ type: 'spring', stiffness: 500, damping: 30 }}
        >
          {isDark ? (
            <Moon size={12} className="theme-icon" />
          ) : (
            <Sun size={12} className="theme-icon" />
          )}
        </motion.div>
      </motion.div>
    </motion.button>
  )
}

export default ThemeToggle
