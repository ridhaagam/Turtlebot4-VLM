import axios from 'axios'

const API_BASE = '/api'

const api = axios.create({
  baseURL: API_BASE,
  timeout: 10000,
})

export async function fetchLatestDetection() {
  const response = await api.get('/detections/latest')
  return response.data
}

export async function fetchDetections({ page = 1, per_page = 20, label, device_id }) {
  const params = { page, per_page }
  if (label) params.label = label
  if (device_id) params.device_id = device_id

  const response = await api.get('/detections', { params })
  return response.data
}

export async function fetchDetection(id) {
  const response = await api.get(`/detections/${id}`)
  return response.data
}

export async function deleteDetection(id) {
  const response = await api.delete(`/detections/${id}`)
  return response.data
}

export async function fetchStats() {
  const response = await api.get('/detections/stats')
  return response.data
}

export async function exportDetections({ start_date, end_date, limit = 1000 }) {
  const params = { limit }
  if (start_date) params.start_date = start_date
  if (end_date) params.end_date = end_date

  const response = await api.get('/detections/export', { params })
  return response.data
}

export async function checkHealth() {
  const response = await api.get('/health')
  return response.data
}

export default api
