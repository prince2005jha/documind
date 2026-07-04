import axios from 'axios'

const BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
const client = axios.create({ baseURL: BASE })

export async function listDocuments() {
  const resp = await client.get('/documents')
  return resp.data
}

export async function uploadFiles(formData) {
  const resp = await client.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return resp.data
}

export async function ingest() {
  const resp = await client.post('/ingest')
  return resp.data
}

export async function ask(question, history = []) {
  const resp = await client.post('/ask', { question, history })
  return resp.data
}
