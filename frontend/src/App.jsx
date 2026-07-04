import React, { useEffect, useState } from 'react'
import { listDocuments, uploadFiles, ingest, ask } from './api'

function Sidebar({ files, indexReady, isIndexing, onUpload, onIngest, onRefresh }) {
  return (
    <div className="sidebar">
      <h3>DocuMind</h3>
      <p className="subtitle">Ask questions from your uploaded documents.</p>

      <div className="upload-card">
        <label className="label">Upload files</label>
        <input type="file" multiple onChange={onUpload} />
      </div>

      <div className="controls">
        <button onClick={onIngest} disabled={isIndexing}>
          {isIndexing ? 'Building...' : 'Build Index'}
        </button>
        <button onClick={onRefresh} className="secondary">Refresh</button>
      </div>

      <div className="doc-list">
        <strong>Indexed files</strong>
        {files.length === 0 ? (
          <p className="muted">No documents uploaded yet.</p>
        ) : (
          <ul>
            {files.map((f) => <li key={f}>{f}</li>)}
          </ul>
        )}
      </div>

      <div className={`status-box ${indexReady ? 'ready' : 'pending'}`}>
        {indexReady ? 'Index ready' : 'Index not built yet'}
      </div>
    </div>
  )
}

function ChatWindow({ indexReady, statusMessage, isIndexing }) {
  const [question, setQuestion] = useState('')
  const [messages, setMessages] = useState(() => {
    if (typeof window === 'undefined') return []
    try {
      return JSON.parse(localStorage.getItem('documind-chat-history') || '[]')
    } catch {
      return []
    }
  })

  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('documind-chat-history', JSON.stringify(messages))
    }
  }, [messages])

  const suggestedPrompts = [
    'Summarize the uploaded documents',
    'What are the key points?',
    'Explain the main concepts in simple terms',
  ]

  async function handleAsk() {
    if (!question.trim() || !indexReady) return

    const q = question.trim()
    const history = messages.slice(-6).map((msg) => ({
      role: msg.role === 'user' ? 'user' : 'assistant',
      content: msg.text,
    }))

    setMessages((m) => [...m, { role: 'user', text: q }])
    setQuestion('')

    try {
      const res = await ask(q, history)
      setMessages((m) => [...m, { role: 'assistant', text: res.answer, sources: res.sources }])
    } catch (error) {
      setMessages((m) => [...m, { role: 'assistant', text: 'The request failed. Please try again.' }])
    }
  }

  function handleClearChat() {
    setMessages([])
  }

  async function handleCopy(text) {
    try {
      await navigator.clipboard.writeText(text)
    } catch {
      // Ignore clipboard errors and keep the UI responsive
    }
  }

  return (
    <div className="main">
      <div className="chat-card">
        <div className="chat-header">
          <h2>Ask your documents</h2>
          <div className="history-actions">
            <span className="turn-count">{messages.length} turns</span>
            <button className="secondary small" onClick={handleClearChat} disabled={messages.length === 0}>
              Clear chat
            </button>
            <span className={`status-pill ${indexReady ? 'ready' : 'pending'}`}>
              {indexReady ? 'Ready' : 'Needs indexing'}
            </span>
          </div>
        </div>

        {!indexReady && (
          <div className="status-banner">{statusMessage}</div>
        )}

        <div className="messages">
          {messages.length === 0 ? (
            <div className="empty-state">
              {indexReady
                ? 'Ask a question about your uploaded documents.'
                : 'Upload documents and click Build Index to start asking questions.'}
            </div>
          ) : (
            messages.map((m, i) => (
              <div className={`message ${m.role === 'user' ? 'user-message' : 'assistant-message'}`} key={i}>
                <div className="message-topline">
                  <strong>{m.role === 'user' ? 'You' : 'Assistant'}:</strong>
                  {m.role === 'assistant' && (
                    <button className="secondary small copy-btn" onClick={() => handleCopy(m.text)}>
                      Copy
                    </button>
                  )}
                </div>
                <div className="message-text">{m.text}</div>
                {m.sources && m.sources.length > 0 && (
                  <div className="source-row">
                    {m.sources.map((src) => (
                      <span className="source-chip" key={src}>{src}</span>
                    ))}
                  </div>
                )}
              </div>
            ))
          )}
        </div>

        <div className="suggestions">
          {suggestedPrompts.map((prompt) => (
            <button key={prompt} className="secondary small" onClick={() => setQuestion(prompt)}>
              {prompt}
            </button>
          ))}
        </div>

        <div className="composer">
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault()
                handleAsk()
              }
            }}
            placeholder="Ask a question..."
            disabled={!indexReady || isIndexing}
          />
          <button onClick={handleAsk} disabled={!indexReady || !question.trim() || isIndexing}>
            Ask
          </button>
        </div>
      </div>
    </div>
  )
}

export default function App() {
  const [files, setFiles] = useState([])
  const [indexReady, setIndexReady] = useState(false)
  const [isIndexing, setIsIndexing] = useState(false)
  const [statusMessage, setStatusMessage] = useState('Upload documents and click Build Index to enable answers.')

  async function refresh() {
    try {
      const data = await listDocuments()
      setFiles(data.files || [])
      setIndexReady(Boolean(data.index_ready))
      setStatusMessage(
        data.index_ready
          ? 'Index ready. Ask questions about your documents.'
          : 'No index built yet. Upload documents and click Build Index.'
      )
    } catch (error) {
      setStatusMessage('The backend could not be reached. Please confirm the API is running.')
    }
  }

  useEffect(() => {
    refresh()
  }, [])

  async function handleUpload(e) {
    const selected = e.target.files
    if (!selected || selected.length === 0) return

    const formData = new FormData()
    for (const file of selected) formData.append('files', file)

    try {
      await uploadFiles(formData)
      setStatusMessage('Upload complete. Build the index to enable questions.')
      await refresh()
    } catch (error) {
      setStatusMessage('Upload failed. Please try again.')
    }
  }

  async function handleIngest() {
    setIsIndexing(true)
    try {
      const result = await ingest()
      if (result?.status === 'success') {
        setStatusMessage(`Index built successfully with ${result.chunks_created} chunks.`)
      } else {
        setStatusMessage('Indexing finished.')
      }
      await refresh()
    } catch (error) {
      setStatusMessage('Index build failed. Please try again.')
    } finally {
      setIsIndexing(false)
    }
  }

  return (
    <div className="app">
      <Sidebar
        files={files}
        indexReady={indexReady}
        isIndexing={isIndexing}
        onUpload={handleUpload}
        onIngest={handleIngest}
        onRefresh={refresh}
      />
      <ChatWindow indexReady={indexReady} statusMessage={statusMessage} isIndexing={isIndexing} />
    </div>
  )
}
