import { useState } from 'react'
import axios from 'axios'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const EXAMPLES = [
  "Delivery was extremely late and package arrived damaged.",
  "Absolutely love this product! Fast shipping, great quality!",
  "I was charged twice and support hasn't responded in 3 days.",
  "Product is okay, nothing special. Average experience.",
  "The return process is a nightmare. Terrible service.",
]

const EMOJI = {
  sentiment: { negative: '🔴', positive: '🟢', neutral: '🟡' },
  urgency:   { high: '🔴', medium: '🟡', low: '🟢' },
  churn:     { at_risk: '⚠️', no_risk: '✅' },
}

export default function SingleAnalysis() {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function analyze() {
    if (!text.trim()) return
    setLoading(true); setError(null); setResult(null)
    try {
      const res = await axios.post(`${API}/analyze`, {
        text, include_absa: true, include_entities: true
      })
      setResult(res.data)
    } catch (e) {
      setError(e.message || 'API error — is the server running?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <div className="page-header">
        <h2>Single Analysis</h2>
        <p>Analyze one customer feedback in real-time</p>
      </div>

      <div className="card">
        {/* Examples */}
        <div className="section-label">Try an example</div>
        <div className="examples">
          {EXAMPLES.map((ex, i) => (
            <span key={i} className="example-pill" onClick={() => setText(ex)}>
              {ex.slice(0, 38)}…
            </span>
          ))}
        </div>

        {/* Input */}
        <textarea
          className="input-area"
          placeholder="Paste customer feedback here..."
          value={text}
          onChange={e => setText(e.target.value)}
          rows={5}
        />

        <div style={{ display: 'flex', gap: 10, marginTop: 12 }}>
          <button className="btn btn-primary" onClick={analyze} disabled={loading || !text.trim()}>
            {loading ? 'Analyzing…' : '↗ Analyze'}
          </button>
          <button className="btn btn-ghost" onClick={() => { setText(''); setResult(null); setError(null) }}>
            Clear
          </button>
        </div>

        {loading && <div className="loading-bar"><div className="loading-bar-fill" /></div>}
        {error   && <div className="error-box">⚠ {error}</div>}
      </div>

      {result && <Results result={result} />}
    </div>
  )
}

function Results({ result }) {
  const metrics = [
    { label: 'Sentiment',  value: result.sentiment,  conf: result.sentiment_confidence },
    { label: 'Category',   value: result.category,   conf: result.category_confidence },
    { label: 'Urgency',    value: result.urgency,     conf: result.urgency_confidence },
    { label: 'Churn Risk', value: result.churn_risk,  conf: result.churn_confidence },
  ]

  return (
    <div style={{ marginTop: 20 }}>
      {/* Metric Cards */}
      <div className="metrics-grid">
        {metrics.map(m => (
          <div key={m.label} className="metric-card">
            <div className="metric-label">{m.label}</div>
            <div className={`metric-value ${m.value}`}>
              {EMOJI.sentiment[m.value] || EMOJI.urgency[m.value] || EMOJI.churn[m.value] || ''} {m.value}
            </div>
            {m.conf !== null && (
              <>
                <div className="metric-conf">{(m.conf * 100).toFixed(1)}% confidence</div>
                <div className="conf-bar-wrap">
                  <div className="conf-bar" style={{ width: `${m.conf * 100}%` }} />
                </div>
              </>
            )}
          </div>
        ))}
      </div>

      {/* ABSA */}
      {result.aspect_sentiments && Object.keys(result.aspect_sentiments).length > 0 && (
        <div className="card" style={{ marginTop: 16 }}>
          <div className="section-label">Aspect-Based Sentiment</div>
          <div className="absa-grid">
            {Object.entries(result.aspect_sentiments).map(([aspect, sentiment]) => (
              <span key={aspect} className={`absa-pill ${sentiment}`}>
                {aspect} · {sentiment}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Entities */}
      {result.entities && Object.keys(result.entities).length > 0 && (
        <div className="card" style={{ marginTop: 12 }}>
          <div className="section-label">Named Entities</div>
          <div className="absa-grid">
            {Object.entries(result.entities).map(([type, values]) =>
              values.map(v => (
                <span key={`${type}-${v}`} className="absa-pill neutral">
                  {type} · {v}
                </span>
              ))
            )}
          </div>
        </div>
      )}

      {/* Meta */}
      <div style={{ marginTop: 12, fontSize: 11, color: 'var(--text3)', fontFamily: 'DM Mono, monospace' }}>
        ⚡ Processed in {result.processing_time_ms?.toFixed(0)}ms
      </div>
    </div>
  )
}
