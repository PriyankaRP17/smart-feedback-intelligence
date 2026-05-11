import { useState } from 'react'
import axios from 'axios'

const API = 'http://localhost:8000'

const SENTIMENT_COLOR = { negative: 'var(--neg)', positive: 'var(--pos)', neutral: 'var(--neu)' }
const URGENCY_COLOR   = { high: 'var(--neg)', medium: 'var(--neu)', low: 'var(--pos)' }

export default function BatchAnalysis() {
  const [texts, setTexts]   = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]   = useState(null)

  async function analyze() {
    const lines = texts.split('\n').map(t => t.trim()).filter(Boolean)
    if (!lines.length) return
    setLoading(true); setError(null); setResult(null)
    try {
      const res = await axios.post(`${API}/batch`, { texts: lines, include_absa: false })
      setResult(res.data)
    } catch (e) {
      setError(e.message || 'API error')
    } finally {
      setLoading(false)
    }
  }

  function downloadCSV() {
    if (!result) return
    const headers = 'text,sentiment,category,urgency,churn_risk'
    const rows = result.results.map(r =>
      `"${r.text.replace(/"/g,'""')}",${r.sentiment},${r.category},${r.urgency},${r.churn_risk}`
    )
    const blob = new Blob([[headers, ...rows].join('\n')], { type: 'text/csv' })
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob)
    a.download = 'feedback_analysis.csv'; a.click()
  }

  return (
    <div>
      <div className="page-header">
        <h2>Batch Analysis</h2>
        <p>Analyze up to 500 reviews at once — one per line</p>
      </div>

      <div className="card">
        <div className="section-label">Paste reviews — one per line</div>
        <textarea
          className="input-area"
          style={{ minHeight: 180 }}
          placeholder={`Great product, fast delivery!\nTerrible customer support, never resolved my issue.\nPackage arrived late but quality is good.`}
          value={texts}
          onChange={e => setTexts(e.target.value)}
        />
        <div style={{ display: 'flex', gap: 10, marginTop: 12, alignItems: 'center' }}>
          <button className="btn btn-primary" onClick={analyze} disabled={loading || !texts.trim()}>
            {loading ? 'Analyzing…' : `↗ Analyze ${texts.split('\n').filter(Boolean).length || 0} reviews`}
          </button>
          <button className="btn btn-ghost" onClick={() => { setTexts(''); setResult(null) }}>Clear</button>
          {result && (
            <button className="btn btn-ghost" onClick={downloadCSV} style={{ marginLeft: 'auto' }}>
              ⬇ Download CSV
            </button>
          )}
        </div>
        {loading && <div className="loading-bar"><div className="loading-bar-fill" /></div>}
        {error   && <div className="error-box">⚠ {error}</div>}
      </div>

      {result && (
        <>
          {/* Summary Cards */}
          <div className="metrics-grid" style={{ marginTop: 20 }}>
            {Object.entries(result.summary.sentiment_distribution).map(([label, count]) => (
              <div key={label} className="metric-card">
                <div className="metric-label">Sentiment · {label}</div>
                <div className="metric-value" style={{ color: SENTIMENT_COLOR[label] || 'var(--text)' }}>
                  {count}
                </div>
                <div className="metric-conf">{((count / result.total) * 100).toFixed(1)}% of total</div>
              </div>
            ))}
            <div className="metric-card">
              <div className="metric-label">High Urgency</div>
              <div className="metric-value" style={{ color: 'var(--neg)' }}>
                {result.summary.high_urgency_count}
              </div>
              <div className="metric-conf">need immediate action</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Churn Risk</div>
              <div className="metric-value" style={{ color: 'var(--neg)' }}>
                {result.summary.churn_rate_pct}%
              </div>
              <div className="metric-conf">at risk of leaving</div>
            </div>
          </div>

          {/* Results Table */}
          <div className="card" style={{ marginTop: 16 }}>
            <div className="section-label">Results — {result.total} reviews</div>
            <table className="results-table">
              <thead>
                <tr>
                  <th>Review</th>
                  <th>Sentiment</th>
                  <th>Category</th>
                  <th>Urgency</th>
                  <th>Churn</th>
                </tr>
              </thead>
              <tbody>
                {result.results.map((r, i) => (
                  <tr key={i}>
                    <td style={{ maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {r.text}
                    </td>
                    <td style={{ color: SENTIMENT_COLOR[r.sentiment] }}>{r.sentiment}</td>
                    <td>{r.category}</td>
                    <td style={{ color: URGENCY_COLOR[r.urgency] }}>{r.urgency}</td>
                    <td style={{ color: r.churn_risk === 'at_risk' ? 'var(--neg)' : 'var(--pos)' }}>
                      {r.churn_risk}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}
