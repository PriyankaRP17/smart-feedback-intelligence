import { useState } from 'react'
import {
  PieChart, Pie, Cell, BarChart, Bar,
  XAxis, YAxis, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

// Demo data — replace with real API data in production
const SENTIMENT_DATA = [
  { name: 'Positive', value: 420, color: '#4dffb4' },
  { name: 'Negative', value: 300, color: '#ff4d6d' },
  { name: 'Neutral',  value: 180, color: '#ffd166' },
]

const CATEGORY_DATA = [
  { name: 'Product',  count: 280 },
  { name: 'Delivery', count: 220 },
  { name: 'Support',  count: 180 },
  { name: 'Billing',  count: 140 },
  { name: 'Returns',  count: 80  },
]

const URGENCY_DATA = [
  { name: 'Low',    value: 520, color: '#4dffb4' },
  { name: 'Medium', value: 280, color: '#ffd166' },
  { name: 'High',   value: 100, color: '#ff4d6d' },
]

const TOOLTIP_STYLE = {
  background: '#18181c',
  border: '1px solid #252528',
  borderRadius: 8,
  color: '#f0f0f2',
  fontFamily: 'DM Mono, monospace',
  fontSize: 12,
}

export default function Dashboard() {
  return (
    <div>
      <div className="page-header">
        <h2>Dashboard</h2>
        <p>Aggregate analytics across all feedback</p>
      </div>

      {/* KPIs */}
      <div className="kpi-grid">
        <div className="kpi-card">
          <div className="kpi-label">Total Reviews</div>
          <div className="kpi-value">900</div>
          <div className="kpi-sub">↑ 12% this week</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Positive Rate</div>
          <div className="kpi-value" style={{ color: 'var(--pos)' }}>46.7%</div>
          <div className="kpi-sub">↑ 3.2% vs last week</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">High Urgency</div>
          <div className="kpi-value" style={{ color: 'var(--neg)' }}>100</div>
          <div className="kpi-sub">↓ 8 fewer than last week</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Churn Risk</div>
          <div className="kpi-value" style={{ color: 'var(--neg)' }}>28%</div>
          <div className="kpi-sub">↓ 3% improvement</div>
        </div>
      </div>

      {/* Charts */}
      <div className="chart-grid">
        <div className="chart-card">
          <div className="chart-title">Sentiment Distribution</div>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={SENTIMENT_DATA} cx="50%" cy="50%" innerRadius={60} outerRadius={90}
                dataKey="value" paddingAngle={3}>
                {SENTIMENT_DATA.map((d, i) => <Cell key={i} fill={d.color} />)}
              </Pie>
              <Tooltip contentStyle={TOOLTIP_STYLE} />
              <Legend
                formatter={v => <span style={{ color: '#8a8a96', fontSize: 12 }}>{v}</span>}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card">
          <div className="chart-title">Issues by Category</div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={CATEGORY_DATA} barSize={28}>
              <XAxis dataKey="name" tick={{ fill: '#8a8a96', fontSize: 11, fontFamily: 'DM Mono, monospace' }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: '#8a8a96', fontSize: 11, fontFamily: 'DM Mono, monospace' }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
              <Bar dataKey="count" fill="#e8ff47" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card">
          <div className="chart-title">Urgency Distribution</div>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={URGENCY_DATA} cx="50%" cy="50%" innerRadius={60} outerRadius={90}
                dataKey="value" paddingAngle={3}>
                {URGENCY_DATA.map((d, i) => <Cell key={i} fill={d.color} />)}
              </Pie>
              <Tooltip contentStyle={TOOLTIP_STYLE} />
              <Legend formatter={v => <span style={{ color: '#8a8a96', fontSize: 12 }}>{v}</span>} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card">
          <div className="chart-title">Top ABSA Aspects</div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart layout="vertical" barSize={18}
              data={[
                { aspect: 'Product',  pos: 160, neg: 115 },
                { aspect: 'Returns',  pos: 207, neg: 102 },
                { aspect: 'Delivery', pos: 51,  neg: 52  },
                { aspect: 'Billing',  pos: 82,  neg: 7   },
                { aspect: 'Support',  pos: 74,  neg: 20  },
              ]}>
              <XAxis type="number" tick={{ fill: '#8a8a96', fontSize: 11, fontFamily: 'DM Mono, monospace' }} axisLine={false} tickLine={false} />
              <YAxis type="category" dataKey="aspect" tick={{ fill: '#8a8a96', fontSize: 11, fontFamily: 'DM Mono, monospace' }} axisLine={false} tickLine={false} width={60} />
              <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
              <Bar dataKey="pos" fill="#4dffb4" radius={[0, 4, 4, 0]} name="Positive" />
              <Bar dataKey="neg" fill="#ff4d6d" radius={[0, 4, 4, 0]} name="Negative" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}
