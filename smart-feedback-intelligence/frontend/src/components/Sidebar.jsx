export default function Sidebar({ page, setPage }) {
  const nav = [
    { id: 'single', label: 'Single Analysis', icon: <SearchIcon /> },
    { id: 'batch',  label: 'Batch Analysis',  icon: <LayersIcon /> },
    { id: 'dashboard', label: 'Dashboard',    icon: <BarIcon /> },
  ]

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <div className="logo-mark">SF</div>
        <h1>Smart Feedback Intelligence</h1>
        <span>v1.0.0</span>
      </div>

      <nav>
        {nav.map(n => (
          <div
            key={n.id}
            className={`nav-item ${page === n.id ? 'active' : ''}`}
            onClick={() => setPage(n.id)}
          >
            {n.icon}
            {n.label}
          </div>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="status-dot">
          <span className="dot" />
          API connected
        </div>
      </div>
    </aside>
  )
}

function SearchIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
    </svg>
  )
}

function LayersIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polygon points="12 2 2 7 12 12 22 7 12 2"/>
      <polyline points="2 17 12 22 22 17"/>
      <polyline points="2 12 12 17 22 12"/>
    </svg>
  )
}

function BarIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <line x1="18" y1="20" x2="18" y2="10"/>
      <line x1="12" y1="20" x2="12" y2="4"/>
      <line x1="6"  y1="20" x2="6"  y2="14"/>
    </svg>
  )
}
