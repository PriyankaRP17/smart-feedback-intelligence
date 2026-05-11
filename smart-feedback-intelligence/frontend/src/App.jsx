import { useState } from 'react'
import Sidebar from './components/Sidebar'
import SingleAnalysis from './pages/SingleAnalysis'
import BatchAnalysis from './pages/BatchAnalysis'
import Dashboard from './pages/Dashboard'
import './App.css'

export default function App() {
  const [page, setPage] = useState('single')
  return (
    <div className="app">
      <Sidebar page={page} setPage={setPage} />
      <main className="main-content">
        {page === 'single'    && <SingleAnalysis />}
        {page === 'batch'     && <BatchAnalysis />}
        {page === 'dashboard' && <Dashboard />}
      </main>
    </div>
  )
}
