import { useState } from 'react'
import LandingPage from './pages/landing'
import LiveDetection from './pages/LiveDetection'

function App() {
  const [currentView, setCurrentView] = useState('landing');

  return (
    <>
      {currentView === 'landing' ? (
        <LandingPage onStartLive={() => setCurrentView('live')} />
      ) : (
        <LiveDetection onBack={() => setCurrentView('landing')} />
      )}
    </>
  )
}

export default App
