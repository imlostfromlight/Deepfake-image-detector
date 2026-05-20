import { useState } from 'react'
import LandingPage from './pages/landing'
import LiveDetection from './pages/LiveDetection'
import VideoCallDetection from './pages/VideoCallDetection'
import DocumentDetection from './pages/DocumentDetection'
import VideoLinkDetection from './pages/VideoLinkDetection'

function App() {
  const [currentView, setCurrentView] = useState('landing');

  const goHome = () => setCurrentView('landing');

  return (
    <>
      {currentView === 'landing' && (
        <LandingPage
          onStartLive={() => setCurrentView('live')}
          onStartVideoCall={() => setCurrentView('videocall')}
          onStartDocument={() => setCurrentView('document')}
          onStartVideoLink={() => setCurrentView('videolink')}
        />
      )}
      {currentView === 'live'      && <LiveDetection      onBack={goHome} />}
      {currentView === 'videocall' && <VideoCallDetection onBack={goHome} />}
      {currentView === 'document'  && <DocumentDetection  onBack={goHome} />}
      {currentView === 'videolink' && <VideoLinkDetection onBack={goHome} />}
    </>
  )
}

export default App
