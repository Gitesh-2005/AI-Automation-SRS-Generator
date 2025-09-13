import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { motion } from 'framer-motion'

// Pages
import HomePage from './pages/HomePage'
import UploadPage from './pages/UploadPage'
import GeneratePage from './pages/GeneratePage'
import EditorPage from './pages/EditorPage'

// Components
import Navbar from './components/Navbar'

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gradient-to-br from-secondary-50 via-white to-primary-50">
        <Navbar />
        
        <motion.main 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.3 }}
          className="relative"
        >
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/generate" element={<GeneratePage />} />
            <Route path="/editor/:documentId?" element={<EditorPage />} />
          </Routes>
        </motion.main>
      </div>
    </Router>
  )
}

export default App
