import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import { 
  FileText, 
  Upload, 
  Layers, 
  Sparkles,
  Github,
  Heart 
} from 'lucide-react'

const Navbar: React.FC = () => {
  const location = useLocation()

  const navItems = [
    { path: '/', label: 'Home', icon: FileText },
    { path: '/upload', label: 'Upload', icon: Upload },
    { path: '/workspace', label: 'Workspace', icon: Layers },
  ]

  const isActive = (path: string) => {
    if (path === '/') {
      return location.pathname === '/'
    }
    return location.pathname.startsWith(path)
  }

  return (
    <motion.nav 
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.5 }}
      className="sticky top-0 z-50 w-full border-b border-secondary-200/50 bg-white/80 backdrop-blur-lg"
    >
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          {/* Logo */}
          <Link to="/" className="flex items-center space-x-2">
            <motion.div 
              whileHover={{ scale: 1.05, rotate: 5 }}
              className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-primary-500 to-primary-600 text-white"
            >
              <Sparkles className="h-5 w-5" />
            </motion.div>
            <div className="hidden sm:block">
              <h1 className="text-lg font-bold text-secondary-900">
                AI SRS Generator
              </h1>
              <p className="text-xs text-secondary-500">
                Intelligent Requirements
              </p>
            </div>
          </Link>

          {/* Navigation Links */}
          <div className="flex items-center space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon
              const active = isActive(item.path)
              
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className="relative"
                >
                  <motion.div
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className={`
                      flex items-center space-x-2 rounded-lg px-3 py-2 text-sm font-medium transition-colors
                      ${active 
                        ? 'bg-primary-50 text-primary-700' 
                        : 'text-secondary-600 hover:bg-secondary-100 hover:text-secondary-900'
                      }
                    `}
                  >
                    <Icon className="h-4 w-4" />
                    <span className="hidden sm:block">{item.label}</span>
                  </motion.div>
                  
                  {active && (
                    <motion.div
                      layoutId="navbar-active"
                      className="absolute -bottom-px left-0 right-0 h-0.5 bg-primary-600"
                      initial={false}
                      transition={{ type: "spring", stiffness: 500, damping: 30 }}
                    />
                  )}
                </Link>
              )
            })}
          </div>

          {/* Right side */}
          <div className="flex items-center space-x-3">
            {/* GitHub Link */}
            <motion.a
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              href="https://github.com/Gitesh-2005/AI-Automation-SRS-Generator.git"
              target="_blank"
              rel="noopener noreferrer"
              className="flex h-8 w-8 items-center justify-center rounded-lg text-secondary-600 transition-colors hover:bg-secondary-100 hover:text-secondary-900"
            >
              <Github className="h-4 w-4" />
            </motion.a>
          </div>
        </div>
      </div>
    </motion.nav>
  )
}

export default Navbar
