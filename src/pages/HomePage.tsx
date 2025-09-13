import React from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { 
  FileText, 
  Upload, 
  Bot, 
  Edit3, 
  Sparkles,
  Zap,
  Users,
  Download,
  ArrowRight,
  CheckCircle2,
  Brain,
  FileImage
} from 'lucide-react'

const HomePage: React.FC = () => {
  const features = [
    {
      icon: Brain,
      title: 'AI-Powered Generation',
      description: 'Advanced LangGraph workflow with Groq LLM for intelligent SRS document creation',
      color: 'from-purple-500 to-purple-600'
    },
    {
      icon: FileImage,
      title: 'Multi-Format Support',
      description: 'Upload PDF, Word, PowerPoint, Markdown, LaTeX files with OCR text extraction',
      color: 'from-blue-500 to-blue-600'
    },
    {
      icon: Users,
      title: 'Real-Time Collaboration',
      description: 'Live editing with multiple users, instant synchronization via WebSockets',
      color: 'from-green-500 to-green-600'
    },
    {
      icon: Download,
      title: 'Export Anywhere',
      description: 'Export your SRS documents to PDF, Word, Markdown, or LaTeX formats',
      color: 'from-orange-500 to-orange-600'
    }
  ]

  const workflow = [
    {
      step: '01',
      title: 'Upload Documents',
      description: 'Upload your requirements documents in any supported format',
      icon: Upload,
      color: 'bg-primary-500'
    },
    {
      step: '02',
      title: 'AI Processing',
      description: 'Our AI analyzes your content and generates structured SRS',
      icon: Bot,
      color: 'bg-purple-500'
    },
    {
      step: '03',
      title: 'Collaborate & Edit',
      description: 'Real-time editing with your team using our advanced editor',
      icon: Edit3,
      color: 'bg-green-500'
    },
    {
      step: '04',
      title: 'Export & Share',
      description: 'Download your professional SRS in your preferred format',
      icon: Download,
      color: 'bg-orange-500'
    }
  ]

  const stats = [
    { label: 'Documents Processed', value: '10K+' },
    { label: 'Active Users', value: '500+' },
    { label: 'Time Saved', value: '80%' },
    { label: 'Accuracy Rate', value: '95%' }
  ]

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative overflow-hidden px-4 pb-20 pt-16 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-7xl">
          <div className="grid items-center gap-12 lg:grid-cols-2">
            {/* Content */}
            <motion.div 
              initial={{ opacity: 0, x: -50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              className="text-center lg:text-left"
            >
              <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="mb-6 inline-flex items-center rounded-full bg-primary-50 px-4 py-2 text-primary-700"
              >
                <Sparkles className="mr-2 h-4 w-4" />
                <span className="text-sm font-medium">Powered by Advanced AI</span>
              </motion.div>

              <motion.h1 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="mb-6 text-4xl font-bold text-secondary-900 sm:text-5xl lg:text-6xl"
              >
                Generate Perfect{' '}
                <span className="bg-gradient-to-r from-primary-600 to-purple-600 bg-clip-text text-transparent">
                  SRS Documents
                </span>{' '}
                with AI
              </motion.h1>

              <motion.p 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
                className="mb-8 text-lg text-secondary-600 lg:text-xl"
              >
                Transform your requirements into professional Software Requirements 
                Specifications using advanced AI, with real-time collaboration and 
                multi-format support.
              </motion.p>

              <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
                className="flex flex-col space-y-4 sm:flex-row sm:space-x-4 sm:space-y-0"
              >
                <Link to="/upload" className="btn-primary group">
                  <Upload className="mr-2 h-4 w-4" />
                  Start Creating SRS
                  <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
                </Link>
                
                <Link to="/generate" className="btn-secondary">
                  <Bot className="mr-2 h-4 w-4" />
                  Try AI Generation
                </Link>
              </motion.div>

              <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6 }}
                className="mt-8 flex items-center justify-center space-x-6 lg:justify-start"
              >
                {stats.map((stat, index) => (
                  <div key={stat.label} className="text-center">
                    <div className="text-2xl font-bold text-primary-600">
                      {stat.value}
                    </div>
                    <div className="text-sm text-secondary-500">
                      {stat.label}
                    </div>
                  </div>
                ))}
              </motion.div>
            </motion.div>

            {/* Visual */}
            <motion.div 
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              className="relative"
            >
              <div className="relative mx-auto max-w-md">
                {/* Floating cards */}
                <motion.div
                  animate={{ y: [0, -10, 0] }}
                  transition={{ repeat: Infinity, duration: 4, ease: "easeInOut" }}
                  className="absolute -top-8 left-8 rounded-xl bg-white p-4 shadow-lg ring-1 ring-secondary-200"
                >
                  <div className="flex items-center space-x-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary-100">
                      <FileText className="h-5 w-5 text-primary-600" />
                    </div>
                    <div>
                      <div className="text-sm font-medium text-secondary-900">
                        SRS Document
                      </div>
                      <div className="text-xs text-secondary-500">
                        Auto-generated
                      </div>
                    </div>
                  </div>
                </motion.div>

                <motion.div
                  animate={{ y: [0, 10, 0] }}
                  transition={{ repeat: Infinity, duration: 4, ease: "easeInOut", delay: 2 }}
                  className="absolute -right-8 top-16 rounded-xl bg-white p-4 shadow-lg ring-1 ring-secondary-200"
                >
                  <div className="flex items-center space-x-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-green-100">
                      <Users className="h-5 w-5 text-green-600" />
                    </div>
                    <div>
                      <div className="text-sm font-medium text-secondary-900">
                        3 Collaborators
                      </div>
                      <div className="text-xs text-secondary-500">
                        Live editing
                      </div>
                    </div>
                  </div>
                </motion.div>

                {/* Main interface mockup */}
                <div className="relative rounded-2xl bg-white p-6 shadow-xl ring-1 ring-secondary-200">
                  <div className="mb-4 flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <div className="h-3 w-3 rounded-full bg-red-400"></div>
                      <div className="h-3 w-3 rounded-full bg-yellow-400"></div>
                      <div className="h-3 w-3 rounded-full bg-green-400"></div>
                    </div>
                    <div className="text-xs text-secondary-500">AI SRS Generator</div>
                  </div>
                  
                  <div className="space-y-3">
                    <div className="h-4 w-3/4 rounded bg-secondary-200"></div>
                    <div className="h-4 w-1/2 rounded bg-secondary-200"></div>
                    <div className="h-4 w-5/6 rounded bg-secondary-200"></div>
                    <div className="mt-4 h-20 rounded-lg bg-gradient-to-br from-primary-50 to-purple-50 p-3">
                      <div className="flex items-center space-x-2">
                        <Bot className="h-4 w-4 text-primary-600" />
                        <div className="text-xs text-primary-700">AI is generating your SRS...</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="bg-secondary-50 px-4 py-20 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-7xl">
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="mb-16 text-center"
          >
            <h2 className="mb-4 text-3xl font-bold text-secondary-900 sm:text-4xl">
              Powerful Features for Modern Teams
            </h2>
            <p className="mx-auto max-w-2xl text-lg text-secondary-600">
              Everything you need to create, collaborate, and deliver professional 
              SRS documents with the power of artificial intelligence.
            </p>
          </motion.div>

          <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-4">
            {features.map((feature, index) => {
              const Icon = feature.icon
              return (
                <motion.div
                  key={feature.title}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.6, delay: index * 0.1 }}
                  className="group"
                >
                  <div className="card h-full transition-all duration-300 hover:shadow-lg">
                    <div className={`mb-4 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-gradient-to-r text-white ${feature.color}`}>
                      <Icon className="h-6 w-6" />
                    </div>
                    <h3 className="mb-2 text-lg font-semibold text-secondary-900">
                      {feature.title}
                    </h3>
                    <p className="text-secondary-600">
                      {feature.description}
                    </p>
                  </div>
                </motion.div>
              )
            })}
          </div>
        </div>
      </section>

      {/* Workflow Section */}
      <section className="px-4 py-20 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-7xl">
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="mb-16 text-center"
          >
            <h2 className="mb-4 text-3xl font-bold text-secondary-900 sm:text-4xl">
              Simple 4-Step Process
            </h2>
            <p className="mx-auto max-w-2xl text-lg text-secondary-600">
              From upload to export, our streamlined workflow makes SRS generation 
              fast and effortless.
            </p>
          </motion.div>

          <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-4">
            {workflow.map((step, index) => {
              const Icon = step.icon
              return (
                <motion.div
                  key={step.step}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.6, delay: index * 0.1 }}
                  className="relative text-center"
                >
                  <div className="relative">
                    <div className={`mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full text-white ${step.color}`}>
                      <Icon className="h-8 w-8" />
                    </div>
                    <div className="absolute -top-2 -right-2 flex h-8 w-8 items-center justify-center rounded-full bg-secondary-900 text-xs font-bold text-white">
                      {step.step}
                    </div>
                  </div>
                  
                  <h3 className="mb-2 text-lg font-semibold text-secondary-900">
                    {step.title}
                  </h3>
                  <p className="text-secondary-600">
                    {step.description}
                  </p>

                  {/* Connector line */}
                  {index < workflow.length - 1 && (
                    <div className="absolute left-1/2 top-8 hidden w-full lg:block">
                      <div className="h-px bg-secondary-200" style={{ width: 'calc(100% + 2rem)', marginLeft: '1rem' }}></div>
                    </div>
                  )}
                </motion.div>
              )
            })}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="bg-gradient-to-r from-primary-600 to-purple-600 px-4 py-20 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-4xl text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <h2 className="mb-4 text-3xl font-bold text-white sm:text-4xl">
              Ready to Transform Your Requirements Process?
            </h2>
            <p className="mb-8 text-xl text-primary-100">
              Join thousands of teams already using AI to create better SRS documents.
            </p>
            
            <div className="flex flex-col space-y-4 sm:flex-row sm:justify-center sm:space-x-4 sm:space-y-0">
              <Link to="/upload" className="inline-flex items-center justify-center rounded-lg bg-white px-6 py-3 text-base font-medium text-primary-700 shadow-sm transition-all hover:bg-primary-50">
                <Upload className="mr-2 h-5 w-5" />
                Get Started Now
              </Link>
              
              <Link to="/generate" className="inline-flex items-center justify-center rounded-lg border border-white px-6 py-3 text-base font-medium text-white transition-all hover:bg-white/10">
                <Bot className="mr-2 h-5 w-5" />
                Try AI Generation
              </Link>
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  )
}

export default HomePage
