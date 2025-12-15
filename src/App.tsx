import { BrowserRouter as Router, Routes, Route, useNavigate, useLocation } from 'react-router-dom'
import { AppLayout, TopNavigation, SideNavigation } from '@cloudscape-design/components'
import Dashboard from './pages/Dashboard'
import BenefitDocAll from './pages/BenefitDocAll'
import BenefitDocDeductible from './pages/BenefitDocDeductible'
import DataReportAll from './pages/DataReportAll'
import QueryBenefitDoc from './pages/QueryBenefitDoc'
import { useState } from 'react'

function AppContent() {
  const [navigationOpen, setNavigationOpen] = useState(false)
  const navigate = useNavigate()
  const location = useLocation()

  const navigationItems = [
    {
      type: 'expandable-link-group' as const,
      text: 'Automated Extractions',
      href: '#',
      items: [
        {
          type: 'link' as const,
          text: 'Benefit doc - all',
          href: '/automated-extractions/benefit-doc-all'
        },
        {
          type: 'link' as const,
          text: 'Benefit doc - deductible',
          href: '/automated-extractions/benefit-doc-deductible'
        },
        {
          type: 'link' as const,
          text: 'Data Report - all',
          href: '/automated-extractions/data-report-all'
        }
      ]
    },
    {
      type: 'expandable-link-group' as const,
      text: 'Document Query',
      href: '#',
      items: [
        {
          type: 'link' as const,
          text: 'Query Benefit doc',
          href: '/document-query/query-benefit-doc'
        }
      ]
    }
  ]

  return (
    <>
      <TopNavigation
        identity={{
          href: '/',
          title: 'Cloudscape App'
        }}
        utilities={[
          {
            type: 'button',
            text: 'Settings',
            href: '/settings'
          }
        ]}
      />
      <AppLayout
        navigationOpen={navigationOpen}
        onNavigationChange={({ detail }) => setNavigationOpen(detail.open)}
        navigation={
          <SideNavigation
            activeHref={location.pathname}
            header={{ href: '/', text: 'Navigation' }}
            onFollow={(event) => {
              if (!event.detail.external) {
                event.preventDefault()
                navigate(event.detail.href)
              }
            }}
            items={navigationItems}
          />
        }
        toolsHide
        content={
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/automated-extractions/benefit-doc-all" element={<BenefitDocAll />} />
            <Route path="/automated-extractions/benefit-doc-deductible" element={<BenefitDocDeductible />} />
            <Route path="/automated-extractions/data-report-all" element={<DataReportAll />} />
            <Route path="/document-query/query-benefit-doc" element={<QueryBenefitDoc />} />
          </Routes>
        }
      />
    </>
  )
}

function App() {
  return (
    <Router>
      <AppContent />
    </Router>
  )
}

export default App