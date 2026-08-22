import { Link, Route, Routes } from 'react-router-dom'

import { ConfigurePage } from './features/configure/ConfigurePage'
import { HelpPage } from './features/help/HelpPage'
import { ModelsListPage } from './features/models/ModelsListPage'
import { ModelPage } from './features/results/ModelPage'
import { UploadPage } from './features/upload/UploadPage'

export default function App() {
  return (
    <>
      <header className="app-header">
        <span className="logo">A</span>
        <Link to="/">AutoML Studio</Link>
        <Link to="/help" className="header-help" title="Feature guide and how-to documentation">
          Help
        </Link>
      </header>
      <main className="page">
        <Routes>
          <Route path="/" element={<UploadPage />} />
          <Route path="/datasets/:id/configure" element={<ConfigurePage />} />
          <Route path="/datasets/:id/models" element={<ModelsListPage />} />
          <Route path="/models/:id" element={<ModelPage />} />
          <Route path="/help" element={<HelpPage />} />
        </Routes>
      </main>
    </>
  )
}
