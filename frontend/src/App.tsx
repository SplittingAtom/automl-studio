import { Link, Route, Routes } from 'react-router-dom'

import { ConfigurePage } from './features/configure/ConfigurePage'
import { ModelsListPage } from './features/models/ModelsListPage'
import { ModelPage } from './features/results/ModelPage'
import { UploadPage } from './features/upload/UploadPage'

export default function App() {
  return (
    <>
      <header className="app-header">
        <span className="logo">A</span>
        <Link to="/">AutoML Studio</Link>
      </header>
      <main className="page">
        <Routes>
          <Route path="/" element={<UploadPage />} />
          <Route path="/datasets/:id/configure" element={<ConfigurePage />} />
          <Route path="/datasets/:id/models" element={<ModelsListPage />} />
          <Route path="/models/:id" element={<ModelPage />} />
        </Routes>
      </main>
    </>
  )
}
