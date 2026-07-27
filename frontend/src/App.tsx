import { Link, Route, Routes } from 'react-router-dom'

import { ConfigurePage } from './features/configure/ConfigurePage'
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
          <Route path="/models/:id" element={<ModelPage />} />
        </Routes>
      </main>
    </>
  )
}
