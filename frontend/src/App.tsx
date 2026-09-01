import { NavLink, Route, Routes } from 'react-router-dom'
import DatasetDetailPage from './pages/DatasetDetail'
import DatasetsPage from './pages/Datasets'
import GeneratePage from './pages/Generate'
import ModelsPage from './pages/Models'
import TrainPage from './pages/Train'

const navLinkClass = ({ isActive }: { isActive: boolean }) =>
  `px-3 py-2 rounded-md text-sm font-medium transition-colors ${
    isActive ? 'bg-neutral-800 text-white' : 'text-neutral-400 hover:text-white hover:bg-neutral-900'
  }`

function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-neutral-800 sticky top-0 bg-neutral-950/90 backdrop-blur z-10">
        <div className="max-w-6xl mx-auto flex items-center gap-6 px-4 py-3">
          <span className="font-semibold text-lg tracking-tight">
            imuse<span className="text-violet-400">studio</span>
          </span>
          <nav className="flex gap-1">
            <NavLink to="/" end className={navLinkClass}>
              Datasets
            </NavLink>
            <NavLink to="/train" className={navLinkClass}>
              Train
            </NavLink>
            <NavLink to="/models" className={navLinkClass}>
              Models
            </NavLink>
            <NavLink to="/generate" className={navLinkClass}>
              Generate
            </NavLink>
          </nav>
        </div>
      </header>
      <main className="flex-1 max-w-6xl w-full mx-auto px-4 py-8">
        <Routes>
          <Route path="/" element={<DatasetsPage />} />
          <Route path="/datasets/:id" element={<DatasetDetailPage />} />
          <Route path="/train" element={<TrainPage />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="/generate" element={<GeneratePage />} />
        </Routes>
      </main>
    </div>
  )
}

export default App
