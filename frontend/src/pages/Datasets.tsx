import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, type Dataset } from '../api/client'

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  function refresh() {
    api
      .listDatasets()
      .then(setDatasets)
      .catch((e) => setError(String(e.message || e)))
      .finally(() => setLoading(false))
  }

  useEffect(refresh, [])

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault()
    if (!name.trim()) return
    setError(null)
    try {
      await api.createDataset(name.trim(), description.trim())
      setName('')
      setDescription('')
      refresh()
    } catch (e) {
      setError(String((e as Error).message || e))
    }
  }

  async function handleDelete(id: number) {
    if (!confirm('Delete this dataset and all its images?')) return
    await api.deleteDataset(id)
    refresh()
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold">Datasets</h1>
        <p className="text-neutral-400 mt-1">
          A dataset is a set of images (plus captions) for one concept you want to teach the model
          &mdash; a person, a character, an object, or a style.
        </p>
      </div>

      <form onSubmit={handleCreate} className="flex gap-3 flex-wrap items-end bg-neutral-900 border border-neutral-800 rounded-lg p-4">
        <div className="flex flex-col gap-1">
          <label className="text-xs text-neutral-400">Name</label>
          <input
            className="bg-neutral-950 border border-neutral-700 rounded px-3 py-1.5 text-sm w-56"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. my_dog"
          />
        </div>
        <div className="flex flex-col gap-1 flex-1 min-w-[200px]">
          <label className="text-xs text-neutral-400">Description (optional)</label>
          <input
            className="bg-neutral-950 border border-neutral-700 rounded px-3 py-1.5 text-sm w-full"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="what is this concept?"
          />
        </div>
        <button className="bg-violet-600 hover:bg-violet-500 transition-colors px-4 py-1.5 rounded text-sm font-medium">
          Create dataset
        </button>
      </form>

      {error && <p className="text-red-400 text-sm">{error}</p>}

      {loading ? (
        <p className="text-neutral-500 text-sm">Loading&hellip;</p>
      ) : datasets.length === 0 ? (
        <p className="text-neutral-500 text-sm">No datasets yet &mdash; create one above to get started.</p>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {datasets.map((d) => (
            <div key={d.id} className="bg-neutral-900 border border-neutral-800 rounded-lg p-4 flex flex-col gap-2">
              <Link to={`/datasets/${d.id}`} className="font-medium hover:text-violet-400">
                {d.name}
              </Link>
              {d.description && <p className="text-sm text-neutral-400">{d.description}</p>}
              <div className="flex justify-between items-center mt-2">
                <span className="text-xs text-neutral-500">{new Date(d.created_at).toLocaleString()}</span>
                <button
                  onClick={() => handleDelete(d.id)}
                  className="text-xs text-red-400 hover:text-red-300"
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
