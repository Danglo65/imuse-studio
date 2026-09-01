import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, type TrainJob } from '../api/client'

export default function ModelsPage() {
  const [models, setModels] = useState<TrainJob[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.listModels().then(setModels).catch((e) => setError(String(e.message || e)))
  }, [])

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Models</h1>
        <p className="text-neutral-400 mt-1">Trained LoRA adapters, ready to generate with.</p>
      </div>

      {error && <p className="text-red-400 text-sm">{error}</p>}

      {models.length === 0 ? (
        <p className="text-neutral-500 text-sm">
          No trained models yet &mdash; head to <Link to="/train" className="text-violet-400">Train</Link> to
          start one.
        </p>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {models.map((m) => (
            <div key={m.id} className="bg-neutral-900 border border-neutral-800 rounded-lg p-4 flex flex-col gap-2">
              <span className="font-medium">{m.name}</span>
              <span className="text-sm text-neutral-400">{m.instance_prompt}</span>
              <span className="text-xs text-neutral-500">
                {m.base_model} &middot; {m.max_train_steps} steps &middot; rank {m.lora_rank}
              </span>
              <span className="text-xs text-neutral-600">
                finished {m.finished_at ? new Date(m.finished_at).toLocaleString() : ''}
              </span>
              <Link
                to={`/generate?train_job_id=${m.id}`}
                className="mt-2 self-start bg-violet-600 hover:bg-violet-500 transition-colors px-3 py-1.5 rounded text-xs font-medium"
              >
                Generate with this model &rarr;
              </Link>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
