import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api, type Dataset, type TrainJob } from '../api/client'
import StatusBadge from '../components/StatusBadge'

const DEFAULT_BASE_MODEL = 'runwayml/stable-diffusion-v1-5'

export default function TrainPage() {
  const [searchParams] = useSearchParams()
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [jobs, setJobs] = useState<TrainJob[]>([])
  const [error, setError] = useState<string | null>(null)
  const [expandedLogs, setExpandedLogs] = useState<Record<number, string>>({})

  const [datasetId, setDatasetId] = useState<string>(searchParams.get('dataset_id') || '')
  const [name, setName] = useState('')
  const [instancePrompt, setInstancePrompt] = useState('a photo of sks subject')
  const [baseModel, setBaseModel] = useState(DEFAULT_BASE_MODEL)
  const [resolution, setResolution] = useState(512)
  const [steps, setSteps] = useState(800)
  const [lr, setLr] = useState(0.0001)
  const [rank, setRank] = useState(4)
  const [submitting, setSubmitting] = useState(false)

  function refreshJobs() {
    api.listJobs().then(setJobs).catch((e) => setError(String(e.message || e)))
  }

  useEffect(() => {
    api.listDatasets().then(setDatasets)
    refreshJobs()
  }, [])

  useEffect(() => {
    const hasActive = jobs.some((j) => j.status === 'pending' || j.status === 'running')
    if (!hasActive) return
    const interval = setInterval(refreshJobs, 2000)
    return () => clearInterval(interval)
  }, [jobs])

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!datasetId || !name.trim() || !instancePrompt.trim()) return
    setSubmitting(true)
    setError(null)
    try {
      await api.createJob({
        dataset_id: Number(datasetId),
        name: name.trim(),
        instance_prompt: instancePrompt.trim(),
        base_model: baseModel,
        resolution,
        max_train_steps: steps,
        learning_rate: lr,
        lora_rank: rank,
      })
      setName('')
      refreshJobs()
    } catch (e) {
      setError(String((e as Error).message || e))
    } finally {
      setSubmitting(false)
    }
  }

  async function toggleLogs(jobId: number) {
    if (expandedLogs[jobId] !== undefined) {
      setExpandedLogs((prev) => {
        const next = { ...prev }
        delete next[jobId]
        return next
      })
      return
    }
    const { logs } = await api.getJobLogs(jobId)
    setExpandedLogs((prev) => ({ ...prev, [jobId]: logs }))
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold">Train</h1>
        <p className="text-neutral-400 mt-1">
          Fine-tune a LoRA adapter on top of a base Stable Diffusion model using one of your datasets.
          Training runs as a background job &mdash; for real (non-mock) training this needs a GPU, so
          deploy the backend on a cloud GPU box (see docs/TRAINING.md).
        </p>
      </div>

      <form onSubmit={handleSubmit} className="bg-neutral-900 border border-neutral-800 rounded-lg p-4 grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="flex flex-col gap-1">
          <label className="text-xs text-neutral-400">Dataset</label>
          <select
            className="bg-neutral-950 border border-neutral-700 rounded px-3 py-1.5 text-sm"
            value={datasetId}
            onChange={(e) => setDatasetId(e.target.value)}
          >
            <option value="">Select a dataset&hellip;</option>
            {datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name}
              </option>
            ))}
          </select>
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs text-neutral-400">Job name</label>
          <input
            className="bg-neutral-950 border border-neutral-700 rounded px-3 py-1.5 text-sm"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. my_dog v1"
          />
        </div>
        <div className="flex flex-col gap-1 md:col-span-2">
          <label className="text-xs text-neutral-400">
            Instance prompt (used as the caption for images without one &mdash; use a rare token like
            "sks" so it doesn't collide with concepts the base model already knows)
          </label>
          <input
            className="bg-neutral-950 border border-neutral-700 rounded px-3 py-1.5 text-sm"
            value={instancePrompt}
            onChange={(e) => setInstancePrompt(e.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs text-neutral-400">Base model (Hugging Face repo id)</label>
          <input
            className="bg-neutral-950 border border-neutral-700 rounded px-3 py-1.5 text-sm"
            value={baseModel}
            onChange={(e) => setBaseModel(e.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs text-neutral-400">Resolution</label>
          <input
            type="number"
            step={64}
            className="bg-neutral-950 border border-neutral-700 rounded px-3 py-1.5 text-sm"
            value={resolution}
            onChange={(e) => setResolution(Number(e.target.value))}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs text-neutral-400">Training steps</label>
          <input
            type="number"
            className="bg-neutral-950 border border-neutral-700 rounded px-3 py-1.5 text-sm"
            value={steps}
            onChange={(e) => setSteps(Number(e.target.value))}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs text-neutral-400">Learning rate</label>
          <input
            type="number"
            step="0.00001"
            className="bg-neutral-950 border border-neutral-700 rounded px-3 py-1.5 text-sm"
            value={lr}
            onChange={(e) => setLr(Number(e.target.value))}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs text-neutral-400">LoRA rank</label>
          <input
            type="number"
            className="bg-neutral-950 border border-neutral-700 rounded px-3 py-1.5 text-sm"
            value={rank}
            onChange={(e) => setRank(Number(e.target.value))}
          />
        </div>
        <div className="md:col-span-2">
          <button
            disabled={submitting || !datasetId}
            className="bg-violet-600 hover:bg-violet-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors px-4 py-2 rounded text-sm font-medium"
          >
            {submitting ? 'Starting…' : 'Start training job'}
          </button>
        </div>
      </form>

      {error && <p className="text-red-400 text-sm">{error}</p>}

      <div className="space-y-3">
        <h2 className="text-lg font-medium">Jobs</h2>
        {jobs.length === 0 ? (
          <p className="text-neutral-500 text-sm">No training jobs yet.</p>
        ) : (
          jobs.map((job) => {
            const pct = job.progress_total > 0 ? Math.round((job.progress_step / job.progress_total) * 100) : 0
            return (
              <div key={job.id} className="bg-neutral-900 border border-neutral-800 rounded-lg p-4 space-y-2">
                <div className="flex items-center justify-between gap-3 flex-wrap">
                  <div>
                    <span className="font-medium">{job.name}</span>{' '}
                    <span className="text-xs text-neutral-500">&mdash; {job.instance_prompt}</span>
                  </div>
                  <StatusBadge status={job.status} />
                </div>
                {(job.status === 'running' || job.status === 'pending') && (
                  <div className="w-full bg-neutral-800 rounded-full h-1.5 overflow-hidden">
                    <div className="bg-violet-500 h-full transition-all" style={{ width: `${pct}%` }} />
                  </div>
                )}
                {job.status === 'failed' && job.error && (
                  <p className="text-red-400 text-xs break-words">{job.error}</p>
                )}
                <div className="flex justify-between items-center text-xs text-neutral-500">
                  <span>
                    {job.base_model} &middot; {job.max_train_steps} steps &middot; rank {job.lora_rank}
                  </span>
                  <button onClick={() => toggleLogs(job.id)} className="text-neutral-400 hover:text-white">
                    {expandedLogs[job.id] !== undefined ? 'Hide logs' : 'View logs'}
                  </button>
                </div>
                {expandedLogs[job.id] !== undefined && (
                  <pre className="bg-black rounded p-2 text-xs overflow-x-auto max-h-64 overflow-y-auto text-neutral-400">
                    {expandedLogs[job.id] || '(no logs yet)'}
                  </pre>
                )}
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}
