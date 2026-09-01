import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api, type GeneratedImage, type TrainJob } from '../api/client'

const DEFAULT_BASE_MODEL = 'runwayml/stable-diffusion-v1-5'

export default function GeneratePage() {
  const [searchParams] = useSearchParams()
  const [models, setModels] = useState<TrainJob[]>([])
  const [history, setHistory] = useState<GeneratedImage[]>([])
  const [results, setResults] = useState<GeneratedImage[]>([])
  const [error, setError] = useState<string | null>(null)
  const [generating, setGenerating] = useState(false)

  const [trainJobId, setTrainJobId] = useState<string>(searchParams.get('train_job_id') || '')
  const [prompt, setPrompt] = useState('')
  const [negativePrompt, setNegativePrompt] = useState('')
  const [numImages, setNumImages] = useState(4)
  const [steps, setSteps] = useState(30)
  const [guidanceScale, setGuidanceScale] = useState(7.5)
  const [seed, setSeed] = useState(-1)

  useEffect(() => {
    api.listModels().then(setModels)
    api.generationHistory().then(setHistory)
  }, [])

  useEffect(() => {
    const selected = models.find((m) => String(m.id) === trainJobId)
    if (selected) setPrompt((p) => p || selected.instance_prompt)
  }, [models, trainJobId])

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!prompt.trim()) return
    setGenerating(true)
    setError(null)
    try {
      const images = await api.generate({
        train_job_id: trainJobId ? Number(trainJobId) : null,
        prompt: prompt.trim(),
        negative_prompt: negativePrompt.trim(),
        base_model: DEFAULT_BASE_MODEL,
        num_images: numImages,
        num_inference_steps: steps,
        guidance_scale: guidanceScale,
        seed,
      })
      setResults(images)
      setHistory((prev) => [...images, ...prev])
    } catch (e) {
      setError(String((e as Error).message || e))
    } finally {
      setGenerating(false)
    }
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold">Generate</h1>
        <p className="text-neutral-400 mt-1">
          Generate images from the base model, or from one of your trained concepts.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="bg-neutral-900 border border-neutral-800 rounded-lg p-4 grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="flex flex-col gap-1 md:col-span-2">
          <label className="text-xs text-neutral-400">Model</label>
          <select
            className="bg-neutral-950 border border-neutral-700 rounded px-3 py-1.5 text-sm"
            value={trainJobId}
            onChange={(e) => setTrainJobId(e.target.value)}
          >
            <option value="">Base model only ({DEFAULT_BASE_MODEL})</option>
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name} &mdash; {m.instance_prompt}
              </option>
            ))}
          </select>
        </div>
        <div className="flex flex-col gap-1 md:col-span-2">
          <label className="text-xs text-neutral-400">Prompt</label>
          <textarea
            className="bg-neutral-950 border border-neutral-700 rounded px-3 py-1.5 text-sm"
            rows={2}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="a photo of sks subject on a mountain, golden hour, dramatic lighting"
          />
        </div>
        <div className="flex flex-col gap-1 md:col-span-2">
          <label className="text-xs text-neutral-400">Negative prompt (optional)</label>
          <input
            className="bg-neutral-950 border border-neutral-700 rounded px-3 py-1.5 text-sm"
            value={negativePrompt}
            onChange={(e) => setNegativePrompt(e.target.value)}
            placeholder="blurry, low quality"
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs text-neutral-400">Number of images</label>
          <input
            type="number"
            min={1}
            max={8}
            className="bg-neutral-950 border border-neutral-700 rounded px-3 py-1.5 text-sm"
            value={numImages}
            onChange={(e) => setNumImages(Number(e.target.value))}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs text-neutral-400">Seed (-1 = random)</label>
          <input
            type="number"
            className="bg-neutral-950 border border-neutral-700 rounded px-3 py-1.5 text-sm"
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value))}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs text-neutral-400">Inference steps</label>
          <input
            type="number"
            className="bg-neutral-950 border border-neutral-700 rounded px-3 py-1.5 text-sm"
            value={steps}
            onChange={(e) => setSteps(Number(e.target.value))}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs text-neutral-400">Guidance scale</label>
          <input
            type="number"
            step="0.5"
            className="bg-neutral-950 border border-neutral-700 rounded px-3 py-1.5 text-sm"
            value={guidanceScale}
            onChange={(e) => setGuidanceScale(Number(e.target.value))}
          />
        </div>
        <div className="md:col-span-2">
          <button
            disabled={generating || !prompt.trim()}
            className="bg-violet-600 hover:bg-violet-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors px-4 py-2 rounded text-sm font-medium"
          >
            {generating ? 'Generating…' : 'Generate'}
          </button>
        </div>
      </form>

      {error && <p className="text-red-400 text-sm">{error}</p>}

      {results.length > 0 && (
        <div className="space-y-2">
          <h2 className="text-lg font-medium">Latest results</h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
            {results.map((img) => (
              <img
                key={img.id}
                src={api.generatedImageUrl(img.id)}
                alt={img.prompt}
                className="rounded-lg border border-neutral-800 aspect-square object-cover w-full"
              />
            ))}
          </div>
        </div>
      )}

      {history.length > 0 && (
        <div className="space-y-2">
          <h2 className="text-lg font-medium">History</h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
            {history.map((img) => (
              <div key={img.id} className="space-y-1">
                <img
                  src={api.generatedImageUrl(img.id)}
                  alt={img.prompt}
                  className="rounded-lg border border-neutral-800 aspect-square object-cover w-full"
                />
                <p className="text-xs text-neutral-500 line-clamp-2">{img.prompt}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
