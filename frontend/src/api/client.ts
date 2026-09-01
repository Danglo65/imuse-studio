const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const API_KEY = import.meta.env.VITE_API_KEY || ''

function withApiKey(url: string): string {
  if (!API_KEY) return url
  return `${url}${url.includes('?') ? '&' : '?'}api_key=${encodeURIComponent(API_KEY)}`
}

export interface Dataset {
  id: number
  name: string
  description: string
  created_at: string
}

export interface DatasetImage {
  id: number
  dataset_id: number
  filename: string
  caption: string
  created_at: string
}

export interface DatasetWithImages {
  dataset: Dataset
  images: DatasetImage[]
}

export type JobStatus = 'pending' | 'running' | 'completed' | 'failed'

export interface TrainJob {
  id: number
  dataset_id: number
  name: string
  base_model: string
  instance_prompt: string
  resolution: number
  max_train_steps: number
  learning_rate: number
  lora_rank: number
  seed: number
  status: JobStatus
  progress_step: number
  progress_total: number
  error: string | null
  created_at: string
  started_at: string | null
  finished_at: string | null
}

export interface GeneratedImage {
  id: number
  train_job_id: number | null
  base_model: string | null
  prompt: string
  negative_prompt: string
  seed: number
  filename: string
  created_at: string
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const headers = new Headers(options?.headers)
  if (API_KEY) headers.set('X-API-Key', API_KEY)
  const res = await fetch(`${API_BASE}${path}`, { ...options, headers })
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body.detail || detail
    } catch {
      // ignore body parse errors, fall back to statusText
    }
    throw new Error(detail)
  }
  if (res.status === 204) return undefined as T
  return res.json()
}

function jsonBody(body: unknown): RequestInit {
  return { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }
}

export const api = {
  health: () => request<{ status: string; mock_ml: boolean }>('/api/health'),

  listDatasets: () => request<Dataset[]>('/api/datasets'),
  getDataset: (id: number) => request<DatasetWithImages>(`/api/datasets/${id}`),
  createDataset: (name: string, description = '') =>
    request<Dataset>('/api/datasets', jsonBody({ name, description })),
  deleteDataset: (id: number) => request<{ ok: boolean }>(`/api/datasets/${id}`, { method: 'DELETE' }),
  uploadImages: (datasetId: number, files: File[], captions: string[]) => {
    const form = new FormData()
    files.forEach((f) => form.append('files', f))
    form.append('captions', JSON.stringify(captions))
    return request<DatasetImage[]>(`/api/datasets/${datasetId}/images`, { method: 'POST', body: form })
  },
  updateCaption: (datasetId: number, imageId: number, caption: string) => {
    const form = new FormData()
    form.append('caption', caption)
    return request<DatasetImage>(`/api/datasets/${datasetId}/images/${imageId}`, {
      method: 'PATCH',
      body: form,
    })
  },
  deleteImage: (datasetId: number, imageId: number) =>
    request<{ ok: boolean }>(`/api/datasets/${datasetId}/images/${imageId}`, { method: 'DELETE' }),
  imageUrl: (datasetId: number, imageId: number) =>
    withApiKey(`${API_BASE}/api/datasets/${datasetId}/images/${imageId}/file`),

  listJobs: () => request<TrainJob[]>('/api/train/jobs'),
  getJob: (id: number) => request<TrainJob>(`/api/train/jobs/${id}`),
  getJobLogs: (id: number) => request<{ logs: string }>(`/api/train/jobs/${id}/logs`),
  deleteJob: (id: number) => request<{ ok: boolean }>(`/api/train/jobs/${id}`, { method: 'DELETE' }),
  createJob: (body: {
    dataset_id: number
    name: string
    instance_prompt: string
    base_model?: string
    resolution?: number
    max_train_steps?: number
    learning_rate?: number
    lora_rank?: number
    seed?: number
  }) => request<TrainJob>('/api/train/jobs', jsonBody(body)),

  listModels: () => request<TrainJob[]>('/api/models'),

  generate: (body: {
    train_job_id?: number | null
    prompt: string
    negative_prompt?: string
    base_model?: string
    num_images?: number
    num_inference_steps?: number
    guidance_scale?: number
    resolution?: number
    seed?: number
  }) => request<GeneratedImage[]>('/api/generate', jsonBody(body)),
  generationHistory: () => request<GeneratedImage[]>('/api/generate/history'),
  generatedImageUrl: (id: number) => withApiKey(`${API_BASE}/api/generate/${id}/file`),
}
