import { useEffect, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { api, type DatasetImage, type DatasetWithImages } from '../api/client'

export default function DatasetDetailPage() {
  const { id } = useParams()
  const datasetId = Number(id)
  const navigate = useNavigate()
  const [data, setData] = useState<DatasetWithImages | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)
  const fileInput = useRef<HTMLInputElement>(null)

  function refresh() {
    api
      .getDataset(datasetId)
      .then(setData)
      .catch((e) => setError(String(e.message || e)))
  }

  useEffect(refresh, [datasetId])

  async function handleFiles(files: FileList | null) {
    if (!files || files.length === 0) return
    setUploading(true)
    setError(null)
    try {
      const fileArray = Array.from(files)
      await api.uploadImages(datasetId, fileArray, fileArray.map(() => ''))
      refresh()
    } catch (e) {
      setError(String((e as Error).message || e))
    } finally {
      setUploading(false)
      if (fileInput.current) fileInput.current.value = ''
    }
  }

  async function handleCaptionChange(image: DatasetImage, caption: string) {
    await api.updateCaption(datasetId, image.id, caption)
    setData((prev) =>
      prev
        ? { ...prev, images: prev.images.map((img) => (img.id === image.id ? { ...img, caption } : img)) }
        : prev
    )
  }

  async function handleDeleteImage(imageId: number) {
    await api.deleteImage(datasetId, imageId)
    refresh()
  }

  if (error) return <p className="text-red-400 text-sm">{error}</p>
  if (!data) return <p className="text-neutral-500 text-sm">Loading&hellip;</p>

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <button onClick={() => navigate('/')} className="text-xs text-neutral-500 hover:text-neutral-300 mb-1">
            &larr; back to datasets
          </button>
          <h1 className="text-2xl font-semibold">{data.dataset.name}</h1>
          {data.dataset.description && <p className="text-neutral-400">{data.dataset.description}</p>}
        </div>
        <Link
          to={`/train?dataset_id=${datasetId}`}
          className="bg-violet-600 hover:bg-violet-500 transition-colors px-4 py-2 rounded text-sm font-medium"
        >
          Train on this dataset &rarr;
        </Link>
      </div>

      <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-4">
        <label className="text-sm font-medium block mb-2">Upload images</label>
        <input
          ref={fileInput}
          type="file"
          multiple
          accept="image/png,image/jpeg,image/webp,image/bmp"
          onChange={(e) => handleFiles(e.target.files)}
          disabled={uploading}
          className="text-sm text-neutral-400 file:mr-3 file:py-1.5 file:px-3 file:rounded file:border-0 file:bg-neutral-800 file:text-neutral-200 file:text-sm hover:file:bg-neutral-700"
        />
        {uploading && <p className="text-xs text-neutral-500 mt-2">Uploading&hellip;</p>}
        <p className="text-xs text-neutral-500 mt-2">
          For best fine-tuning results: 10&ndash;30 varied images of the same subject, cropped reasonably
          tight. Add a short caption per image below (or leave blank to use the training prompt for all).
        </p>
      </div>

      {data.images.length === 0 ? (
        <p className="text-neutral-500 text-sm">No images uploaded yet.</p>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
          {data.images.map((img) => (
            <div key={img.id} className="bg-neutral-900 border border-neutral-800 rounded-lg overflow-hidden flex flex-col">
              <img src={api.imageUrl(datasetId, img.id)} alt="" className="aspect-square object-cover w-full" />
              <div className="p-2 flex flex-col gap-1.5">
                <input
                  defaultValue={img.caption}
                  placeholder="caption (optional)"
                  onBlur={(e) => handleCaptionChange(img, e.target.value)}
                  className="bg-neutral-950 border border-neutral-700 rounded px-2 py-1 text-xs"
                />
                <button
                  onClick={() => handleDeleteImage(img.id)}
                  className="text-xs text-red-400 hover:text-red-300 self-end"
                >
                  Remove
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
