import type { JobStatus } from '../api/client'

const STYLES: Record<JobStatus, string> = {
  pending: 'bg-neutral-800 text-neutral-300',
  running: 'bg-blue-950 text-blue-300 animate-pulse',
  completed: 'bg-emerald-950 text-emerald-300',
  failed: 'bg-red-950 text-red-300',
}

export default function StatusBadge({ status }: { status: JobStatus }) {
  return (
    <span className={`px-2 py-0.5 rounded-full text-xs font-medium uppercase tracking-wide ${STYLES[status]}`}>
      {status}
    </span>
  )
}
