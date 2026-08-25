export type FruitQuality = 'good' | 'rotten'

export type PredictionResponse = {
  predicted_class: FruitQuality
  confidence: number
  probabilities: Record<FruitQuality, number>
  heatmap_data_url: string
}

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

export async function analyzeFruit(file: File): Promise<PredictionResponse> {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${apiBaseUrl}/predict`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    const payload = await response.json().catch(() => null)
    const detail = payload?.detail ?? 'Unable to analyze the image.'
    throw new Error(detail)
  }

  return response.json() as Promise<PredictionResponse>
}
