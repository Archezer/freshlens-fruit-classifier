import { useEffect, useRef, useState } from 'react'
import type { ChangeEvent } from 'react'

import { analyzeFruit } from './api/client'
import type { PredictionResponse } from './api/client'
import './App.css'

const maximumFileSize = 10 * 1024 * 1024

function App() {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)

  useEffect(() => () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl)
  }, [previewUrl])

  function selectImage(event: ChangeEvent<HTMLInputElement>) {
    const selectedFile = event.target.files?.[0]
    if (!selectedFile) return

    if (!selectedFile.type.startsWith('image/')) {
      setError('Please select an image file.')
      return
    }

    if (selectedFile.size > maximumFileSize) {
      setError('Image must not exceed 10 MB.')
      return
    }

    setFile(selectedFile)
    setPreviewUrl(URL.createObjectURL(selectedFile))
    setResult(null)
    setError(null)
  }

  async function submitAnalysis() {
    if (!file) return

    setIsAnalyzing(true)
    setError(null)

    try {
      setResult(await analyzeFruit(file))
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : 'Unable to analyze the image.',
      )
    } finally {
      setIsAnalyzing(false)
    }
  }

  return (
    <main className="app-shell">
      <header className="site-header">
        <a className="brand" href="/" aria-label="FreshLens home">
          <span className="brand-mark" aria-hidden="true">◉</span>
          FreshLens
        </a>
        <span className="header-status">AI fruit quality analysis</span>
      </header>

      <section className="hero" aria-labelledby="page-title">
        <p className="eyebrow">Computer vision for fruit quality</p>
        <h1 id="page-title">See what your fruit is hiding.</h1>
        <p className="hero-description">
          Upload a photo to classify it as good or rotten and inspect the
          model&apos;s decision with Grad-CAM.
        </p>
      </section>

      <section className="analysis-card" aria-labelledby="upload-title">
        <input
          ref={fileInputRef}
          className="visually-hidden"
          type="file"
          accept="image/jpeg,image/png,image/webp"
          onChange={selectImage}
        />
        {previewUrl ? (
          <img className="image-preview" src={previewUrl} alt="Selected fruit" />
        ) : (
          <div className="upload-placeholder">
            <span className="upload-icon" aria-hidden="true">↑</span>
            <h2 id="upload-title">Choose a fruit photo</h2>
            <p>JPEG, PNG or WebP, up to 10 MB</p>
          </div>
        )}
        <div className="actions">
          <button type="button" onClick={() => fileInputRef.current?.click()}>
            {file ? 'Choose another image' : 'Select image'}
          </button>
          <button type="button" disabled={!file || isAnalyzing} onClick={submitAnalysis}>
            {isAnalyzing ? 'Analyzing…' : 'Analyze fruit'}
          </button>
        </div>
        {isAnalyzing && (
          <div className="loading-state" role="status" aria-live="polite">
            <span className="loading-orbit" aria-hidden="true" />
            <div>
              <strong>Analyzing your fruit</strong>
              <p>Connecting to the model. The server may take up to a minute to wake up.</p>
            </div>
          </div>
        )}
        {error && <p className="message error-message" role="alert">{error}</p>}
      </section>

      {result && (
        <section className="result-card" aria-live="polite">
          <div>
            <p className="eyebrow">Prediction</p>
            <h2 className={`result-label ${result.predicted_class}`}>
              {result.predicted_class}
            </h2>
            <p>Confidence: {(result.confidence * 100).toFixed(1)}%</p>
            <div className="probabilities" aria-label="Class probabilities">
              {(['good', 'rotten'] as const).map((quality) => (
                <div className="probability" key={quality}>
                  <span>{quality}</span>
                  <strong>{(result.probabilities[quality] * 100).toFixed(1)}%</strong>
                  <div className="probability-track">
                    <span
                      className={`probability-fill ${quality}`}
                      style={{ width: `${result.probabilities[quality] * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
          <img className="heatmap" src={result.heatmap_data_url} alt="Grad-CAM explanation" />
        </section>
      )}

      <p className="disclaimer">
        Educational prototype. Results are probabilistic and should be checked visually.
      </p>
    </main>
  )
}

export default App
