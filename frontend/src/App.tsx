import './App.css'

function App() {
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
        <div className="upload-placeholder">
          <span className="upload-icon" aria-hidden="true">↑</span>
          <h2 id="upload-title">Choose a fruit photo</h2>
          <p>JPEG, PNG or WebP, up to 10 MB</p>
          <button type="button">Select image</button>
        </div>
      </section>

      <p className="disclaimer">
        Educational prototype. Results are probabilistic and should be checked
        visually.
      </p>
    </main>
  )
}

export default App
