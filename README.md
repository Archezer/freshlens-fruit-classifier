# FreshLens — Fruit Quality Classifier

FreshLens is an educational computer vision project that estimates the quality
of a fruit from a photo.

The first version will classify apples into two classes:

- `good` — visually normal apple with no significant visible spoilage;
- `rotten` — apple with visible rot, mold, or severe spoilage.

The application will also generate a Grad-CAM visualization. Grad-CAM is an
explainability technique that highlights image regions that influenced the
model's prediction.

> The model is an assistive visual classifier, not a food-safety expert. Its
> output depends on image quality, lighting, camera angle, and dataset quality.

## Project goals

- Build a reproducible image-classification pipeline with PyTorch.
- Use transfer learning with a pretrained CNN model.
- Expose inference through a FastAPI backend.
- Create a React interface for image upload and result visualization.
- Package the application with Docker.
- Evaluate not only accuracy, but also per-class quality and failure cases.

## MVP scope

The minimum useful version will:

1. accept one apple image;
2. validate the file type and size;
3. return the predicted class and class probabilities;
4. return a Grad-CAM heatmap overlay;
5. display the original image, prediction, confidence, and heatmap in a browser.

The MVP will not initially support real-time video, object detection, multiple
fruits in one image, medical or food-safety certification, or automatic dataset
collection from the internet.

## Dataset

The current baseline dataset contains 1,795 original apple photographs:

- `good`: 965 images;
- `rotten`: 830 images.

The images come from two independently collected original-photo datasets. Only
the apple classes are used:

- *Fresh and Rotten Fruits Dataset for Machine-Based Evaluation of Fruit
  Quality* by Sultana, Jahan, and Uddin: 200 good and 200 rotten apples.
  Source: https://data.mendeley.com/datasets/bdd69gyhv8/1
- *FruitVision: A Benchmark Dataset for Fresh, Rotten, and Formalin-mixed Fruit
  Detection*: 765 good and 630 rotten apples. The `Formalin-mixed` class is
  excluded because it does not match this project's API contract. Source:
  https://data.mendeley.com/datasets/xkbjx8959c/2

This set is sufficient for a serious transfer-learning baseline. Before a
production claim, it still needs source-aware splitting, duplicate checks,
independent held-out evaluation, and additional images representing deployment
conditions.

## Planned architecture

```text
React frontend
        |
        | multipart image upload
        v
FastAPI backend
        |
        +--> image validation and preprocessing
        +--> PyTorch model inference
        +--> Grad-CAM generation
        v
Prediction response: class, probabilities, heatmap
```

## Development roadmap

### Phase 1 — Foundation

- Define the class policy and labeling guide.
- Create the Python package structure.
- Add dependency management and project configuration.
- Add a minimal FastAPI health endpoint.
- Add basic tests and code quality checks.

### Phase 2 — Dataset

- Collect or obtain legally usable images.
- Organize data into `train`, `validation`, and `test` splits.
- Prevent near-duplicate images from leaking between splits.
- Document class balance, source, license, and labeling decisions.
- Add preprocessing and augmentation for training only.

### Phase 3 — Model training

- Start with a pretrained EfficientNet or ResNet model.
- Replace the classification head with two output classes.
- Track training, validation, and test metrics separately.
- Save the model artifact together with class names and preprocessing settings.
- Inspect a confusion matrix and representative errors.

### Phase 4 — Inference and explainability

- Implement deterministic inference preprocessing.
- Add confidence scores and a low-confidence warning.
- Implement Grad-CAM for the selected CNN layer.
- Verify that heatmaps are aligned with the input image.

### Phase 5 — API

- Add `POST /predict` for image inference.
- Validate content type, image dimensions, and maximum file size.
- Return a stable response schema.
- Add API tests for valid images and invalid uploads.

### Phase 6 — Frontend

- Add image upload and preview.
- Show prediction, probabilities, and uncertainty warning.
- Show the Grad-CAM overlay beside the original image.
- Handle loading, validation, and backend errors clearly.

### Phase 7 — Delivery

- Add Docker images and a local compose configuration.
- Add environment-based configuration.
- Add a reproducible startup guide.
- Run an end-to-end smoke test.
- Document limitations and future improvements.

## Initial repository structure

```text
freshlens-fruit-classifier/
├── app/
│   ├── api/             # FastAPI routes
│   ├── core/            # Configuration and shared utilities
│   ├── ml/              # Model loading, inference, and Grad-CAM
│   └── schemas/         # API request and response models
├── data/                # Local dataset; do not commit large raw files
├── models/              # Trained model artifacts
├── tests/               # Unit and API tests
├── frontend/            # React application
├── Dockerfile
├── pyproject.toml
└── README.md
```

## Local run

Start the backend from the repository root:

```powershell
uv run uvicorn app.main:app --reload
```

Start the frontend in a second terminal:

```powershell
cd frontend
npm run dev -- --host 0.0.0.0
```

The frontend uses `http://127.0.0.1:8000` by default. To point it at another
backend, create `frontend/.env` from `frontend/.env.example` and set
`VITE_API_BASE_URL`.

## Deployment

The backend Docker image includes `models/efficientnet_b0_final.pt` and starts
Uvicorn on Render's `PORT`. Build it locally with:

```powershell
docker build -t freshlens-api .
```

The deployment checkpoint `models/efficientnet_b0_final.pt` is versioned in
Git so that a Render Blueprint build can reproduce the service. Other model
artifacts and all dataset files remain ignored.

For Render, set `CORS_ALLOWED_ORIGINS` to the GitHub Pages origin, without the
repository path, for example `https://archezer.github.io`.

For GitHub Pages, set the repository Actions variable `VITE_API_BASE_URL` to
the deployed Render API URL, for example `https://freshlens-api.onrender.com`.
Then enable **Settings → Pages → Build and deployment → GitHub Actions**. The
workflow in `.github/workflows/deploy-frontend.yml` publishes each push to
`master`.

## Quality and reproducibility principles

- Keep training, validation, test evaluation, and inference as separate stages.
- Record random seeds, model version, class mapping, and preprocessing settings.
- Measure precision, recall, F1-score, and a confusion matrix per class.
- Treat a high-confidence wrong prediction as an important failure case.
- Do not claim food safety from a visual classification result.
- Do not commit private images, credentials, or large model artifacts without a
  deliberate storage policy.

## Status

- Dataset prepared: 1,795 original apple images.
- EfficientNet-B0 checkpoint trained with CUDA locally.
- FastAPI `POST /predict` validates uploads and returns prediction plus Grad-CAM.
- React frontend supports upload, local preview, loading state, error display,
  prediction, and Grad-CAM result.
- Dockerfile and GitHub Pages workflow are ready.
- External deployment still requires a Render service URL and a model-artifact
  delivery decision.
