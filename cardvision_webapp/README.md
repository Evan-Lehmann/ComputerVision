# CardVision dashboard

Run from the project root:

```sh
python -m pip install -r requirements.txt
python -m cardvision_webapp.app
```

Or, from this folder, `python app.py` still starts the app.

See the root [README](../README.md) for capture permissions, selecting card regions, replaying a recorded round and reviewing the results. Model paths default to the versioned files under `models/`.

The app now runs analysis through `round_pipeline.py`. The legacy recognizer is retained only for historical comparison. A count is provisional until reviewed, and unresolved ranks prevent a complete count.

The central shared hand is monitored separately from the two primary boxes. Check its preview box, inspect the coverage panel and resolve any shared Unknown candidates. A complete rank list can still have unchecked coverage; confirm coverage issues before saving a verified review. Reanalyze existing folders after pulling updates. Primary-only mode leaves cross-view coverage unchecked.

Start recording now analyzes each saved frame continuously. Choose the table display before previewing the regions; use a separate display for the dashboard. The panel reports analyzed frames, pending frames and analysis age. Stop & drain waits for every saved frame before review. Stop during the countdown cancels startup. Counts carry across rounds within one recording; a new recording starts at zero. The engine uses MPS when available and one detector tile per call, based on the native benchmarks documented in the root README and VALIDATION.md.

### Dashboard preview

Review & corrections starts collapsed. Expand it for the existing Cards, Rounds, Coverage, corrections and downloads. The main dashboard shows a table preview alongside the count and recording status.

**Highlight $1 chip** is off on every page load. Enable it to outline the first chip in the recognized 1/2/5/10 selector row inside CardVision's screenshot preview. It never sends mouse/keyboard input or changes the website. The detector uses a cropped selector template from the supplied screen; unfamiliar layouts may not match. This is visual annotation, not a betting-window classifier or action trigger.

While recording, the preview uses saved capture frames (normally the previous saved PNG to avoid an in-progress write). With highlighting off, it still refreshes during recording. When idle, Refresh preview or enabling highlighting captures the selected display. The image source is labeled; refresh failures clear the annotation and label any remaining image as stale. No extra capture thread is started.
