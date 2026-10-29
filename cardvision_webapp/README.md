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
