# MoE Prefetch Lab: interactive results dashboard

A static web page (HTML + CSS + JavaScript, no build step) for presenting the project: an architecture walkthrough
(steps 1-7) and every main result, all driven by controls for model, batch, host link and HBM cache size.

Live (after GitHub Pages is enabled): <https://alimohammad12314.github.io/Group_25_CA_PROJECT/>

| File | What it is |
|---|---|
| `index.html` | page structure and text |
| `style.css` | layout and light / dark themes |
| `app.js` | draws the charts with d3 v7 (loaded from cdnjs) |
| `data.js` | all numbers, generated from `endsem/results/` and `endsem/gpu/results/` |
| `build_data.py` | regenerates `data.js` |

## Publish with GitHub Pages

1. On GitHub: **Settings → Pages → Build and deployment → Source: GitHub Actions**.
2. Push to `main`. The workflow `.github/workflows/pages.yml` uploads this folder and deploys it (Actions tab shows
   progress; it takes about a minute).
3. The page is at `https://alimohammad12314.github.io/Group_25_CA_PROJECT/`.

## Open locally

Double-click `index.html` (it needs internet access only for the d3 library and fonts), or serve the folder:

```bash
cd dashboard && python3 -m http.server 8000     # then open http://localhost:8000
```

## Update the numbers

After re-running any experiment:

```bash
cd endsem && .venv/bin/python ../dashboard/build_data.py
```

then commit `dashboard/data.js`.

## Sections

| Section | Data | Follows the controls |
|---|---|---|
| How one MoE layer runs | – (diagram) | – |
| At a glance (5 tiles) | e3 ablation | yes |
| What each component adds | e3 ablation ladder + reference policies | yes |
| Where the design helps or hurts | e3, full design ÷ LRU + miss overlap | cache size; click a cell to select |
| Prediction accuracy | e2 (public traces) + e7 (H100, Mixtral) | model, batch; metric switch |
| Context length | e6 | model, batch, link; cache switch |
| Energy | e3 | yes |
| Bonus: real routing in LLMSimulator | bonus_llmsim | model, batch, link |
| Validation on a real H100 | gpu/results/compare.csv, e7 | fixed (Mixtral, PCIe 5.0) |

Any view can be shared as a link: the page keeps the selected controls in the URL, e.g.
`?model=DeepSeek-R1&batch=16&link=C2C-900`.
