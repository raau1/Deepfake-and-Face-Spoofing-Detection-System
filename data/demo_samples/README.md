# Demo samples

This folder holds a small set of short video clips drawn from the five
evaluation datasets (FaceForensics++, Celeb-DF v2, DeepFakeDetection, DFDC,
WildDeepfake), provided so the deployed FastAPI service can be exercised
end-to-end without the marker having to re-download any full dataset.

## Licence and use

These clips are derivative samples of academic deepfake-detection datasets
each released under terms that **prohibit further redistribution**. They are
provided here exclusively for academic evaluation of this dissertation
submission (CO3201, University of Leicester, Raul Blanco Vazquez,
student 239034344) and **must not be redistributed, reposted, mirrored, or
shared in any form** beyond marking this submission. If you require further
samples for any other purpose, download the original datasets from the
sources documented in `../data/README.md`.

## Layout

```
demo_samples/
  ff_real/         500.mp4
  ff_fake/         01_02__talking_against_wall__YVGY8LOK.mp4
  celebdf_real/    id5_0003.mp4
  celebdf_fake/    id5_id59_0003.mp4
  dfdc_real/       cqxxumarvp.mp4
  dfdc_fake/       qeaxtxpvyq.mp4, aassnaulhq.mp4
```

WildDeepfake is intentionally absent from the demo set. The dataset ships
as extracted PNG frames rather than source videos, and the FastAPI service
only accepts video formats (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm` per
`config/config.yaml`). WildDeepfake's contribution to the project is the
batch-evaluation row in section 6.10, and the corresponding AUC numbers
remain reproducible from the JSONs in
`outputs/results/{ensemble_v1,temporal_v1,robust_v2,robust,ensemble,temporal}_eval_wilddeepfake/`.

## Reference clips

Two DFDC fakes are provided, picked to cover the two opposite ends of the
project's confidence regime on out-of-family content.

**`dfdc_fake/aassnaulhq.mp4` is the borderline / multi-model-disagreement clip.**
This is the smoke-test clip referenced in section 5.7 and section 6.10 of
the dissertation. Uploading it through the FastAPI UI should reproduce the
documented multi-model verdicts within the inconclusive-band tolerance:

| Model | Verdict | p(fake) |
|---|---|---|
| mixed | FAKE | ~0.67 |
| robust | REAL | ~0.18 |
| ensemble | REAL | ~0.40 (just outside the 0.4 to 0.6 inconclusive band) |
| temporal | REAL | ~0.01 |

The four-way disagreement and the proximity to the inconclusive band are
the design feature being demonstrated, not a misclassification. DFDC is
the project's only genuinely out-of-family evaluation dataset (its
manipulations - DFAE, MM/NN, NTH, FSGAN, StyleGAN swaps - never appear in
training), so per-clip uncertainty is expected and section 6.10.6 reports
combined out-of-family AUC of 0.61 to 0.72.

**`dfdc_fake/qeaxtxpvyq.mp4` is the unambiguous fake.**
A second DFDC fake on which all four models converge to FAKE at very high
confidence (every model returns p(fake) >= 0.99). Use this clip to confirm
that the deployed pipeline catches obvious fakes cleanly, without the
multi-model disagreement that aassnaulhq is designed to provoke.

| Model | Verdict | p(fake) |
|---|---|---|
| mixed | FAKE | ~1.00 |
| ensemble | FAKE | ~1.00 |
| robust | FAKE | ~1.00 |
| temporal | FAKE | ~1.00 |

Verify the deployed system by uploading aassnaulhq first (sees the design
nuance) and qeaxtxpvyq second (sees the strong-fake case). Together they
cover both faces of the section 6.10.6 out-of-family analysis.

## How to use

With the API container running (see the project root README for
`docker compose up`):

1. Open `http://localhost:8000/` in a browser.
2. Drop any clip from this folder into the upload zone.
3. Pick a model (the four options correspond to the FastAPI checkpoints).
4. Optionally enable Grad-CAM and TTA toggles.
5. Submit and inspect the verdict card, per-frame probability bar chart, and
   (if Grad-CAM was enabled) the heatmap tile grid.

For a richer test, run each clip through all four models and compare. The
mix of clips covers the typical capability and failure modes documented in
section 6: in-family wins (FF++, Celeb-DF), in-the-wild stress
(WildDeepfake), and out-of-family pressure (DFDC).
