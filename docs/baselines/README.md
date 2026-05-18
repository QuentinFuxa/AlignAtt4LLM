# Official IWSLT 2026 baseline scores (cached)

`official_baseline_scores.json` / `.tsv` are a parsed cache of the official
IWSLT 2026 simultaneous-track baseline outputs, produced with:

```bash
python3 tools/reports/parse_official_baseline_outputs.py --output-dir docs/baselines
```

Provenance:

- Baseline repo: https://github.com/owaski/iwslt-2026-baselines
- Source archive: `outputs.zip` linked from that repo
  (https://github.com/user-attachments/files/26411361/outputs.zip), parsed
  2026-07-02. The cache exists because `user-attachments` URLs are not
  archival; keep the parsed scores here even if the upstream link rots.
- Configurations: 15 rows = {en-de, en-it, en-zh} × segment length
  {640, 960, 1280, 1600, 1920} ms, all with `mss5.0` and `h0`
  (**no context history**). These are the no-context anchor points used for
  same-chunk latency/quality comparisons.
- Metrics come verbatim from each run's `segmentation_output/scores.tsv`
  (BLEU, chrF, COMET, and the Long* latency family; LongYAAL CU is the
  latency figure quoted in our docs).

When comparing against these anchors, compare at matched chunk/segment size
first (a system point is only same-chunk evidence if its LongYAAL CU is below
the anchor's at the same segment length); points that trade extra latency for
quality are anchor-curve evidence, not same-chunk evidence.
