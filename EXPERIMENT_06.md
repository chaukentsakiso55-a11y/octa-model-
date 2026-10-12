# OCTA 0.6 — First learning experiment

A CPU experiment was executed on 2026-10-12 using the OCTA model architecture and a tiny repeated synthetic pattern. The local experiment used 21,888 parameters, 100 optimizer steps, and a byte vocabulary.

Observed local results:
- Initial loss: 5.5208
- Final sampled validation loss: 0.2394
- Final sampled validation perplexity: 1.2705
- A generated sample repeated fragments of the training pattern, with mistakes.

These results demonstrate learning a trivial repeated pattern, **not** conversational intelligence or generalization to natural language. The validation data contains the same repeated pattern as training, so this is a smoke test, not a meaningful independent benchmark.

Reproduce with:
```bash
pip install -r requirements-model.txt
python -m octa.synthetic_experiment
```

The checkpoint is a separate local artifact, not committed to GitHub. The code and checkpoint use the same OCTA model parameter names. Next: validate model serving with the trained checkpoint, then move to licensed real text with non-overlapping document-level splits.
