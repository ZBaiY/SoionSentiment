#!/usr/bin/env python3
"""Training entrypoint (placeholder).

Contract (to be implemented):
- Load config from configs/*.yaml
- Build dataset and strict split protocol (time/cross-domain)
- Fine-tune encoder-only backbone (full-parameter)
- Emit run artifacts under runs/<run_id>/
  - curves (loss/metrics)
  - metrics.json
  - confusion matrix
  - errors.csv for bucketed analysis
  - run metadata (commit sha, config hash, seed, data manifest hash)
"""

def main() -> None:
    raise SystemExit("Not implemented yet")

if __name__ == "__main__":
    main()

