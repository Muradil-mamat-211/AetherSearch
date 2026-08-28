# Test Suite

The test suite covers the SFT-2000 schema/masking/collation contract, algorithm
contracts, configuration resolution, topology planning, checkpoint/resume
behavior, runtime adapters, and source-level integration boundaries.

Run the complete suite:

```bash
python -m pytest -q
```

GPU-dependent checks use the shared guard in `gpu_test_guard.py` and should
skip explicitly when CUDA hardware is unavailable. CPU-testable configuration
and mathematical contracts must still run in no-GPU environments.

Synthetic topology tests validate role mapping and derived configuration only;
they do not qualify untested hardware for model fit, runtime stability,
throughput, or training convergence.

Some test filenames preserve historical compatibility terminology. Those names
are implementation identifiers, not the public method vocabulary.
