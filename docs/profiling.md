# Counters and profiling

`gradient_sync_events`: backward steps with DDP gradient reduction enabled.
`parameter_reconciliation_events`: explicit model-parameter averages.
`diagnostic_collectives`: extra drift AllReduce calls (default zero).
`diagnostic_seconds`: wall time for those diagnostics; included in epoch time.
`time_comm_param`: timed parameter averaging plus CPDP-only SlowMo correction.
Optimizer moments are not included in parameter flattening or averaging.

DDP may use several gradient buckets per logical step. NCCL kernel timings in
profile_comm.py are device-kernel occupancy, not logical event counts and not
necessarily additive exposed wall time. SyncBatchNorm and DDP buffer broadcasts
can communicate even under no_sync(). Evaluation reduces statistics and epoch
timing uses barriers; these are outside Table 2 algorithmic event counts.

The profile estimates exposed cost from profiler-disabled communication-ON/OFF
windows. OFF suppresses DDP gradients and explicit reconciliation; it does NOT
disable SyncBatchNorm. Its so-called compute floor must not be interpreted as
communication-free when SyncBatchNorm is active. The training step policy is
shared, but drift diagnostics are absent. The profiler accounts for final partial
epoch boundaries. Window-derived per-epoch estimates are not full measured epochs.

Profiling uses the public implementation and instrumentation utilities included
in this repository. The maintained profiler and trainer share `src/schedule.py`
and `src/synchronization.py`, ensuring consistent synchronization semantics.
