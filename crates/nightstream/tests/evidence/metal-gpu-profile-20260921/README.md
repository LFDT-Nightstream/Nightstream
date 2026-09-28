# Metal GPU profiling — 2026-09-21

The owner approved a 30-minute Instruments/xctrace cap in AGENTS.md. Other
non-Lean tests retain the five-minute cap. This slice added names to existing
commitment and opening encoders so Instruments can identify their work.
Arithmetic, circuit, profile, inputs, and Cargo features are unchanged.

The native three-step benchmark completed and verified under Instruments in
190.291429417 s. Recording and saving the trace took 749.92 s. The external
observer sampled 16,854,876,160 bytes peak RSS, below the stated 16 GiB guard.
This sample maximum is not wait4 `ru_maxrss`. The uninstrumented timing remains
179.514135833 s; no CPU/Metal speed ratio is inferred from profiling.
Inputs, circuit identity, and final state match that saved benchmark.

## Measured GPU work

The native process has 779 GPU execution intervals, with 119.982026133 s of
active time. Other-process intervals were excluded. Intervals can merge several
encoders, so their counts are not dispatch counts.

| Named work | GPU seconds |
| --- | ---: |
| Production commitment and its reductions | 60.776 |
| Opening bar transform and sparse ring products | 22.703 |
| Geometric opening forms, separately recorded intervals | 13.723 |
| Main joint-round intervals | 12.197 |
| Separate chunk sums and final ring reductions | 0.047 |

Some smaller intervals combine these stages. The
[native summary](native-profile-summary.json) and
[target intervals](metal-gpu-direct-target-intervals.json) preserve that detail.
The serial chunk-sum loop is not a measured major cost; it is not an optimization
target from this profile.

Shader Timeline attributed 48.967 s to `production_ajtai_partials`, 14.955 s to
`dec_sparse_ring_partials`, 8.827 s to `joint_round_partials`, and 7.640 s to
`dec_add_geometric_ring_forms`. Commitment reduction received 0.317 s. These
are sampled attributions, not independent per-dispatch stopwatch measurements.
The total attributed shader time is 89.328 s; it does not cover all GPU-active
time. See [the shader summary](metal-gpu-direct-shader-summary.json).

The earlier attach capture recorded 0.031 s of command-buffer encoding. Nested
encoder times were not added again, and queue latencies were not treated as
idle time. Gaps in this process's GPU work may contain CPU work or other GPU
users. The main measured work is in commitment and opening kernels.

## Device limits and guidance

A native Metal query of the compiled pipelines reports SIMD width 32 and a
32,768-byte threadgroup-memory maximum. The commitment kernel uses 128 threads
and 14,688 bytes of threadgroup memory: a 32-column key tile plus static scratch.
It already shares that key tile across the supplied witnesses. The sparse
product kernel has a 128-thread maximum; the other inspected kernels report
1024. See [the native query](metal-pipeline-properties.json).

The tuning method follows Apple's advice to measure GPU timelines and shader
work, then examine resource use and memory traffic. CUDA's guidance on adjacent
memory access, shared-memory reuse, occupancy, and reductions is applicable as
an approach. Threadgroup choices must use Metal's actual pipeline limits.
Goldilocks arithmetic remains exact; smaller floating-point types cannot
replace it.

- [Apple GPU profiling](https://developer.apple.com/documentation/xcode/optimizing-gpu-performance)
- [Apple shader and memory guidance](https://developer.apple.com/videos/play/wwdc2020/10632/)
- [Metal threadgroup sizes](https://developer.apple.com/documentation/metal/calculating-threadgroup-and-grid-sizes)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)

## Capture limits

The initial attach recording hit the old 300-second cap during finalization;
its raw ATRC stream was recovered and exported. A second recording selected the
Python monitor for shader counters. The final recording selected native PID
55664 directly and obtained Shader Timeline data. Its source build includes
the new encoder labels. All three workloads verified.

There are 31 registered counter definitions, but the high-level limiter table
export contains no rows. Full and measured-range raw-counter exports did not
produce usable values and were stopped. No occupancy, bandwidth, ALU-utilization,
or spill-rate result is claimed. The trace also reports `Data stream: Time
Mapping` in its issue store; this record does not assign a cause to that message.
See [the limiter status](metal-gpu-limiter-status.json).

The complete native trace is retained in the private run directory at
`metal-gpu-direct.trace`; its all-process data and symbol archives are not added
to git. Release build, cargo fmt, and the diff whitespace check passed. No
algorithm optimization was made in this profiling slice. The next performance
work should use the measured commitment, ring-product, and geometric-form
costs. The full CPU baseline, the 5× lifecycle claim, and a memory bound for all
supported circuits remain open.
