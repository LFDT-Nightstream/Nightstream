# Profiling

Use the [current benchmark](../../crates/nightstream/README.md#poseidon2-benchmark)
for CPU and Metal measurements. Compare the same package, inputs, security
minimum, and lifecycle steps. Report compilation separately from execution.

The root [AGENTS.md](../../AGENTS.md#profiling) describes the profiling tools
and applicable process caps. A timed-out run is incomplete. Do not compare
measurements from different package identities as an engine speedup.

For Lean, use the active project's `scripts/validate.sh`. A compatible local
compiler can be selected through `elan run TOOLCHAIN`; the repository's
pinned toolchain remains the reference. An emitter runtime speedup does not
establish a build-time speedup.
