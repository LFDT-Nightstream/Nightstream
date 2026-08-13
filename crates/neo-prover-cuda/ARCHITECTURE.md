# CUDA prover boundary

`neo-prover-cuda` is the required CUDA backend target for the canonical
one-joint NIFS protocol. The device kernel does not exist yet. Construction
returns `BackendUnavailable` until it does.

`CudaNifsProver` is the single backend choice for the complete NIFS step. It
does not expose a separate PiCCS, PiRLC, or PiDEC mode. It does not use the CPU
prover as a hidden fallback.

CUDA does not yet implement the selected one-joint `PaddedRowIdentity`
protocol. It therefore owns no protocol messages, transcript steps, proof
fields, or verifier logic.

A future CUDA implementation can replace the host computation only after it
matches the independent PaperExact trace and the canonical proof bytes for:

- the complete joint polynomial;
- all SumCheck rounds;
- all transcript events and challenges;
- the terminal value and output claim; and
- PiRLC and PiDEC results.

The `cuda` feature requires the pinned cuda-oxide toolchain. Normal workspace
builds keep it disabled. When the kernel exists, its tests must compare the
device result with the optimized CPU and PaperExact results.
