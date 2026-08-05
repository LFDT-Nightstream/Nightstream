# CUDA prover boundary

`neo-prover-cuda` selects CUDA device zero and then runs the canonical host
`NifsProverAdapter`.

`CudaNifsProver` is the single backend choice for the complete NIFS step. It
does not expose a separate PiCCS, PiRLC, or PiDEC mode. Its current host
delegate is `OptimizedCpuNifsProver`, so CUDA selection cannot use PaperExact
reductions.

CUDA does not yet implement the selected one-joint `PaddedRowIdentityV1`
protocol. It therefore owns no protocol messages, transcript steps, proof
fields, or verifier logic. Selecting CUDA cannot change proof bytes.

A future CUDA implementation can replace the host computation only after it
matches the independent PaperExact trace and the canonical proof bytes for:

- the complete joint polynomial;
- all SumCheck rounds;
- all transcript events and challenges;
- the terminal value and output claim; and
- PiRLC and PiDEC results.

The `cuda` feature requires the pinned cuda-oxide toolchain. Normal workspace
builds keep it disabled.

`CudaNifsProver::crosschecked` compares the complete selected result with
optimized CPU. The ignored CUDA test is ready to become a device-evaluator
gate when the one-joint kernel exists; it does not claim device evaluation now.
