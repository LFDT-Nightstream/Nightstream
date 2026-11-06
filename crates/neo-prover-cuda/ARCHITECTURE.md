# CUDA prover boundary

`neo-prover-cuda` runs prover work on NVIDIA GPUs. It is the CUDA port of
`neo-prover-metal`: each file of `src/kernels/` ports one Metal shader
file, and each host file ports the Metal host file of the same name. It
compiles its kernels only with the `cuda` feature, which needs the pinned
cuda-oxide codegen backend (README.md). Without the feature, the crate
exports no backend.

- `CudaRowProver` is the prover type of the crate. It gives the fixed-key
  commitments and the compression work: Merkle trees, column DFTs, GKR
  layers, and commitment key weights.
- `CudaSession` owns the CUDA context, one stream, the kernel modules, the
  Poseidon2 round constants, the optional kept commitment key, and the
  buffer accounting. Every launch goes to the one stream, so launches run
  in order. Device buffers of every session of a process stay below
  `BUFFER_LIMIT_BYTES`, a fixed value that is never read from the device.

The crate does not own transcript order, proof assembly, proof encoding, or
verification. A CUDA result is not verifier authority. Each CUDA output
must be byte-identical to the CPU engine output.

`CudaSession::new` uploads `poseidon2::round_constant_words`, so the
kernels hold no Poseidon2 constants.
