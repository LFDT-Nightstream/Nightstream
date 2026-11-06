# CUDA prover boundary

`neo-prover-cuda` runs prover work on NVIDIA GPUs. It is the CUDA port of
`neo-prover-metal`: each file of `src/kernels/` ports one Metal shader
file, and each host file ports the Metal host file of the same name. It
compiles its kernels only with the `cuda` feature, which needs the pinned
cuda-oxide codegen backend (README.md). Without the feature, the crate
exports no backend.

- `CudaRowProver` is the prover type of the crate. It implements
  `PaperJointOracleBackend` for the PiCCS round oracle. It also gives the
  PiRLC witness mix and split, the PiDEC child openings, the fixed-key
  commitments, the first unsatisfied row of a relation, the terminal row
  evaluations, and the compression work: Merkle trees, column DFTs, GKR
  layers, and commitment key weights.
- The round oracle reads `PaperJointOracleInput::norm_sources`. When every
  source has a norm term, every source is a signed-digit witness and has
  masks. Option B's light fold checks the fresh source's norm only: its
  running source is a Π_RLC output with no digit form, so the oracle stores
  its field values (`session/joint/dense.rs`) and the carried table and the
  openings read them.
- `CudaSession` owns the CUDA context, one stream, the kernel modules, the
  Poseidon2 round constants, the optional kept commitment key, and the
  buffer accounting. Every launch goes to the one stream, so launches run
  in order. Device buffers of every session of a process stay below
  `BUFFER_LIMIT_BYTES`, a fixed value that is never read from the device.

The crate does not own transcript order, proof assembly, proof encoding, or
verification. A CUDA result is not verifier authority. Each CUDA output
must be byte-identical to the CPU engine output.

`CudaSession::new` uploads `poseidon2::round_constant_words`, so the
kernels hold no Poseidon2 constants. Metal launches exactly the threads it
names, but a CUDA launch covers whole blocks, so each kernel that Metal
launches by thread count takes that count and returns past it.
