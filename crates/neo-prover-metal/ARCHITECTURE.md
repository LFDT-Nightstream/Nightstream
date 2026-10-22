# Metal prover boundary

`neo-prover-metal` runs prover work for `nightstream` on Apple GPUs. It
compiles only with the `metal` feature, which needs an Apple target. Without
the feature, the crate exports no backend.

- `MetalRowProver` is the type that `nightstream` uses. It implements
  `PaperJointOracleBackend` for the PiCCS round oracle. It also gives the
  PiRLC witness split, the PiDEC child openings, the fixed-key commitments,
  the terminal row evaluations, and the compression work: Merkle trees,
  column DFTs, GKR layers, and commitment key weights.
- `MetalSession` owns the device, the command queue, the compiled pipelines,
  the Poseidon2 round constants, the optional kept commitment key, and the
  buffer accounting. Device buffers stay below `BUFFER_LIMIT_BYTES`.

The crate does not own transcript order, proof assembly, proof encoding, or
verification. A Metal result is not verifier authority. Each Metal output
must be byte-identical to the CPU engine output.

`build.rs` compiles `shaders/nightstream.metal`, which includes the other
shader files, into one metallib that the crate embeds. `MetalSession::new`
uploads `poseidon2::round_constant_words`, so the shaders hold no Poseidon2
constants.
