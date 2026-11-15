# Metal prover boundary

`neo-prover-metal` runs prover work on Apple GPUs. No workspace crate
depends on it now, and `nightstream` proves on the CPU path only. It compiles
only with the `metal` feature, which needs an Apple target. Without the
feature, the crate exports no backend.

- `MetalRowProver` is the prover type of the crate. It implements
  `PaperJointOracleBackend` for the PiCCS round oracle. It also gives the
  PiRLC witness split, the PiDEC child openings, the fixed-key commitments,
  and the terminal row evaluations.
- `MetalSession` owns the device, the command queue, the compiled pipelines,
  the optional kept commitment key, and the buffer accounting. Device buffers stay below `BUFFER_LIMIT_BYTES`.

The crate does not own transcript order, proof assembly, proof encoding, or
verification. A Metal result is not verifier authority. Each Metal output
must be byte-identical to the CPU engine output.

`build.rs` compiles `shaders/nightstream.metal`, which includes the other
shader files, into one metallib that the crate embeds.
