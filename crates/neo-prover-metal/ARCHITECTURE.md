# Metal prover boundary

`neo-prover-metal` runs prover work on Apple GPUs. `nightstream` uses it for
`Engine::Metal` (feature `metal`). It compiles only with the `metal` feature,
which needs an Apple target. Without the feature, the crate exports no
backend.

- `MetalRowProver` is the prover type of the crate. It implements
  `PaperJointOracleBackend` for the PiCCS round oracle. It also gives the
  PiRLC witness split, the PiDEC child openings, the fixed-key commitments,
  the first unsatisfied row of a relation, the terminal row evaluations, and
  the compression work: Merkle trees, column DFTs, GKR layers, commitment
  key weights, and the queried leaves of the layer-1 setup oracles.
- `MetalSession::setup_leaves` (`session/setup.rs`, `shaders/setup.metal`)
  makes each setup oracle on the device (a key row from the production seed,
  or an uploaded table), takes its Möbius transform, applies the coset, runs
  a single-column DIF DFT, and copies back only the queried leaves. Its
  passes give each threadgroup neighboring butterfly groups, so the reads
  are coalesced.
- The round oracle reads `PaperJointOracleInput::norm_sources`. When every
  source has a norm term, every source is a signed-digit witness and has
  masks. Option B's light fold checks the fresh source's norm only: its
  running source is a Π_RLC output with no digit form, so the oracle stores
  its field values (`session/joint/dense.rs`) and the carried table and the
  openings read them.
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
