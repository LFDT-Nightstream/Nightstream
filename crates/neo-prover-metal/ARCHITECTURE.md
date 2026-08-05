# Metal prover boundary

`neo-prover-metal` accelerates protocol-neutral arithmetic and fresh-instance
commitments on Apple GPUs. The selected NIFS proof itself runs through the
canonical host prover.

`MetalNifsProver` is the single backend choice for the complete NIFS step. It
does not expose a separate PiCCS, PiRLC, or PiDEC mode. Its proof delegate is
`OptimizedCpuNifsProver`, so Metal selection cannot use PaperExact reductions.

The crate owns:

- Metal device, queue, pipeline, and buffer state;
- Goldilocks and quadratic-extension arithmetic;
- Poseidon2 permutation and batch hashing primitives;
- Ajtai commitments for signed-unit fresh assignments; and
- optional Nebula lane commitments.

It does not own PiCCS, PiRLC, PiDEC, transcript order, proof assembly, proof
encoding, or verification. A Metal result is not verifier authority.

`MetalNifsProver::build_fresh_instances` can use Metal for commitments.
`MetalNifsProver::prove` delegates the selected one-joint protocol to
`OptimizedCpuNifsProver`. A future joint-polynomial kernel must pass complete
PaperExact and byte-parity checks before this boundary can change.

Apple builds with the `metal` feature compile the retained shader primitives.
Other builds use the API-compatible unavailable backend.

`MetalNifsProver::crosschecked` compares the complete proof, transcript, and
running state with optimized CPU. The ignored hardware test also requires a
real Metal dispatch for fresh-instance commitment construction.
