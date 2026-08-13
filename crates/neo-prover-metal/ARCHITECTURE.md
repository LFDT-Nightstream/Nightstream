# Metal prover boundary

`neo-prover-metal` runs the canonical one-joint oracle and fresh-instance
commitments on Apple GPUs.

`MetalNifsProver` is the single backend choice for the complete NIFS step. It
does not expose a separate PiCCS, PiRLC, or PiDEC mode.

The crate owns:

- Metal device, queue, pipeline, and buffer state;
- Goldilocks and quadratic-extension arithmetic;
- Poseidon2 permutation and batch hashing primitives;
- Ajtai commitments for signed-unit fresh assignments; and
- optional Nebula lane commitments; and
- the one-joint round oracle and terminal openings.

It does not own transcript order, proof assembly, proof encoding, or
verification. A Metal result is not verifier authority.

Apple builds with the `metal` feature compile the retained shader primitives.
Other builds return an explicit unavailable error. They do not run a hidden
CPU fallback.

`MetalNifsProver::crosschecked` compares the complete proof, transcript, and
running state with optimized CPU. The ignored hardware test also requires a
real Metal dispatch for fresh-instance commitment construction.
