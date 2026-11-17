# neo-ajtai

Ajtai commitments over `Rq`.

- `Commitment` and its linear operations: `scale_commitment`, `scale_commitment_add_inplace` and `s_mul_add`.
- `nightstream_fprime_setup`: the production key. SHAKE128 expands a fixed seed into each key element
  (`SETUP_ID` is `nightstream-ajtai-shake128-reject64-v1`), with κ = 18 rows and 8-byte rejection sampling. It commits signed-unit witnesses.
- `setup`, `commit` and `PP`: a key that an RNG samples. Nothing in production calls them. The tests use
  them as the reference for the production key.

The `testing` feature exposes `mul_by_x` to the tests that need it.
