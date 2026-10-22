# neo-ajtai

Ajtai commitments over `Rq`.

- `Commitment` and its linear operations: `scale_commitment`, `scale_commitment_add_inplace` and `s_mul_add`.
- `nightstream_fprime_setup`: the production key. SHAKE128 expands a fixed seed into each key element
  (`SETUP_ID` is `nightstream-ajtai-shake128-wide256-v1`), with κ = 22 rows. It commits signed-unit witnesses.
- `setup`, `commit` and `PP`: a key that an RNG samples. Nothing in production calls them. The tests use
  them as the reference for the production key.

The `testing` feature exposes `rot_step` to the tests that need it.
