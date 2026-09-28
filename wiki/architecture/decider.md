# Terminal verification

`crates/nightstream/src/lifecycle/verify.rs` checks the final fresh relation
and all running committed-evaluation claims. It checks commitments, public
projections, low-norm witness values, Pad (`Eval_K`), and every matrix
(`Eval_A`) opening separately. The verifier derives its expected relation
from the configured package.

The current terminal proof carries witness material for these checks. The
removed Spartan integration is not a supported compression backend.
A future compressed terminal backend requires its own relation proof and
conformance checks; the current API makes no such claim.

The fresh two-fold conformance workflow requires terminal acceptance,
a changed fresh relation rejection, and balanced opening changes that reach
and fail the exact `Eval_K` and `Eval_A` checks. See
[golden conformance](../../scripts/GOLDEN_CONFORMANCE.md).
