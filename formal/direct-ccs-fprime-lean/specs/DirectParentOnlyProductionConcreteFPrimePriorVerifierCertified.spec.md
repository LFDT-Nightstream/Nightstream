# DirectParentOnlyProductionConcreteFPrimePriorVerifierCertified

This module packages the concrete F′ prior-verifier target as the production
`CertifiedPriorVerifier`.

The certified verifier's acceptance predicate is the concrete runtime
exact-public-IO verifier predicate. Its opener is the fixed authority opener
from the verifier surface. Certification requires the theorem that every
accepted concrete prior proof opens to proof-carrying folded F′ authority for
the same `(steps, image)` pair.

The module derives prior-image reachability, the unreachable-prior rejection
form, and the single-terminal end-to-end theorem for callers that provide
concrete prior verification plus the latest Construction-2 step.

For the production-shaped verifier surface, the module also packages exact
runtime soundness together with exact public-IO layout binding. This produces a
certified prior verifier whose accepted exact-runtime proof opens to folded F′
authority for the identical `(steps, image)` pair, and whose certified object is
the one consumed by the terminal end-to-end theorem. The same exact-runtime
surface also induces the strict compressed-F′ `SoundVerifier` object, with
acceptance equivalent to exact-runtime replay and with same-proof functionality
inherited from the fixed opener. A latest Construction-2 step accepted after
exact-runtime prior verification is therefore terminal acceptance through this
strict `SoundVerifier`.

The exact-runtime path exposes the anti-swap facts required by the parent-only
optimization: accepted verification has a non-empty fixed opener, any opened
authority accepts the same public pair, the opened authority's step count and
image are equal to the verifier claim, and one proof cannot verify for two
different public pairs.

The strict exact-runtime `SoundVerifier` exposes the same fixed-opener facts:
acceptance rules out an empty opener, any opened authority accepts the identical
`(steps, image)` pair, the opened authority's step count and image match the
verifier claim, accepted proofs reach the claimed prior image, public-image
invariants follow from reachability, and unreachable prior images are rejected.

No digest chain, aggregate child check, or public-IO aggregate is authority in
this module. Authority is the opened folded F′ reachability object for the
exact public pair consumed by terminal compression.
