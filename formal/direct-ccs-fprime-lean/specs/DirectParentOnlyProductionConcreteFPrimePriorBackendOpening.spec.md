# DirectParentOnlyProductionConcreteFPrimePriorBackendOpening

This component specifies the backend-shaped exact public-IO opening boundary for
the production prior `F'` verifier.

The verifier-visible data mirrors the exact public-IO backend surface: canonical
statements, proof-carried statements, statement boundaries, terminal public
values, boundary public values, committed terminal proofs, compact image replay,
Construction-2 boundary replay, and transcript replay. Acceptance of these
checks is authority only when it is tied to a fixed opener for the same opaque
prior proof.

The boundary separates the cryptographic obligation into the following exact
requirements:

- terminal verifier public IO binds the canonical terminal length;
- replay checks bind the proof statement to the canonical `(steps, image)`
  statement;
- exact bound statements open through the fixed authority opener;
- any opened authority for that exact bound statement carries the same `steps`
  and public image as the verifier statement.

From these requirements, the Lean interface exposes a certified prior verifier.
Accepted backend checks open to a `ProofCarryingPriorProof`, imply
`FoldedFPrimeAuthority.Accepts` for the same public pair, reach the claimed prior
image, preserve public-image invariants, reject unreachable prior images, and
are same-proof functional.

Direct backend exact-IO checks also induce the strict `SoundVerifier` consumed
by terminal production soundness. Acceptance of that strict verifier is
equivalent to the backend checks, accepted proofs open to folded `F'` authority
for the same public pair, one opaque proof cannot certify two different public
pairs, and latest-step acceptance passes through the same strict verifier.

For a fixed accepted proof, the returned opener value is authoritative: if the
opener returns `some authority`, that exact authority accepts the same
`(steps, image)` pair; if it returns `none`, backend verification cannot hold.

A prior verifier surface that packages backend SNARK soundness as
`exactRuntimeSound` may enter this boundary through the fixed-opener path when
it also supplies exact public-IO layout binding. Raw concatenation alone is not
accepted as terminal/boundary split evidence.

The production terminal theorem consumes this certified prior verifier and
returns the parent-only end-to-end result, including the non-aggregate private
DEC/stage facts and the Section 7.1 owner-target stage audit. The boundary
assumes the backend verifier and Poseidon2 binding soundness at the opener
obligation; it does not model Poseidon2 internals.
