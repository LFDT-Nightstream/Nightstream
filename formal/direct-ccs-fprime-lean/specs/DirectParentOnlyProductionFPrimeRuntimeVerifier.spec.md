# Direct Parent-Only Production F' Runtime Verifier

## Contract

This component exposes the production prior F' verifier boundary used by the
parent-only terminal theorem. Accepted runtime verifier audit evidence is
authoritative only when the fixed authority opener returns a folded F'
authority object for the same step count and public image.

## Inputs

- Production exact verifier checks, including public statement replay,
  transcript replay, terminal public IO replay, and the fixed authority opener.
- Runtime authority soundness for those checks.
- A verifier audit package for a claimed `(steps, proof, image)` tuple.
- The latest Construction-2 step evidence for the terminal transition.

## Guarantees

- Verifier acceptance opens an actual folded F' authority object, not a
  digest-only chain.
- The opened authority accepts the same `(steps, image)` pair as the verifier.
- Accepted audit evidence reaches the claimed prior public image and preserves
  the public-image invariants.
- A proof accepted for one public pair cannot be reused for a different public
  pair under the derived strict verifier.
- Terminal acceptance composes with the latest Construction-2 step to expose
  the end-to-end parent-only result, exact private DEC/stage facts, and the
  Section 7.1 stage audit.

## Boundary Assumptions

Poseidon2 binding and backend compressed-verifier soundness are trusted at the
runtime authority boundary. The Lean surface does not model Poseidon2
internals or the backend SNARK verifier implementation.
