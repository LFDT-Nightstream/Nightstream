# Direct Parent-Only Production F' Prior Verifier

## Contract

This component exposes the production prior F' verifier boundary used by the
parent-only terminal theorem. Accepted exact verifier audit evidence is
authoritative only when the fixed opener returns a folded F' authority object
for the same step count and public image.

## Inputs

- Production exact verifier checks for statement replay, Poseidon2 transcript
  replay, terminal public IO replay, final-claim checks, and the fixed
  authority opener.
- Backend compressed-verifier soundness for those checks.
- A split exact-verifier opening surface whose backend obligations separately
  prove fixed authority opening and same-statement authority binding.
- An implementation-shaped exact-runtime verifier surface plus public-IO layout
  binding, when the caller wants the production verifier bridge to construct
  the certified prior verifier directly.
- A verifier audit package for a claimed `(steps, proof, image)` tuple.
- The latest Construction-2 step evidence for the terminal transition.

## Guarantees

- Verifier audit evidence opens an actual folded F' authority object, not a
  digest-only chain.
- Exact verifier acceptance through the split opening surface derives the same
  folded F' authority-opening result.
- Exact-runtime verifier acceptance through the implementation-shaped surface
  derives the same folded F' authority-opening result without a caller-supplied
  loose opening premise.
- The opened authority accepts the same `(steps, image)` pair as the verifier.
- Accepted audit evidence reaches the claimed prior public image.
- A proof accepted for one public pair cannot be reused for a different public
  pair under the derived strict verifier.
- Terminal acceptance composes with the latest Construction-2 step to expose
  the end-to-end parent-only result, exact private DEC/stage facts, and the
  Section 7.1 stage audit.
- Exact-runtime verifier acceptance plus latest-step evidence exposes the
  concrete private DEC no-swap audit for any alternate child table satisfying
  the full pointwise private DEC requirements for the same parent source.

## Boundary Assumptions

Poseidon2 binding and backend compressed-verifier soundness are trusted at the
audit boundary. The Lean surface does not model Poseidon2 internals or the
backend SNARK verifier implementation.
