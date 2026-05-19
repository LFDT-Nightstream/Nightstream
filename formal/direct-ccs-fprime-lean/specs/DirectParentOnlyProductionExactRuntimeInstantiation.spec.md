# Direct Parent Only Production Exact Runtime Instantiation

This component specifies the production exact-runtime instantiation of the
parent-only terminal theorem.

## Contract

Production exact verifier checks plus runtime authority soundness induce a
certified prior verifier for the Section 7.1-backed parent-only terminal path.
The exact prior verifier is authoritative only when accepted verifier evidence
opens folded F' reachability authority for the same step count and public image.

## Inputs

- Production exact verifier checks for compact public-image replay,
  Construction-2 boundary replay, Poseidon2 transcript replay, terminal public
  IO replay, final claim checks, and the fixed authority opener.
- Runtime authority soundness for those exact verifier checks.
- Production exact prior-verifier acceptance for a claimed
  `(steps, proof, image)` tuple.
- Latest Construction-2 step evidence for the terminal transition.
- Optional alternate private child table satisfying the full pointwise private
  DEC requirements for the same parent source.

## Guarantees

- Accepted exact prior verification opens a `ProofCarryingPriorProof`.
- The opened authority satisfies `FoldedFPrimeAuthority.Accepts` for the same
  step count and public image accepted by the verifier.
- Accepted exact prior verification proves prior F' reachability and rejects
  unreachable prior images.
- The induced certified prior verifier is accepted by the parent-only terminal
  theorem.
- Prior acceptance plus latest-step acceptance returns the parent-only terminal
  end-to-end package.
- The production path exposes non-aggregate private DEC and stage facts,
  pointwise no-swap evidence for alternate private DEC child tables, and the
  Section 7.1 owner-target stage audit.

## Boundary Assumptions

Poseidon2 binding and backend exact-runtime verifier soundness are trusted at
the production verifier boundary. This component does not model Poseidon2
internals or the backend proof system implementation.
