import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Semantics.MixedPolynomial

/-!
Contract: connect the Π_CCS NC terminal RHS to the independently evaluated
direct packed-assignment polynomial under an explicit output-projection bound.

Owns: `YZcolBound`, the dot-`chi` projection identity, the terminal RHS, and
an explicit terminal-mismatch predicate.

Does not own: proof that the current verifier establishes `YZcolBound`, output
message hashing, transcript challenges, SumCheck final-value equality,
production row lowering, or permission to remove any R1CS family.

Emits constraints: no.

Authority boundary: prover-carried `y_zcol` is never authority by itself.
`terminalRhs_eq_qNc_of_yZcolBound` requires equality to an independently
evaluated projection of the authoritative raw assignments at every consumed
lane.

| Predicate / theorem | Mathematical obligation | Assumptions | Rust owner / refinement status | Permits row removal? |
|---|---|---|---|---|
| `YZcolBound` | bind every consumed output lane to the direct assignment projection | authoritative assignments and exact output count | required authority bridge is absent/open | no |
| `dotChi_eq_zTilde_of_yZcolBound` | carried dot-`chi` equals independent `zTilde` | `YZcolBound` | NC output evaluations; row refinement open | no |
| `terminalRhs` | reproduce equality gating and `gamma^(i+1)` range mixing | shape-correct carried outputs | NC terminal identity; formula only | no |
| `terminalRhs_eq_qNc_of_yZcolBound` | terminal RHS equals the true mixed polynomial under projection authority | `YZcolBound` | production derivation of premise open | no |
| `TerminalMismatch` | expose failure when the carried terminal surface diverges | independent assignments and carried outputs | diagnostic semantic predicate | no |
-/

namespace Nightstream.Implementation.R1CS.PiCcsNc.Terminal

open Nightstream.Implementation.R1CS.PiCcsNc
open Nightstream.Implementation.R1CS.PiCcsNc.MixedPolynomial
open Nightstream.Implementation.R1CS.PiCcsNc.RangePolynomial
open Nightstream.SuperNeo.Concrete

/-- Zero default for a missing carried output. -/
def zeroYZcol : YZcol := fun _ => K.zero

/-- Exact authority premise for all carried `y_zcol` prefixes consumed by the
terminal identity. -/
structure YZcolBound
    (shape : Shape) (assignments : List (List F))
    (s : List K) (outputs : List YZcol) : Prop where
  outputCount : outputs.length = assignments.length
  lane : ∀ outputIndex laneIndex,
    outputIndex < assignments.length →
    laneIndex < shape.laneDomain →
    (outputs.getD outputIndex zeroYZcol) laneIndex =
      authoritativeYZcol shape
        (assignments.getD outputIndex []) s laneIndex

/-- Under explicit projection authority, the carried terminal dot product is
the independent direct-table evaluation. -/
theorem dotChi_eq_zTilde_of_yZcolBound
    {shape : Shape} {assignments : List (List F)}
    {s : List K} {outputs : List YZcol}
    (bound : YZcolBound shape assignments s outputs)
    {outputIndex : Nat} (outputLt : outputIndex < assignments.length)
    (alpha : List K) :
    dotChi shape (outputs.getD outputIndex zeroYZcol) alpha =
      zTilde shape (assignments.getD outputIndex []) s alpha := by
  unfold dotChi zTilde
  apply sumRange_congr
  intro lane laneLt
  rw [bound.lane outputIndex lane outputLt laneLt]

/-- Terminal-side computation over carried output projections. -/
def terminalRhs
    (shape : Shape) (betaM betaA : List K) (gamma : K)
    (outputs : List YZcol) (s alpha : List K) : K :=
  K.mul
    (K.mul (eqPoint alpha betaA) (eqPoint s betaM))
    (sumRange outputs.length fun outputIndex =>
      K.mul (powK gamma (outputIndex + 1))
        (rangeProductB2
          (dotChi shape (outputs.getD outputIndex zeroYZcol) alpha)))

/-- The terminal surface equals the independently evaluated NC polynomial
only after every consumed `y_zcol` is bound to its authoritative projection. -/
theorem terminalRhs_eq_qNc_of_yZcolBound
    {shape : Shape} {assignments : List (List F)}
    {s : List K} {outputs : List YZcol}
    (bound : YZcolBound shape assignments s outputs)
    (betaM betaA : List K) (gamma : K) (alpha : List K) :
    terminalRhs shape betaM betaA gamma outputs s alpha =
      qNc shape betaM betaA gamma assignments s alpha := by
  unfold terminalRhs qNc mixedRangePolynomial
  rw [bound.outputCount]
  congr 1
  apply sumRange_congr
  intro outputIndex outputLt
  rw [dotChi_eq_zTilde_of_yZcolBound bound outputLt alpha]

/-- Explicit diagnostic for a terminal result that does not equal the
independently evaluated NC polynomial. -/
def TerminalMismatch
    (shape : Shape) (betaM betaA : List K) (gamma : K)
    (assignments : List (List F)) (outputs : List YZcol)
    (s alpha : List K) : Prop :=
  terminalRhs shape betaM betaA gamma outputs s alpha ≠
    qNc shape betaM betaA gamma assignments s alpha

/-- An authoritative projection bound rules out terminal mismatch. -/
theorem not_terminalMismatch_of_yZcolBound
    {shape : Shape} {assignments : List (List F)}
    {s : List K} {outputs : List YZcol}
    (bound : YZcolBound shape assignments s outputs)
    (betaM betaA : List K) (gamma : K) (alpha : List K) :
    ¬ TerminalMismatch shape betaM betaA gamma
      assignments outputs s alpha := by
  intro mismatch
  exact mismatch
    (terminalRhs_eq_qNc_of_yZcolBound
      bound betaM betaA gamma alpha)

end Nightstream.Implementation.R1CS.PiCcsNc.Terminal
