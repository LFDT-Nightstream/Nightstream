import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.CandidateOrder
import Nightstream.SuperNeo.Concrete.Phi81StrongSet

/-!
Shared coefficient-to-Phi81 encoding at the `Pi_RLC` sampler boundary.

Owns: the unique conversion from a production centered-symbol wire to the
Goldilocks coefficient used by the independent Phi81 semantics.

Does not own: transcript provenance, rejection sampling, first-accepted
selection, challenge-column placement, ring-vector assembly, pairwise
strong-set security, Rust conformance, row removal, or cost totals.

Emits constraints: no.

Authority boundary: production supplies only the centered natural
representative. The mathematical coefficient meaning comes from
`Phi81StrongSet.embedCoefficient`; this file contains no second encoding table.

| Protocol | Phase | Constraint family | Mathematical obligation | Lean result |
|---|---|---|---|---|
| `Pi_RLC` | coefficient encoding | canonical field reduction | reduce one production wire modulo Goldilocks | `fieldResidue` |
| `Pi_RLC` | coefficient encoding | centered alphabet | `{-2,-1,0,1,2}` has one semantic Phi81 image | `fieldResidue_centeredField_eq_embedCoefficient` |
| `Pi_RLC` | coefficient encoding | alphabet census | exact canonical representatives | `embeddedAlphabet_values` |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.RingEncoding

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler

/-- Canonical field reduction used only at the Nat-to-Goldilocks boundary. -/
def fieldResidue (value : Nat) : F :=
  ⟨value % goldilocksP, Nat.mod_lt _ (by decide)⟩

/-- The implementation name for the independently defined semantic centered
coefficient embedding. -/
def embedCoefficient
    (coefficient : ProductionAlphabet.Coefficient) : F :=
  Nightstream.SuperNeo.Concrete.Phi81StrongSet.embedCoefficient coefficient

theorem embedCoefficient_eq_semantic
    (coefficient : ProductionAlphabet.Coefficient) :
    embedCoefficient coefficient =
      Nightstream.SuperNeo.Concrete.Phi81StrongSet.embedCoefficient coefficient := by
  rfl

/-- Reducing the production centered representative is exactly the independent
semantic embedding. This is the sole field-encoding bridge used by both fixed
NIFS profiles. -/
theorem fieldResidue_centeredField_eq_embedCoefficient
    (coefficient : ProductionAlphabet.Coefficient) :
    fieldResidue (CandidateOrder.centeredField coefficient) =
      embedCoefficient coefficient := by
  revert coefficient
  decide

/-- The five symbols embed as the canonical representatives of
`[-2,-1,0,1,2]`. -/
theorem embeddedAlphabet_values :
    (List.ofFn fun coefficient : Fin ProductionAlphabet.alphabetSize =>
      (embedCoefficient coefficient).val) =
      [goldilocksP - 2, goldilocksP - 1, 0, 1, 2] := by
  exact Nightstream.SuperNeo.Concrete.Phi81StrongSet.embedCoefficient_values

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.RingEncoding
