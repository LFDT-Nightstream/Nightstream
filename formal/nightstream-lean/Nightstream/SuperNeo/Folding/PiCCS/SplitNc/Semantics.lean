import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Fe
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Nc

/-!
Independent semantic statement for the Phi81 SplitNc `Pi_CCS` reduction.

Protocol: SuperNeo `Pi_CCS`.
Phase: pre-transcript semantic statement.
Constraint family: FE fresh CCS, FE running evaluation, and NC full-carrier
strict norm.

Owns: the exact conjunction of FE and NC semantic truth and its equivalence
with all independently constructed uncompressed residual families.

Does not own: random compression, bad-root probability, either SumCheck,
Fiat--Shamir, commitments/openings, production decoding, Rust, R1CS, or
constraint counts.

Emits constraints: no.

Authority boundary: this proposition is the target that later verifier and
R1CS refinements must imply. Existing verifier acceptance or constraint
counts do not define it.

| Protocol | Phase | Family | Mathematical obligation |
|---|---|---|---|
| `Pi_CCS` | FE | fresh CCS | all explicit relation rows vanish |
| `Pi_CCS` | FE | running CE | all derived coefficient evaluations match claims |
| `Pi_CCS` | NC | all sources / full carrier | every authoritative coefficient has strict norm `< 2` |
| assurance | uncompressed residuals | FE / NC | residual zero iff the independent semantic conjunction |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources

/-- Complete independent mathematical truth for one SplitNc input product. -/
def Truth
    {shape : SemanticShape}
    (data : Data shape) : Prop :=
  Fe.Truth data /\ Nc.Truth data

/-- All uncompressed semantic residual families vanish. -/
def ResidualsZero
    {shape : SemanticShape}
    (data : Data shape) : Prop :=
  Fe.ResidualsZero data /\ Nc.ResidualsZero data

/-- Before random compression, the explicit residual families are sound and
complete for the independent SplitNc semantic statement. -/
theorem residualsZero_iff_truth
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {shape : SemanticShape}
    (data : Data shape) :
    ResidualsZero data <-> Truth data := by
  constructor
  · rintro ⟨feResiduals, ncResiduals⟩
    exact ⟨(Fe.residualsZero_iff_truth data).mp feResiduals,
      (Nc.residualsZero_iff_truth noZeroDivisors data).mp ncResiduals⟩
  · rintro ⟨feTruth, ncTruth⟩
    exact ⟨(Fe.residualsZero_iff_truth data).mpr feTruth,
      (Nc.residualsZero_iff_truth noZeroDivisors data).mpr ncTruth⟩

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics
