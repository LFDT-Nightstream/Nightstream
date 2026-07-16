import Nightstream.Implementation.R1CS.Artifacts.Phi81
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81CoefficientKernel

/-!
Exhaustive refinement of Rust's Phi81 bar matrix to the independent kernel.

Protocol: SuperNeo coefficient embedding (Section 5, Theorems 3 and 4).
Phase: runtime bar-matrix evidence to paper-derived coefficient semantics.
Constraint family: transformed basis / constant-term inner product.

Owns: canonical interpretation of every generated Rust matrix entry as a
Goldilocks element; proof that the generated `54 by 54` matrix has the exact
shape; exhaustive equality of all 2,916 entries with the independently
defined closed-form Phi81 bar transform; and transport of the semantic
constant-term law to the runtime-derived basis.

Does not own: conformance of `superneo_bar_block`,
`build_superneo_ring_forms`, the matrix cache, `Mat` indexing, row/block
packing, R1CS lowering, row removal, or constraint counts.

Emits constraints: no.

Assurance tier: artifact-checked Rust correspondence for the output of
`neo_math::superneo_bar_matrix`, conditional on running its fail-closed drift
test. The semantic target is independently defined in Lean; the generated
artifact is evidence rather than authority.

Authority boundary: no caller supplies the mathematical bar matrix. The Rust
output must equal the independent Phi81 transform entry by entry before its
constant-term property is inherited.

| Protocol | Phase | Family | Mathematical obligation |
|---|---|---|---|
| coefficient embedding | artifact shape | rows / columns | runtime export is exactly `54 by 54` |
| coefficient embedding | entry refinement | output / input coefficient | every runtime entry equals `nativeBarEntry` |
| coefficient embedding | basis refinement | transformed coefficient basis | runtime column equals independent `barBasis` |
| coefficient embedding | kernel refinement | output / row / assignment lane | every runtime-derived kernel weight equals independent `phi81Kernel` |
| coefficient embedding | semantic transport | constant-term kernel | runtime-derived kernel satisfies the Kronecker law |
-/

namespace Nightstream.Implementation.R1CS.Phi81BarMatrixRefinement

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.MatrixCoefficientSource
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81CoefficientKernel

set_option maxRecDepth 100000
set_option maxHeartbeats 2000000

/-- Row-major natural entry exported from Rust. Shape is proved separately,
so the total lookup default is unreachable for valid coefficient indices. -/
def runtimeEntryNat (output input : Fin ringDegree) : Nat :=
  (Nightstream.Implementation.R1CS.Phi81BarMatrixArtifact.rows.getD
    output.val []).getD input.val 0

/-- Canonical Goldilocks interpretation of one exported Rust entry. -/
def runtimeBarEntry (output input : Fin ringDegree) : F :=
  ⟨runtimeEntryNat output input % goldilocksModulus,
    Nat.mod_lt _ (by decide)⟩

/-- The exhaustive export has exactly the production Phi81 dimensions. -/
theorem artifact_shape :
    Nightstream.Implementation.R1CS.Phi81BarMatrixArtifact.rowCount = ringDegree ∧
      Nightstream.Implementation.R1CS.Phi81BarMatrixArtifact.columnCount = ringDegree ∧
      Nightstream.Implementation.R1CS.Phi81BarMatrixArtifact.rows.length = ringDegree ∧
      (∀ row ∈ Nightstream.Implementation.R1CS.Phi81BarMatrixArtifact.rows,
        row.length = ringDegree) := by
  decide

/-- Exhaustive Rust-to-semantics bridge for every matrix cell. -/
theorem runtimeBarEntry_eq_native :
    ∀ output input : Fin ringDegree,
      runtimeBarEntry output input = nativeBarEntry output input := by
  unfold runtimeBarEntry runtimeEntryNat nativeBarEntry
  decide

/-- Runtime-exported image of one input coefficient basis. -/
def runtimeBarBasis (input : Fin ringDegree) : RingF :=
  fun output => runtimeBarEntry output input

/-- Every runtime-exported basis column is the independent Phi81 basis. -/
theorem runtimeBarBasis_eq_barBasis (input : Fin ringDegree) :
    runtimeBarBasis input = barBasis input := by
  funext output
  exact runtimeBarEntry_eq_native output input

/-- Coefficient kernel built from the runtime-exported bar matrix while using
the independently specified Phi81 ring multiplication. -/
def runtimeKernel : CoefficientKernel F ringDegree where
  constant := constant
  weight := fun output row assignment =>
    ringFMul (runtimeBarBasis row) (ringFMonomial assignment.val 1) output

/-- The runtime-derived and independent kernels select the same constant
coefficient. -/
theorem runtimeKernel_constant_eq :
    runtimeKernel.constant = phi81Kernel.constant := by
  rfl

/-- Every runtime-derived kernel weight equals its independent Phi81 target.
The pointwise statement keeps the 2,916-entry artifact out of structure
equality normalization. -/
theorem runtimeKernel_weight_eq_phi81Kernel
    (output row assignment : Fin ringDegree) :
    runtimeKernel.weight output row assignment =
      phi81Kernel.weight output row assignment := by
  change
    ringFMul (runtimeBarBasis row) (ringFMonomial assignment.val 1) output =
      ringFMul (barBasis row) (ringFMonomial assignment.val 1) output
  rw [runtimeBarBasis_eq_barBasis]

/-- The runtime-derived coefficient kernel inherits the independent
constant-term Kronecker law only after exhaustive matrix equality. -/
theorem runtimeConstantTermLaw :
    ConstantTermLaw
      Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.baseOps
      runtimeKernel := by
  constructor
  intro row assignment
  change
    ringFMul (runtimeBarBasis row) (ringFMonomial assignment.val 1) constant =
      if row = assignment then 1 else 0
  rw [runtimeBarBasis_eq_barBasis]
  exact basisConstantTerm row assignment

end Nightstream.Implementation.R1CS.Phi81BarMatrixRefinement
