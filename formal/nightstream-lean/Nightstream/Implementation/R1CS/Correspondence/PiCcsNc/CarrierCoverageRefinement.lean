import Nightstream.Implementation.R1CS.Artifacts.PiCcsNc
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Necessity.CarrierCoverage
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Nc

/-!
Artifact-checked correspondence for the production SplitNc carrier omission.

Protocol: SuperNeo `Pi_CCS`.
Phase: packed witness source decoding before the NC SumCheck.
Constraint family: complete-carrier coverage and `y_zcol` observation.

Owns: canonical interpretation of one exact Rust-generated packed witness
pair; proof that the production logical decoder and `y_zcol` observation
collide on the pair; proof that the full decoder retains their differing tail;
exact acceptance results from the optimized public `Pi_CCS` API; and a
kernel-checked semantic witness that the two full carriers have different strict
`b = 2` NC truth.

Does not own: a general refinement theorem for the Rust helpers or `Pi_CCS`, a
claim about NIFS or F-prime acceptance, security-reduced SumCheck or transcript
soundness, R1CS rows, row removal, or constraint counts.

Emits constraints: no.

Assurance tier: artifact-checked for one exact `m = 257`, auto-R1CS, `b = 2`,
optimized `Pi_CCS` execution. Lean independently defines and kernel-checks the
NC truth predicate; generated outputs and acceptance booleans remain execution
evidence, not semantic authority. Running the Rust drift gate is part of the
evidence.

Authority boundary: the full 54-lane decode is interpreted as the semantic
carrier. The logical-width decoder and `y_zcol` projection are observations,
not authority. Equality of those observations cannot imply full-carrier norm
truth.

| Protocol | Phase | Family | Mathematical guarantee | Permits row removal? |
|---|---|---|---|---|
| packed witness | shape | `54 x 5` carrier | both generated matrices pass production shape validation | no |
| `Pi_CCS` | optimized public API | prove / verify | both exact witnesses are accepted in the generated execution | no |
| SplitNc | source decode | logical prefix | production logical outputs are identical | no |
| SplitNc | source decode | complete carrier | production full outputs differ at completed coordinate 257 | no |
| SplitNc | projection | `y_zcol` | production outputs are identical | no |
| SplitNc | independent semantics | strict norm over all 54 lanes | zero carrier satisfies; tail value `2` violates | no |
-/

namespace Nightstream.Implementation.R1CS.PiCcsNc.CarrierCoverageRefinement

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

/-- Canonical base-field interpretation of the real limb of one exported
quadratic-extension value. The imaginary-limb zero property is proved below. -/
def baseFieldOfPair (value : Nat × Nat) : F :=
  ⟨value.1 % goldilocksModulus, Nat.mod_lt _ (by decide)⟩

/-- Base-field view of one Rust-exported extension-field vector. -/
def baseFields (values : List (Nat × Nat)) : List F :=
  values.map baseFieldOfPair

/-- Independent semantic shape corresponding to one logical CCS coordinate
and one complete Phi81 carrier block. Rows and matrices are irrelevant to this
NC-only counterexample. -/
def counterexampleShape : SemanticShape where
  rowVariables := 0
  logicalWidth := 257
  freshCount := 0
  runningCount := 1
  matrixCount := 0

@[simp] theorem counterexampleShape_carrierWidth :
    counterexampleShape.carrierWidth = 270 := by
  decide

/-- Full semantic carrier read from the production zero-witness decode. -/
def zeroSemanticAssignment :
    PaperLinearAlgebra.Assignment F counterexampleShape.carrierWidth :=
  fun column =>
    baseFieldOfPair
      (Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.zeroFullDecode.getD
        column.val (0, 0))

/-- Full semantic carrier read from the production tail-witness decode. -/
def tailSemanticAssignment :
    PaperLinearAlgebra.Assignment F counterexampleShape.carrierWidth :=
  fun column =>
    baseFieldOfPair
      (Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.tailFullDecode.getD
        column.val (0, 0))

/-- The generated snapshot has the independently fixed Phi81 dimensions. -/
theorem artifact_dimensions :
    Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.ringDegree = ringDegree ∧
      Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.logicalWidth =
        counterexampleShape.logicalWidth ∧
      Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.ncBound = 2 ∧
      Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.packedColumnCount = 5 ∧
      Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.firstCompletedTail = 257 ∧
      Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.firstTailBlock = 4 ∧
      Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.firstTailLane = 41 := by
  decide

/-- Production shape validation accepts both packed matrices. This is only a
shape fact; it is not full-verifier acceptance. -/
theorem artifact_shape_validation_accepts_pair :
    Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.zeroShapeAccepted = true ∧
      Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.tailShapeAccepted = true := by
  decide

/-- The generated execution records acceptance of both exact witnesses by the
optimized public `Pi_CCS` prove/verify API. This theorem kernel-checks the
artifact value; the Rust drift gate is what connects that value to execution. -/
theorem artifact_pi_ccs_accepts_pair :
    Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.zeroPiCcsAccepted = true ∧
      Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.tailPiCcsAccepted = true := by
  decide

section ExactArtifactEvaluation

/- Exact evaluation of the 257/270-coordinate generated vectors needs more
than Lean's default recursion depth. This option is scoped to this section. -/
set_option maxRecDepth 4096

/-- Every exported extension-field value in the evidence has zero imaginary
limb, so interpreting its real limb as the NC base-field source is exact. -/
theorem artifact_extension_values_are_base :
    (∀ value ∈ Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.zeroLogicalDecode,
      value.2 = 0) ∧
      (∀ value ∈ Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.tailLogicalDecode,
        value.2 = 0) ∧
      (∀ value ∈ Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.zeroFullDecode,
        value.2 = 0) ∧
      (∀ value ∈ Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.tailFullDecode,
        value.2 = 0) ∧
      (∀ value ∈ Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.zeroYZcol,
        value.2 = 0) ∧
      (∀ value ∈ Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.tailYZcol,
        value.2 = 0) := by
  decide

/-- The production logical-width decodes cannot distinguish the pair. -/
theorem logical_decodes_equal :
    Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.zeroLogicalDecode =
      Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.tailLogicalDecode := by
  decide

/-- The production full-carrier decodes do distinguish the pair. -/
theorem full_decodes_differ :
    Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.zeroFullDecode ≠
      Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.tailFullDecode := by
  decide

/-- The exact differing full-carrier coordinate is the first coordinate after
the 257 logical coordinates, whose value is outside the strict `b = 2` range. -/
theorem tail_first_completed_coordinate :
    Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.tailFullDecode.getD
        Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.firstCompletedTail
        (0, 0) = (2, 0) ∧
      Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.zeroFullDecode.getD
        Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.firstCompletedTail
        (0, 0) = (0, 0) := by
  decide

/-- The production `y_zcol` helper also cannot distinguish the pair. -/
theorem y_zcol_outputs_equal :
    Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.zeroYZcol =
      Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.tailYZcol := by
  decide

/-- The logical output is exactly the logical-width prefix of the full output
for the tail witness in this runtime snapshot. -/
theorem tail_logical_is_full_prefix :
    Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.tailFullDecode.take
        Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.logicalWidth =
      Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.tailLogicalDecode := by
  decide

/-- The independent full-carrier materialization agrees exactly with the
production zero-witness full decode. -/
theorem zero_assignment_materializes_full_decode :
    (canonicalFinIndices counterexampleShape.carrierWidth).map
        zeroSemanticAssignment =
      baseFields
        Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.zeroFullDecode := by
  decide

/-- The independent full-carrier materialization agrees exactly with the
production tail-witness full decode. -/
theorem tail_assignment_materializes_full_decode :
    (canonicalFinIndices counterexampleShape.carrierWidth).map
        tailSemanticAssignment =
      baseFields
        Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.tailFullDecode := by
  decide

/-- The logical-width decoder sees only an in-range zero. -/
theorem tail_logical_base_fields_exact :
    baseFields
        Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.tailLogicalDecode =
      List.replicate counterexampleShape.logicalWidth 0 := by
  decide

/-- The zero-witness full decode contains exactly one zero per Phi81 lane. -/
theorem zero_full_base_fields_exact :
    baseFields
        Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.zeroFullDecode =
      List.replicate counterexampleShape.carrierWidth 0 := by
  decide

/-- The logical-width decoder sees only an in-range zero. -/
theorem tail_logical_decode_norm_bounded :
    normBounded 2
      (baseFields
        Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.tailLogicalDecode) := by
  rw [tail_logical_base_fields_exact]
  intro value member
  have valueZero : value = 0 := by
    have parts : counterexampleShape.logicalWidth ≠ 0 ∧ value = 0 := by
      simpa using member
    exact parts.2
  rw [valueZero]
  decide

/-- The production zero-witness full decode satisfies the strict norm as a
finite list. -/
theorem zero_full_decode_norm_bounded :
    normBounded 2
      (baseFields
        Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.zeroFullDecode) := by
  rw [zero_full_base_fields_exact]
  intro value member
  have valueZero : value = 0 := by
    have parts : counterexampleShape.carrierWidth ≠ 0 ∧ value = 0 := by
      simpa using member
    exact parts.2
  rw [valueZero]
  decide

/-- The production tail-witness full decode violates the strict norm as a
finite list. -/
theorem tail_full_decode_not_norm_bounded :
    ¬ normBounded 2
      (baseFields
        Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.tailFullDecode) := by
  intro bounded
  have member :
      baseFieldOfPair (2, 0) ∈
        baseFields
          Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.tailFullDecode := by
    decide
  have bad := bounded (baseFieldOfPair (2, 0)) member
  exact (by decide : ¬ centeredMagnitude (baseFieldOfPair (2, 0)) < 2) bad

end ExactArtifactEvaluation

/-- The all-zero full carrier satisfies the independent strict NC norm. -/
theorem zero_semantic_assignment_truth :
    Semantics.Nc.AssignmentTruth zeroSemanticAssignment := by
  intro column
  have member :
      zeroSemanticAssignment column ∈
        (canonicalFinIndices counterexampleShape.carrierWidth).map
          zeroSemanticAssignment := by
    apply List.mem_map.mpr
    exact ⟨column, by simp [canonicalFinIndices], rfl⟩
  rw [zero_assignment_materializes_full_decode] at member
  exact zero_full_decode_norm_bounded _ member

/-- The tail-mutated full carrier violates the independent strict NC norm at
completed coordinate 257. -/
theorem tail_semantic_assignment_not_truth :
    ¬ Semantics.Nc.AssignmentTruth tailSemanticAssignment := by
  intro truth
  apply tail_full_decode_not_norm_bounded
  rw [← tail_assignment_materializes_full_decode]
  intro value member
  rcases List.mem_map.mp member with ⟨column, _, rfl⟩
  exact truth column

/-- The exact production observation collision changes independent NC truth.

This is a kernel-checked necessity witness for complete-carrier coverage. It
does not assert that NIFS or F-prime accepts the bad member. -/
theorem production_observation_collision_changes_nc_truth :
    Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.zeroShapeAccepted = true ∧
      Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.tailShapeAccepted = true ∧
      Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.zeroLogicalDecode =
        Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.tailLogicalDecode ∧
      Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.zeroYZcol =
        Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.tailYZcol ∧
      Semantics.Nc.AssignmentTruth zeroSemanticAssignment ∧
      ¬ Semantics.Nc.AssignmentTruth tailSemanticAssignment := by
  exact ⟨artifact_shape_validation_accepts_pair.1,
    artifact_shape_validation_accepts_pair.2,
    logical_decodes_equal,
    y_zcol_outputs_equal,
    zero_semantic_assignment_truth,
    tail_semantic_assignment_not_truth⟩

/-- The exact artifact-checked optimized `Pi_CCS` execution accepts two
observation-equivalent carriers whose independently specified NC truth differs.

This is a profile-specific semantic counterexample, not a general Rust
refinement theorem and not an NIFS/F-prime acceptance theorem. -/
theorem artifact_checked_pi_ccs_acceptance_changes_nc_truth :
    Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.zeroPiCcsAccepted = true ∧
      Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.tailPiCcsAccepted = true ∧
      Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.zeroLogicalDecode =
        Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.tailLogicalDecode ∧
      Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.zeroYZcol =
        Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.tailYZcol ∧
      Semantics.Nc.AssignmentTruth zeroSemanticAssignment ∧
      ¬ Semantics.Nc.AssignmentTruth tailSemanticAssignment := by
  exact ⟨artifact_pi_ccs_accepts_pair.1,
    artifact_pi_ccs_accepts_pair.2,
    logical_decodes_equal,
    y_zcol_outputs_equal,
    zero_semantic_assignment_truth,
    tail_semantic_assignment_not_truth⟩

end Nightstream.Implementation.R1CS.PiCcsNc.CarrierCoverageRefinement
