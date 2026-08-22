import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.PiRLCFinite

/-! Provenance: copied from `formal/nightstream-lean/Nightstream/SuperNeo/Concrete/Phi81Relation/PiRLCAlgebra/PublicInput.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespaces renamed, otherwise unchanged. -/

/-!
Whole-ring public-input homomorphism for the typed Phi81 `Pi_RLC` action.

Protocol: SuperNeo Theorem 5, public-input branch of `Pi_RLC`.
Phase: complete assignment action to the verifier-owned aligned public prefix.
Constraint family: semantic public-input combination only; this file emits no
rows.

Owns: the block/lane view of the exact `54 * publicRingColumns` public carrier;
the independently executable `RingF` action on that carrier; the proof that
projection of one complete-assignment action equals the public-only action;
canonical finite public-input combination; and the exact
`PiRLC.Algebra.publicInput_hom`-shaped theorem.

Does not own: the assignment or evaluation action, commitments, challenge-set
membership, norm growth, transcript derivation, Rust/R1CS refinement, row
removal, or constraint counts.

Emits constraints: no.

Authority boundary: `combinePublicInputs` consumes only public inputs and
challenges. It is not defined through hidden assignments or a caller-supplied
projection oracle. The proof relies on the typed public width being a complete
number of Phi81 blocks; the legacy 257-field prefix is intentionally
unrepresentable by `Phi81Relation.PublicInput`.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_rlc.verify.public_input_hom.block` | every public block contains exactly 54 authoritative lanes | typed input | `publicBlock` |
| `nifs.pi_rlc.verify.public_input_hom.action` | one public block uses the same executable `ringFMul` as the complete assignment | computed | `publicAct` |
| `nifs.pi_rlc.verify.public_input_hom.projection` | projection commutes with one challenge action | derived | `projectPublicInput_act` |
| `nifs.pi_rlc.verify.public_input_hom.finite` | every public coordinate uses the canonical head-first challenge fold | computed/derived | `combinePublicInputs`, `projectPublicInput_combine` |
| `nifs.pi_rlc.verify.public_input_hom.algebra` | theorem has the exact algebra-field signature | derived | `relation_publicInput_hom` |
-/

namespace NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.PublicInput

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Phi81Relation
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

/-! ## Exact public block layout -/

/-- One public carrier block. Its type makes a partial final block impossible. -/
def publicBlock {shape : Shape}
    (input : PublicInput shape) (block : Fin shape.publicRingColumns) : RingF :=
  fun lane => input ⟨block.val * ringDegree + lane.val, by
    have blockLt := block.isLt
    have laneLt : lane.val < 54 := by
      change lane.val < 54
      exact lane.isLt
    change block.val * 54 + lane.val < 54 * shape.publicRingColumns
    omega⟩

/-- Public block containing one aligned public coordinate. -/
def publicBlockIndex (shape : Shape) (column : Fin shape.publicWidth) :
    Fin shape.publicRingColumns :=
  ⟨column.val / ringDegree, by
    have columnLt := column.isLt
    simp only [Shape.publicWidth, ringDegree] at columnLt ⊢
    omega⟩

/-- Lane containing one aligned public coordinate. -/
def publicLaneIndex {shape : Shape} (column : Fin shape.publicWidth) :
    Fin ringDegree :=
  ⟨column.val % ringDegree, Nat.mod_lt _ (by simp [ringDegree])⟩

/-- Embed a public ring block into the complete carrier's block domain. -/
def carrierBlock {shape : Shape} (block : Fin shape.publicRingColumns) :
    Fin (Phi81ColumnLayout.blockCount shape.carrierWidth) :=
  ⟨block.val, by
    change block.val < Phi81ColumnLayout.blockCount
      (Phi81CarrierLayout.carrierWidth shape.logicalWidth)
    rw [Phi81CarrierLayout.blockCount_carrierWidth]
    have fits := shape.publicFits
    rw [Phi81CarrierLayout.carrierWidth_eq] at fits
    have publicBlocksFit :
        shape.publicRingColumns <=
          Phi81ColumnLayout.blockCount shape.logicalWidth := by
      simp only [ringDegree] at fits ⊢
      omega
    exact Nat.lt_of_lt_of_le block.isLt publicBlocksFit⟩

/-- The complete carrier block and the public-only block read the same 54
assignment coordinates. -/
theorem assignmentBlock_projectPublicInput
    {shape : Shape} (assignment : Assignment shape)
    (block : Fin shape.publicRingColumns) :
    CarrierAction.assignmentBlock assignment (carrierBlock block) =
      publicBlock (projectPublicInput assignment) block := by
  funext lane
  unfold CarrierAction.assignmentBlock publicBlock projectPublicInput
  apply congrArg assignment
  apply Fin.ext
  rfl

private theorem decode_publicColumn
    {shape : Shape} (column : Fin shape.publicWidth) :
    Phi81ColumnLayout.decode (shape.publicColumn column) =
      (carrierBlock (publicBlockIndex shape column), publicLaneIndex column) := by
  apply Prod.ext
  · apply Fin.ext
    rfl
  · apply Fin.ext
    rfl

/-! ## Public-only action and finite combination -/

/-- The executable `RingF` action computed solely from one public input. -/
def publicAct {shape : Shape}
    (challenge : RingF) (input : PublicInput shape) : PublicInput shape :=
  fun column =>
    ringFMul challenge (publicBlock input (publicBlockIndex shape column))
      (publicLaneIndex column)

/-- Pointwise addition on the aligned public carrier. -/
def publicAdd {shape : Shape}
    (left right : PublicInput shape) : PublicInput shape :=
  fun column => left column + right column

/-- Canonical zero aligned public carrier. -/
def publicZero {shape : Shape} : PublicInput shape := fun _ => 0

/-- Projection commutes with one complete-carrier challenge action. -/
theorem projectPublicInput_act
    {shape : Shape} (challenge : RingF) (assignment : Assignment shape) :
    projectPublicInput (CarrierAction.act challenge assignment) =
      publicAct challenge (projectPublicInput assignment) := by
  funext column
  unfold projectPublicInput CarrierAction.act publicAct
  simp only [decode_publicColumn, assignmentBlock_projectPublicInput]
  rfl

/-- Projection commutes with pointwise assignment addition. -/
theorem projectPublicInput_add
    {shape : Shape} (left right : Assignment shape) :
    projectPublicInput (BaseLinear.assignmentAdd left right) =
      publicAdd (projectPublicInput left) (projectPublicInput right) := by
  rfl

/-- Projection of the complete-carrier zero is the public zero. -/
theorem projectPublicInput_zero {shape : Shape} :
    projectPublicInput (BaseLinear.assignmentZero : Assignment shape) =
      publicZero := by
  rfl

/-- Canonical head-first finite combination computed only from public inputs. -/
def combinePublicInputs {shape : Shape} :
    {count : Nat} ->
      (Fin count -> RingF) ->
      (Fin count -> PublicInput shape) -> PublicInput shape
  | 0, _, _ => publicZero
  | _ + 1, challenges, inputs =>
      publicAdd
        (publicAct (challenges 0) (inputs 0))
        (combinePublicInputs
          (fun index => challenges index.succ)
          (fun index => inputs index.succ))

/-- The complete assignment fold and the public-only fold agree exactly. -/
theorem projectPublicInput_combine
    {shape : Shape} {count : Nat}
    (challenges : Fin count -> RingF)
    (assignments : Fin count -> Assignment shape) :
    projectPublicInput (PiRLCFinite.combineAssignments challenges assignments) =
      combinePublicInputs challenges
        (fun index => projectPublicInput (assignments index)) := by
  induction count with
  | zero => exact projectPublicInput_zero
  | succ count inductionHypothesis =>
      rw [PiRLCFinite.combineAssignments, combinePublicInputs,
        projectPublicInput_add, projectPublicInput_act]
      rw [inductionHypothesis
        (fun index => challenges index.succ)
        (fun index => assignments index.succ)]

/-- Exact public-input field required by a future concrete
`Folding.PiRLC.Algebra`. -/
theorem relation_publicInput_hom
    {shape : Shape} {Commitment : Type}
    (commit : Assignment shape -> Commitment)
    {count : Nat} (challenges : Fin count -> RingF)
    (assignments : Fin count -> Assignment shape) :
    (relationSemantics commit).projectPublicInput
        (PiRLCFinite.combineAssignments challenges assignments) =
      combinePublicInputs challenges
        (fun index =>
          (relationSemantics commit).projectPublicInput (assignments index)) := by
  exact projectPublicInput_combine challenges assignments

end NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.PublicInput
