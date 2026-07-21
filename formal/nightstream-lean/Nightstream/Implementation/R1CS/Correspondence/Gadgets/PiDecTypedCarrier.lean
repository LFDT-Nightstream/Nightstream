import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain
import Nightstream.Implementation.R1CS.Correspondence.Gadgets.PiDecStrictCompiler
import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.Assignment
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra

/-!
Typed paper boundary for a strict production `PiDEC` source layout.

Protocol: SuperNeo Section 7.5 at `b = 2`, `k = 14`.
Phase: decode one strict parent and fourteen strict child claims into the exact
operational paper verifier.
Constraint family: representation and retained paper equations only; this file
emits no rows.

Assurance tier: model-level representation refinement.

Owns: a layout-parametric, dimension-checked decoder for commitment, all 270
public coordinates, the semantic prefix of every evaluation row, and the
common row point; the four decoded equations sufficient for exact paper
acceptance; and the active `t = 13`, `r = 24` shape specialization.

Does not own: generated row identity, Rust lowering, satisfaction of the
decoded equations, private CE openings, Ajtai binding, transcript authority,
`s_col`, `y_zcol`, fold digests, inactive columns, evaluation padding, costs,
or row removal.

Authority boundary: `Profile` proves that every total list read used by the
decoder is in range. `Accepted` is an independent typed endpoint for a future
row/artifact theorem; this module does not infer it from a measured circuit.
Only fields consumed by the Section-7.5 verifier enter the decoded CE values.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_dec.typed.layout.children` | the source layout has exactly fourteen ordered children | checked shape | `Profile.childCount` |
| `nifs.pi_dec.typed.layout.commitment` | commitment payload is exactly `kappa * 54` fields | checked shape | `Profile.*CommitmentLength` |
| `nifs.pi_dec.typed.layout.public_input` | all claims expose the exact lane-major 270-coordinate public carrier | checked shape | `Profile.*PublicLength`, `decodePublicInput` |
| `nifs.pi_dec.typed.layout.evaluations` | exactly `t` rows expose at least 108 semantic limbs each | checked shape | `Profile.*EvaluationCount`, `Profile.*EvaluationWidth` |
| `nifs.pi_dec.typed.layout.point` | every physical `r` has the relation-owned dimension | checked shape | `Profile.*PointLength`, `decodePoint` |
| `nifs.pi_dec.typed.acceptance` | decoded commitment, canonical public children, evaluations, and points realize paper `PiDEC` | independent/model | `Accepted`, `accepted_refines_paper` |
| `nifs.pi_dec.typed.active` | active relation has `t = 13`, `r = 24`, and 270 public fields | computed profile | `Active.*_exact` |
-/

namespace Nightstream.Implementation.R1CS.PiDecTypedCarrier

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra
open Nightstream.SuperNeo.Folding
open Nightstream.Implementation.R1CS.PiDecStrictCompiler

abbrev Child := PiDECAlgebra.Radix.ChildIndex

/-- Exact host-layout facts needed by the paper decoder. Sidecars are absent
by construction. -/
structure Profile
    (shape : Phi81Relation.Shape)
    (verifierRows : Nat)
    (layout : PiDecStrictCompiler.Layout) : Prop where
  childCount : layout.children.length = productionGlobalParams.k
  radixTwo : layout.radix = productionGlobalParams.b
  ringDimension : layout.ringDimension = ringDegree
  extensionLimbs : layout.extensionLimbs = 2
  activePublicColumns : activeColumns layout = publicRingColumns
  parentCommitmentLength :
    layout.parent.commitment.dataCols.length = verifierRows * ringDegree
  childCommitmentLength : forall child, child ∈ layout.children ->
    child.commitment.dataCols.length = verifierRows * ringDegree
  publicWidth : shape.publicWidth = alignedPublicWidth
  parentPublicLength :
    layout.parent.xActiveCols.length = alignedPublicWidth
  childPublicLength : forall child, child ∈ layout.children ->
    child.xActiveCols.length = alignedPublicWidth
  parentEvaluationCount :
    layout.parent.yRingCols.length = shape.matrixCount
  childEvaluationCount : forall child, child ∈ layout.children ->
    child.yRingCols.length = shape.matrixCount
  parentEvaluationWidth : forall row,
    row < layout.parent.yRingCols.length ->
      2 * ringDegree <= (layout.parent.yRingCols.getD row []).length
  childEvaluationWidth : forall child, child ∈ layout.children ->
    forall row, row < child.yRingCols.length ->
      2 * ringDegree <= (child.yRingCols.getD row []).length
  parentPointLength : layout.parent.rCols.length = shape.rowVariables
  childPointLength : forall child, child ∈ layout.children ->
    child.rCols.length = shape.rowVariables

namespace Profile

/-- Ordered child lookup justified by the fixed paper arity. -/
def childLayout
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    {layout : PiDecStrictCompiler.Layout}
    (profile : Profile shape verifierRows layout)
    (child : Child) : ClaimLayout :=
  layout.children.get ⟨child.val, by
    rw [profile.childCount]
    exact child.isLt⟩

theorem childLayout_mem
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    {layout : PiDecStrictCompiler.Layout}
    (profile : Profile shape verifierRows layout)
    (child : Child) : profile.childLayout child ∈ layout.children := by
  unfold childLayout
  exact List.get_mem _ _

end Profile

/-- Canonical interpretation of one R1CS field representative. -/
def decodeField (assignment : Nat -> Nat) (column : Nat) : F :=
  PiDECAlgebra.Radix.fieldOfNat (assignment column)

/-- Decode one interleaved `(c0,c1)` extension-field pair. -/
def decodeK (assignment : Nat -> Nat) (columns : Nat × Nat) : K :=
  ⟨decodeField assignment columns.1, decodeField assignment columns.2⟩

/-- Decode one flat row-major family of Phi81 commitment rings. -/
def decodeCommitment
    {verifierRows : Nat}
    (assignment : Nat -> Nat)
    (claim : ClaimLayout) : PiRLCAlgebra.Commitment.Value verifierRows :=
  fun row coefficient =>
    decodeField assignment (claim.commitment.dataCols.getD
      (row.val * ringDegree + coefficient.val) 0)

/-- Production stores public `X` lane-major: `(lane, block)`. The typed paper
carrier is logical block-major, so this is the sole transpose. -/
def publicSlot
    {shape : Phi81Relation.Shape}
    (column : Fin shape.publicWidth) : Nat :=
  (column.val % ringDegree) * publicRingColumns + column.val / ringDegree

theorem publicSlot_lt
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    {layout : PiDecStrictCompiler.Layout}
    (profile : Profile shape verifierRows layout)
    (column : Fin shape.publicWidth) :
    publicSlot column < alignedPublicWidth := by
  have columnLt : column.val < alignedPublicWidth := by
    rw [← profile.publicWidth]
    exact column.isLt
  have laneLt := Nat.mod_lt column.val (by decide : 0 < ringDegree)
  have blockLt : column.val / ringDegree < publicRingColumns := by
    simp only [alignedPublicWidth, ringDegree, publicRingColumns] at columnLt ⊢
    omega
  simp only [publicSlot, alignedPublicWidth, ringDegree, publicRingColumns]
    at laneLt blockLt ⊢
  omega

/-- Typed form of the production slot, once the profile has established the
exact 270-coordinate carrier. -/
def publicSlotFin
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    {layout : PiDecStrictCompiler.Layout}
    (profile : Profile shape verifierRows layout)
    (column : Fin shape.publicWidth) : Fin alignedPublicWidth :=
  ⟨publicSlot column, publicSlot_lt profile column⟩

/-- Inverse transpose from one lane-major production slot to one typed
block-major coordinate. -/
def logicalColumn
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    {layout : PiDecStrictCompiler.Layout}
    (profile : Profile shape verifierRows layout)
    (slot : Fin alignedPublicWidth) : Fin shape.publicWidth :=
  ⟨(slot.val % publicRingColumns) * ringDegree +
      slot.val / publicRingColumns, by
    rw [profile.publicWidth]
    have slotLt := slot.isLt
    have blockLt := Nat.mod_lt slot.val
      (by decide : 0 < publicRingColumns)
    have laneLt : slot.val / publicRingColumns < ringDegree := by
      simp only [alignedPublicWidth, ringDegree, publicRingColumns] at slotLt ⊢
      omega
    simp only [alignedPublicWidth, ringDegree, publicRingColumns]
      at blockLt laneLt ⊢
    omega⟩

@[simp] theorem logicalColumn_publicSlotFin
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    {layout : PiDecStrictCompiler.Layout}
    (profile : Profile shape verifierRows layout)
    (column : Fin shape.publicWidth) :
    logicalColumn profile (publicSlotFin profile column) = column := by
  apply Fin.ext
  have columnLt : column.val < alignedPublicWidth := by
    rw [← profile.publicWidth]
    exact column.isLt
  simp only [logicalColumn, publicSlotFin, publicSlot, ringDegree,
    publicRingColumns, alignedPublicWidth] at columnLt ⊢
  omega

@[simp] theorem publicSlotFin_logicalColumn
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    {layout : PiDecStrictCompiler.Layout}
    (profile : Profile shape verifierRows layout)
    (slot : Fin alignedPublicWidth) :
    publicSlotFin profile (logicalColumn profile slot) = slot := by
  apply Fin.ext
  have slotLt := slot.isLt
  simp only [publicSlotFin, publicSlot, logicalColumn, ringDegree,
    publicRingColumns, alignedPublicWidth] at slotLt ⊢
  omega

/-- The transpose observes every physical public slot exactly once. -/
theorem publicTranspose_exact
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    {layout : PiDecStrictCompiler.Layout}
    (profile : Profile shape verifierRows layout) :
    Function.Injective (publicSlotFin profile) /\
      forall slot, exists column, publicSlotFin profile column = slot := by
  constructor
  · intro left right equal
    have inverse := congrArg (logicalColumn profile) equal
    simpa using inverse
  · intro slot
    exact ⟨logicalColumn profile slot, publicSlotFin_logicalColumn profile slot⟩

theorem parentPublicSlot_lt
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    {layout : PiDecStrictCompiler.Layout}
    (profile : Profile shape verifierRows layout)
    (column : Fin shape.publicWidth) :
    publicSlot column < layout.parent.xActiveCols.length := by
  rw [profile.parentPublicLength]
  exact publicSlot_lt profile column

theorem childPublicSlot_lt
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    {layout : PiDecStrictCompiler.Layout}
    (profile : Profile shape verifierRows layout)
    (child : Child)
    (column : Fin shape.publicWidth) :
    publicSlot column < (profile.childLayout child).xActiveCols.length := by
  rw [profile.childPublicLength (profile.childLayout child)
    (profile.childLayout_mem child)]
  exact publicSlot_lt profile column

/-- Decode all 270 public coordinates. `Profile.publicWidth` and the per-claim
length facts are the separate in-range certificate for this total read. -/
def decodePublicInput
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    {layout : PiDecStrictCompiler.Layout}
    (_profile : Profile shape verifierRows layout)
    (assignment : Nat -> Nat)
    (claim : ClaimLayout) : Phi81Relation.PublicInput shape :=
  fun column =>
    decodeField assignment (claim.xActiveCols.getD
      (publicSlot column) 0)

/-- Decode only the 108 semantic limbs of one evaluation row. Any remaining
physical limbs are padding and never enter the paper CE value. -/
def decodeEvaluation
    (assignment : Nat -> Nat) (row : List Nat) : Phi81Relation.Evaluation :=
  fun coefficient => ⟨
    decodeField assignment (row.getD (2 * coefficient.val) 0),
    decodeField assignment (row.getD (2 * coefficient.val + 1) 0)⟩

/-- Preserve matrix order and no other row family. -/
def decodeEvaluations
    (assignment : Nat -> Nat) (claim : ClaimLayout) :
    Array Phi81Relation.Evaluation :=
  (claim.yRingCols.map (decodeEvaluation assignment)).toArray

theorem parentEvaluationLimb_lt
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    {layout : PiDecStrictCompiler.Layout}
    (profile : Profile shape verifierRows layout)
    (matrix : Fin shape.matrixCount)
    (coefficient : Fin ringDegree)
    (limb : Fin 2) :
    2 * coefficient.val + limb.val <
      (layout.parent.yRingCols.getD matrix.val []).length := by
  have matrixLt : matrix.val < layout.parent.yRingCols.length := by
    rw [profile.parentEvaluationCount]
    exact matrix.isLt
  have width := profile.parentEvaluationWidth matrix.val matrixLt
  have coefficientLt := coefficient.isLt
  have limbLt := limb.isLt
  simp only [ringDegree] at coefficientLt width ⊢
  omega

theorem childEvaluationLimb_lt
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    {layout : PiDecStrictCompiler.Layout}
    (profile : Profile shape verifierRows layout)
    (child : Child)
    (matrix : Fin shape.matrixCount)
    (coefficient : Fin ringDegree)
    (limb : Fin 2) :
    2 * coefficient.val + limb.val <
      ((profile.childLayout child).yRingCols.getD matrix.val []).length := by
  have childMember := profile.childLayout_mem child
  have matrixLt : matrix.val <
      (profile.childLayout child).yRingCols.length := by
    rw [profile.childEvaluationCount (profile.childLayout child) childMember]
    exact matrix.isLt
  have width := profile.childEvaluationWidth
    (profile.childLayout child) childMember matrix.val matrixLt
  have coefficientLt := coefficient.isLt
  have limbLt := limb.isLt
  simp only [ringDegree] at coefficientLt width ⊢
  omega

/-- Dimension-checked decoding of the paper row point. -/
def decodePoint
    {shape : Phi81Relation.Shape}
    (assignment : Nat -> Nat)
    (claim : ClaimLayout)
    (dimension : claim.rCols.length = shape.rowVariables) :
    Phi81Relation.Point shape where
  coordinates := claim.rCols.map (decodeK assignment)
  dimension := by simpa using dimension

/-- Typed combined parent decoded from exactly the paper fields. -/
def decodedParent
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    {layout : PiDecStrictCompiler.Layout}
    (profile : Profile shape verifierRows layout)
    (system : Phi81Relation.Structure shape)
    (assignment : Nat -> Nat) :
    CE.Instance
      (Phi81Relation.Structure shape)
      (Phi81Relation.PublicInput shape)
      (Phi81Relation.Point shape)
      Phi81Relation.Evaluation
      (PiRLCAlgebra.Commitment.Value verifierRows) where
  constraintSystem := system
  commitment := decodeCommitment assignment layout.parent
  publicInput := decodePublicInput profile assignment layout.parent
  point := decodePoint assignment layout.parent profile.parentPointLength
  evaluations := decodeEvaluations assignment layout.parent
  stage := .combined

/-- Typed child claims decoded in the physical source order. Structure and
stage are verifier-computed; commitment, public input, evaluations, and `r`
are decoded from their explicit strict carriers. -/
def decodedOutput
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    {layout : PiDecStrictCompiler.Layout}
    (profile : Profile shape verifierRows layout)
    (system : Phi81Relation.Structure shape)
    (assignment : Nat -> Nat) :
    Child ->
      CE.Instance
        (Phi81Relation.Structure shape)
        (Phi81Relation.PublicInput shape)
        (Phi81Relation.Point shape)
        Phi81Relation.Evaluation
        (PiRLCAlgebra.Commitment.Value verifierRows) :=
  fun child =>
    let claim := profile.childLayout child
    {
      constraintSystem := system
      commitment := decodeCommitment assignment claim
      publicInput := decodePublicInput profile assignment claim
      point := decodePoint assignment claim
        (profile.childPointLength claim (profile.childLayout_mem child))
      evaluations := decodeEvaluations assignment claim
      stage := .fresh
    }

@[simp] theorem decodedParent_evaluations_size
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    {layout : PiDecStrictCompiler.Layout}
    (profile : Profile shape verifierRows layout)
    (system : Phi81Relation.Structure shape)
    (assignment : Nat -> Nat) :
    (decodedParent profile system assignment).evaluations.size =
      shape.matrixCount := by
  simpa [decodedParent, decodeEvaluations] using
    profile.parentEvaluationCount

@[simp] theorem decodedOutput_evaluations_size
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    {layout : PiDecStrictCompiler.Layout}
    (profile : Profile shape verifierRows layout)
    (system : Phi81Relation.Structure shape)
    (assignment : Nat -> Nat)
    (child : Child) :
    (decodedOutput profile system assignment child).evaluations.size =
      shape.matrixCount := by
  simpa [decodedOutput, decodeEvaluations] using
    profile.childEvaluationCount (profile.childLayout child)
      (profile.childLayout_mem child)

/-- Exact decoded endpoint expected from the retained strict source families.
A row/artifact refinement must prove this predicate; it is not a restatement
of row satisfaction. -/
structure Accepted
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    {layout : PiDecStrictCompiler.Layout}
    (profile : Profile shape verifierRows layout)
    (key : PiRLCAlgebra.Commitment.Key shape verifierRows)
    (system : Phi81Relation.Structure shape)
    (assignment : Nat -> Nat) : Prop where
  commitmentEquation :
    (decodedParent profile system assignment).commitment =
      (PiDECAlgebra.Algebra.concrete key).recomposeCommitment
        (fun child =>
          (decodedOutput profile system assignment child).commitment)
  canonicalPublicInput : forall child,
    (decodedOutput profile system assignment child).publicInput =
      (PiDECAlgebra.PaperVerifier.publicInputSplit key).split
        (decodedParent profile system assignment).publicInput child
  evaluationEquation :
    (decodedParent profile system assignment).evaluations =
      (PiDECAlgebra.Algebra.concrete key).recomposeEvaluations
        (fun child =>
          (decodedOutput profile system assignment child).evaluations)
  commonPoint : forall child,
    (decodedOutput profile system assignment child).point =
      (decodedParent profile system assignment).point

/-- The typed source endpoint is sufficient for exact operational paper
acceptance. No sidecar field appears in either the premise or conclusion. -/
theorem accepted_refines_paper
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    {layout : PiDecStrictCompiler.Layout}
    (profile : Profile shape verifierRows layout)
    (key : PiRLCAlgebra.Commitment.Key shape verifierRows)
    (system : Phi81Relation.Structure shape)
    (assignment : Nat -> Nat)
    (accepted : Accepted profile key system assignment) :
    PiDEC.PaperVerifier.OutputAccepted
      (PiDECAlgebra.Algebra.concrete key)
      (PiDECAlgebra.PaperVerifier.publicInputSplit key)
      (PiDECAlgebra.PaperVerifier.evaluationArity key)
      (decodedParent profile system assignment)
      (decodedOutput profile system assignment) := by
  refine {
    outputComputed := ?_
    checks := {
      parentCombined := rfl
      parentEvaluationSize := decodedParent_evaluations_size
        profile system assignment
      messageEvaluationSize := decodedOutput_evaluations_size
        profile system assignment
      commitmentEquation := accepted.commitmentEquation
      evaluationEquation := accepted.evaluationEquation
    }
  }
  funext child
  have publicInput := accepted.canonicalPublicInput child
  have point := accepted.commonPoint child
  simp only [decodedParent, decodedOutput] at publicInput point
  simp only [PiDEC.PaperVerifier.children,
    PiDEC.PaperVerifier.attemptForOutput,
    PiDEC.PaperVerifier.messagesOf, decodedParent, decodedOutput]
  rw [publicInput, point]

namespace Active

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc

/-- The five-ring public prefix fits the independently owned active relation
carrier. -/
def publicFits :
    ringDegree * publicRingColumns <=
      ProductionDomain.semanticShape.carrierWidth := by
  rw [ProductionDomain.semanticShape_carrierWidth]
  decide

/-- Exact active relation shape, independent of any `PiDEC` row artifact. -/
def shape : Phi81Relation.Shape :=
  Phi81Relation.Shape.ofSemantic ProductionDomain.semanticShape
    publicRingColumns publicFits

@[simp] theorem rowVariables_exact : shape.rowVariables = 24 := by
  rfl

@[simp] theorem matrixCount_exact : shape.matrixCount = 13 := by
  rfl

@[simp] theorem publicWidth_exact : shape.publicWidth = 270 := by
  rfl

/-- Active specialization of the generic strict carrier profile. A concrete
layout inhabitant remains the responsibility of the generated artifact. -/
abbrev ProfileFor (verifierRows : Nat)
    (layout : PiDecStrictCompiler.Layout) : Prop :=
  PiDecTypedCarrier.Profile shape verifierRows layout

theorem parent_evaluation_count_exact
    {verifierRows : Nat}
    {layout : PiDecStrictCompiler.Layout}
    (profile : ProfileFor verifierRows layout) :
    layout.parent.yRingCols.length = 13 := by
  simpa using profile.parentEvaluationCount

theorem child_evaluation_count_exact
    {verifierRows : Nat}
    {layout : PiDecStrictCompiler.Layout}
    (profile : ProfileFor verifierRows layout)
    (child : Child) :
    (profile.childLayout child).yRingCols.length = 13 := by
  simpa using profile.childEvaluationCount (profile.childLayout child)
    (profile.childLayout_mem child)

theorem parent_point_count_exact
    {verifierRows : Nat}
    {layout : PiDecStrictCompiler.Layout}
    (profile : ProfileFor verifierRows layout) :
    layout.parent.rCols.length = 24 := by
  simpa using profile.parentPointLength

theorem child_point_count_exact
    {verifierRows : Nat}
    {layout : PiDecStrictCompiler.Layout}
    (profile : ProfileFor verifierRows layout)
    (child : Child) :
    (profile.childLayout child).rCols.length = 24 := by
  simpa using profile.childPointLength (profile.childLayout child)
    (profile.childLayout_mem child)

end Active

end Nightstream.Implementation.R1CS.PiDecTypedCarrier
