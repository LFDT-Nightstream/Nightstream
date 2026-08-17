import Nightstream.Implementation.Nebula.NIFS.Core.PaperAlgebra
import Nightstream.Implementation.R1CS.Correspondence.Gadgets.PiDecStrictSound

/-!
Contract: exact coordinate rows for the V2 product PiDEC recomposition.

Assurance tier: generated-row semantic model.

Owns the fixed four-component, 18-row, 54-coefficient commitment layout; the
14-matrix, 54-coefficient, two-limb evaluation layout; the verifier-owned
binary radix weights; exactly one recomposition row per coordinate; and a
row-soundness theorem stated as independent coordinate equations.

Does not own placement of these columns in a complete recursive artifact,
the typed paper-PiDEC bridge, PiCCS, PiRLC, transcript rows, Rust refinement,
or cryptographic soundness.

Emits constraints: 5,400 linear R1CS rows.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.Nebula.ProductPiDecRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.PiDecStrictCompiler
open Nightstream.Implementation.R1CS.PiDecStrictSound
open Nightstream.Protocol.Nebula.CommitmentBundle
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

abbrev ChildIndex := Fin productionGlobalParams.k
abbrev CommitmentRow := Fin ProductCommitmentAlgebra.Rank
abbrev RingLane := Fin ringDegree
abbrev MatrixIndex := Fin ProductNifsCodec.shape.matrixCount
abbrev CoefficientIndex := Fin ProductNifsCodec.shape.coefficientCount
abbrev ExtensionLimb := Fin 2

/-- Column ownership for one complete four-component public commitment. -/
structure BundleLayout where
  column : Component -> CommitmentRow -> RingLane -> Nat

/-- Column ownership for one complete packed `K` evaluation family. -/
structure EvaluationLayout where
  column : MatrixIndex -> CoefficientIndex -> ExtensionLimb -> Nat

/-- Parent and sixteen child coordinate locations. -/
structure Layout where
  parentBundle : BundleLayout
  childBundle : ChildIndex -> BundleLayout
  parentEvaluation : EvaluationLayout
  childEvaluation : ChildIndex -> EvaluationLayout

/-- Canonical component order. This order is verifier-key-bound. -/
def components : List Component :=
  [.full, .operations, .initialSnapshot, .finalSnapshot]

theorem component_mem (component : Component) : component ∈ components := by
  cases component <;> simp [components]

/-- Canonical typed enumeration of a finite coordinate axis. -/
def indices (count : Nat) : List (Fin count) :=
  List.ofFn fun index => index

theorem index_mem {count : Nat} (index : Fin count) :
    index ∈ indices count := by
  exact List.mem_ofFn.mpr ⟨index, rfl⟩

@[simp] theorem indices_length (count : Nat) :
    (indices count).length = count := by
  simp [indices]

private theorem length_flatMap_uniform
    {Alpha Beta : Type} (items : List Alpha) (values : Alpha -> List Beta)
    (count : Nat) (uniform : ∀ item, (values item).length = count) :
    (items.flatMap values).length = items.length * count := by
  induction items with
  | nil => simp
  | cons head tail inductionHypothesis =>
      simp [uniform, inductionHypothesis, Nat.add_mul, Nat.add_comm]

/-- Exact natural coefficients used by the emitted sparse rows. -/
def radixPowers : List Nat :=
  List.ofFn fun child : ChildIndex => (PiDEC.radixWeight child).val

theorem radixPowers_length : radixPowers.length = 16 := by
  rfl

/-- Every emitted coefficient is a nonzero canonical Goldilocks residue. -/
theorem radixPowers_canonical :
    ∀ coefficient ∈ radixPowers,
      0 < coefficient ∧ coefficient < goldilocksP := by
  decide

def commitmentInstruction
    (layout : Layout) (component : Component)
    (row : CommitmentRow) (lane : RingLane) : Instruction :=
  recompositionCheck
    (layout.parentBundle.column component row lane)
    (List.ofFn fun child : ChildIndex =>
      (layout.childBundle child).column component row lane)
    radixPowers

def componentInstructions
    (layout : Layout) (component : Component) : List Instruction :=
  (indices ProductCommitmentAlgebra.Rank).flatMap fun row =>
    (indices ringDegree).map fun lane =>
      commitmentInstruction layout component row lane

def commitmentInstructions (layout : Layout) : List Instruction :=
  components.flatMap (componentInstructions layout)

def evaluationInstruction
    (layout : Layout) (matrix : MatrixIndex)
    (coefficient : CoefficientIndex) (limb : ExtensionLimb) : Instruction :=
  recompositionCheck
    (layout.parentEvaluation.column matrix coefficient limb)
    (List.ofFn fun child : ChildIndex =>
      (layout.childEvaluation child).column matrix coefficient limb)
    radixPowers

def matrixInstructions
    (layout : Layout) (matrix : MatrixIndex) : List Instruction :=
  (indices ProductNifsCodec.shape.coefficientCount).flatMap fun coefficient =>
    (indices 2).map fun limb =>
      evaluationInstruction layout matrix coefficient limb

def evaluationInstructions (layout : Layout) : List Instruction :=
  (indices ProductNifsCodec.shape.matrixCount).flatMap
    (matrixInstructions layout)

def instructions (layout : Layout) : List Instruction :=
  commitmentInstructions layout ++ evaluationInstructions layout

def rows (layout : Layout) : List Row :=
  CheckedProgram.rows (instructions layout)

theorem componentInstructions_length
    (layout : Layout) (component : Component) :
    (componentInstructions layout component).length = 972 := by
  rw [componentInstructions,
    length_flatMap_uniform _ _ ringDegree]
  · decide
  · intro row
    simp [indices]

theorem commitmentInstructions_length (layout : Layout) :
    (commitmentInstructions layout).length = 3888 := by
  rw [commitmentInstructions,
    length_flatMap_uniform _ _ 972 (componentInstructions_length layout)]
  decide

theorem matrixInstructions_length
    (layout : Layout) (matrix : MatrixIndex) :
    (matrixInstructions layout matrix).length = 108 := by
  rw [matrixInstructions,
    length_flatMap_uniform _ _ 2]
  · decide
  · intro coefficient
    simp [indices]

theorem evaluationInstructions_length (layout : Layout) :
    (evaluationInstructions layout).length = 1512 := by
  rw [evaluationInstructions,
    length_flatMap_uniform _ _ 108 (matrixInstructions_length layout)]
  decide

theorem instructions_length (layout : Layout) :
    (instructions layout).length = 5400 := by
  simp [instructions, commitmentInstructions_length,
    evaluationInstructions_length]

theorem rows_length (layout : Layout) : (rows layout).length = 5400 := by
  simp [rows, CheckedProgram.rows, instructions_length]

theorem commitmentInstruction_mem
    (layout : Layout) (component : Component)
    (row : CommitmentRow) (lane : RingLane) :
    commitmentInstruction layout component row lane ∈
      commitmentInstructions layout := by
  apply List.mem_flatMap.mpr
  refine ⟨component, component_mem component, ?_⟩
  apply List.mem_flatMap.mpr
  refine ⟨row, index_mem row, ?_⟩
  exact List.mem_map.mpr ⟨lane, index_mem lane, rfl⟩

theorem evaluationInstruction_mem
    (layout : Layout) (matrix : MatrixIndex)
    (coefficient : CoefficientIndex) (limb : ExtensionLimb) :
    evaluationInstruction layout matrix coefficient limb ∈
      evaluationInstructions layout := by
  apply List.mem_flatMap.mpr
  refine ⟨matrix, index_mem matrix, ?_⟩
  apply List.mem_flatMap.mpr
  refine ⟨coefficient, index_mem coefficient, ?_⟩
  exact List.mem_map.mpr ⟨limb, index_mem limb, rfl⟩

/-- Independent decoded meaning of all V2 PiDEC coordinate rows. -/
structure Accepted (layout : Layout) (assignment : Nat -> Nat) : Prop where
  commitment : forall component row lane,
    Recomposes assignment
      (layout.parentBundle.column component row lane)
      (List.ofFn fun child : ChildIndex =>
        (layout.childBundle child).column component row lane)
      radixPowers
  evaluation : forall matrix coefficient limb,
    Recomposes assignment
      (layout.parentEvaluation.column matrix coefficient limb)
      (List.ofFn fun child : ChildIndex =>
        (layout.childEvaluation child).column matrix coefficient limb)
      radixPowers

/-- Exact rows imply all independent coordinate equations. -/
theorem rows_sound
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows layout) assignment) :
    Accepted layout assignment := by
  constructor
  · intro component row lane
    apply recompositionCheck_sound canonical one _ _ _ radixPowers_canonical
    apply satisfies _
    apply List.mem_map.mpr
    exact ⟨commitmentInstruction layout component row lane,
      List.mem_append_left _
        (commitmentInstruction_mem layout component row lane), rfl⟩
  · intro matrix coefficient limb
    apply recompositionCheck_sound canonical one _ _ _ radixPowers_canonical
    apply satisfies _
    apply List.mem_map.mpr
    exact ⟨evaluationInstruction layout matrix coefficient limb,
      List.mem_append_right _
        (evaluationInstruction_mem layout matrix coefficient limb), rfl⟩

end Nightstream.Implementation.Nebula.ProductPiDecRows
