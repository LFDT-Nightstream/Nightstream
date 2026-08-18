import Nightstream.Implementation.Nebula.NIFS.PiRLC.AlgebraRows
import Nightstream.Implementation.Nebula.NIFS.PiRLC.AlgebraSound
import Nightstream.Implementation.Nebula.NIFS.Core.PaperAlgebraFor
import Nightstream.Implementation.Nebula.NIFS.Core.Poseidon2
import Nightstream.Implementation.R1CS.Correspondence.Projection.Phi81.Carrier

/-!
Contract: bounded-family continuation semantics for the production PiRLC
algebra.

Assurance tier: model-level exact refinement, local-row soundness, and work
geometry.

Owns a 110-step family schedule. Each step computes one complete output ring
from all seventeen PiRLC sources and absorbs its 54 canonical coefficients into
one Poseidon2 continuation state. It proves exact equality with the current
monolithic typed PiRLC output and the exact local schoolbook work census.

Does not own generated rows, authority for replayed phase inputs, a recursive
state codec, Rust conformance, Poseidon2 collision resistance, terminal
lifecycle integration, or a final generated relation size claim.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiRlc

open Nightstream.Implementation.Nebula.ProductPiRlcAlgebraRows
open Nightstream.Implementation.Nebula.ProductPiRlcRingCombinationRows
open Nightstream.Implementation.Nebula.ProductPiRlcRingCombinationSound
open Nightstream.Implementation.Nebula.ProductNifsCodec
open Nightstream.Implementation.Nebula.ProductPiRlcAlgebraSound
open Nightstream.Implementation.Nebula.ProductPiRlcRingCombinationRows
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Protocol.Nebula.CommitmentBundle
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

abbrev Family := ProductPiRlcAlgebraRows.Family
abbrev Source := ProductPiRlcRingCombinationRows.Source
abbrev BindingState := ProductPoseidon2.State

/-- All ring-valued inputs for one PiRLC source after PiCCS. -/
abbrev SourceRings := Family → RingF

/-- All seventeen source-ring families. -/
abbrev InputRings := Source → SourceRings

/-- Exact value computed by one narrow family phase. -/
def combineOne
    (challenges : Source → RingF) (inputs : Source → RingF) : RingF :=
  ProjectionPhi81.productSum challenges inputs

def familyOutput
    (challenges : Source → RingF) (inputs : InputRings)
    (family : Family) : RingF :=
  combineOne challenges (fun source => inputs source family)

/-- Verifier-owned family order: 72 commitment rings, ten public rings, then
28 evaluation-limb rings. -/
def familySchedule : List Family := families

@[simp] theorem familySchedule_length : familySchedule.length = 110 := by
  exact families_length

theorem familySchedule_covers (family : Family) :
    family ∈ familySchedule := by
  exact family_mem family

/-- The verifier schedule computes every output family exactly once. -/
theorem familySchedule_nodup : familySchedule.Nodup := by
  decide

/-- Verifier-owned PiCCS source order: one fresh source followed by the
sixteen running sources of the selected Nightstream profile. -/
def sourceSchedule : List Source := canonicalFinIndices sourceCount

@[simp] theorem sourceSchedule_length : sourceSchedule.length = 17 := by
  rw [sourceSchedule, canonicalFinIndices_length]
  exact sourceCount_eq

theorem sourceSchedule_covers (source : Source) :
    source ∈ sourceSchedule := by
  simp [sourceSchedule, canonicalFinIndices]

theorem sourceSchedule_nodup : sourceSchedule.Nodup := by
  exact canonicalFinIndices_nodup sourceCount

/-- Canonical coefficients emitted by one family phase. The family position
is verifier-owned by `familySchedule` and is not prover advice. -/
def familyOutputFields
    (challenges : Source → RingF) (inputs : InputRings)
    (family : Family) : List Nat :=
  ProductPoseidon2.ringFFields (familyOutput challenges inputs family)

@[simp] theorem familyOutputFields_length
    (challenges : Source → RingF) (inputs : InputRings)
    (family : Family) :
    (familyOutputFields challenges inputs family).length = 54 := by
  simp [familyOutputFields, ProductPoseidon2.ringFFields,
    ProductPoseidon2.finFields, ProductPoseidon2.fFields,
    canonicalFinIndices_length, ringDegree]

/-- Canonical coefficients of one PiCCS-derived input ring. -/
def sourceInputFields
    (inputs : InputRings) (family : Family) (source : Source) : List Nat :=
  ProductPoseidon2.ringFFields (inputs source family)

@[simp] theorem sourceInputFields_length
    (inputs : InputRings) (family : Family) (source : Source) :
    (sourceInputFields inputs family source).length = 54 := by
  simp [sourceInputFields, ProductPoseidon2.ringFFields,
    ProductPoseidon2.finFields, ProductPoseidon2.fFields,
    canonicalFinIndices_length, ringDegree]

/-- All seventeen PiCCS-derived rings used by one family phase, in canonical
source order. The same list is the phase replay frame and algebra input. -/
def familyInputFrame (inputs : InputRings) (family : Family) : List Nat :=
  sourceSchedule.flatMap (sourceInputFields inputs family)

/-- Complete PiCCS-derived PiRLC input frame. Families are outermost, sources
are next, and ring coefficients are innermost. -/
def inputFrame (inputs : InputRings) : List Nat :=
  familySchedule.flatMap (familyInputFrame inputs)

/-- Complete canonical PiRLC output stream. This list is a semantic
definition. The recursive circuit does not carry it between phases. -/
def outputFrame
    (challenges : Source → RingF) (inputs : InputRings) : List Nat :=
  familySchedule.flatMap (familyOutputFields challenges inputs)

private theorem length_flatMap_uniform
    {Alpha Beta : Type} (items : List Alpha) (values : Alpha → List Beta)
    (count : Nat) (uniform : ∀ item, (values item).length = count) :
    (items.flatMap values).length = items.length * count := by
  induction items with
  | nil => simp
  | cons head tail inductionHypothesis =>
      simp [uniform, inductionHypothesis, Nat.add_mul, Nat.add_comm]

@[simp] theorem familyInputFrame_length
    (inputs : InputRings) (family : Family) :
    (familyInputFrame inputs family).length = 918 := by
  unfold familyInputFrame
  rw [length_flatMap_uniform _ _ 54
    (sourceInputFields_length inputs family)]
  simp

@[simp] theorem inputFrame_length (inputs : InputRings) :
    (inputFrame inputs).length = 100980 := by
  unfold inputFrame
  rw [length_flatMap_uniform _ _ 918 (familyInputFrame_length inputs)]
  simp

@[simp] theorem outputFrame_length
    (challenges : Source → RingF) (inputs : InputRings) :
    (outputFrame challenges inputs).length = 5940 := by
  unfold outputFrame
  rw [length_flatMap_uniform _ _ 54
    (familyOutputFields_length challenges inputs)]
  simp [familySchedule_length]

/-! ## Compact continuation machine -/

/-- Persistent semantic state between family phases. No output-ring array is
carried. The verifier-owned family schedule is also not stored here. -/
structure State where
  binding : BindingState
  familyCursor : Nat

def initialState (binding : BindingState) : State where
  binding := binding
  familyCursor := 0

/-- One phase computes one complete family and immediately absorbs it into
the carried Poseidon2 state. -/
@[irreducible] noncomputable def absorbFamily
    (challenges : Source → RingF) (inputs : InputRings)
    (binding : BindingState) (family : Family) : BindingState :=
  Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
    (familyOutputFields challenges inputs family) binding

noncomputable def step
    (challenges : Source → RingF) (inputs : InputRings)
    (state : State) (family : Family) : State where
  binding := absorbFamily challenges inputs state.binding family
  familyCursor := state.familyCursor + 1

@[simp] theorem step_binding
    (challenges : Source → RingF) (inputs : InputRings)
    (state : State) (family : Family) :
    (step challenges inputs state family).binding =
      absorbFamily challenges inputs state.binding family := by
  rfl

noncomputable def run
    (challenges : Source → RingF) (inputs : InputRings) :
    List Family → State → State
  | [], state => state
  | family :: rest, state =>
      run challenges inputs rest (step challenges inputs state family)

theorem run_cursor
    (challenges : Source → RingF) (inputs : InputRings)
    (program : List Family) (state : State) :
    (run challenges inputs program state).familyCursor =
      state.familyCursor + program.length := by
  induction program generalizing state with
  | nil => simp [run]
  | cons family rest inductionHypothesis =>
      rw [run, inductionHypothesis]
      simp only [step, List.length_cons]
      omega

private theorem step_binding_normalized
    (challenges : Source → RingF) (inputs : InputRings)
    (state : State) (family : Family) :
    (step challenges inputs state family).binding.absorbed <
      Poseidon2Sponge.rate := by
  rw [step_binding]
  unfold absorbFamily
  unfold Poseidon2Duplex.absorbSlice
  exact Poseidon2Duplex.guarded_absorbed_lt ProductPoseidon2.constants _

/-- Repeated family absorption is exactly one absorption of the concatenated
output frame. This is an algebraic equality and uses no hash assumption. -/
theorem run_binding
    (challenges : Source → RingF) (inputs : InputRings)
    (program : List Family) (state : State)
    (normalized : state.binding.absorbed < Poseidon2Sponge.rate) :
    (run challenges inputs program state).binding =
      Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
        (program.flatMap (familyOutputFields challenges inputs))
        state.binding := by
  induction program generalizing state with
  | nil =>
      simp only [run, List.flatMap_nil, Poseidon2Duplex.absorbSlice,
        Poseidon2Duplex.absorbList]
      unfold Poseidon2Duplex.guarded
      rw [if_neg (by omega)]
  | cons family rest inductionHypothesis =>
      rw [run]
      rw [inductionHypothesis _
        (step_binding_normalized challenges inputs state family)]
      simp only [List.flatMap_cons]
      rw [Poseidon2Duplex.absorbSlice_append]
      rw [step_binding]
      unfold absorbFamily
      rfl

theorem run_familySchedule_cursor
    (challenges : Source → RingF) (inputs : InputRings)
    (binding : BindingState) :
    (run challenges inputs familySchedule
      (initialState binding)).familyCursor = 110 := by
  rw [run_cursor]
  simp [initialState]

theorem run_familySchedule_binding
    (challenges : Source → RingF) (inputs : InputRings)
    (binding : BindingState)
    (normalized : binding.absorbed < Poseidon2Sponge.rate) :
    (run challenges inputs familySchedule (initialState binding)).binding =
      Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
        (outputFrame challenges inputs) binding := by
  simpa [outputFrame, initialState] using
    run_binding challenges inputs familySchedule (initialState binding)
      normalized

/-- Canonical field serialization of one Poseidon2 duplex state. -/
def bindingFields (state : BindingState) : List Nat :=
  List.ofFn state.lanes ++ [state.absorbed]

@[simp] theorem bindingFields_length (state : BindingState) :
    (bindingFields state).length = 9 := by
  simp [bindingFields, Poseidon2Core.width]

/-- Exact semantic continuation width: one eight-lane Poseidon2 state, its
absorb cursor, and the verifier-checked family cursor. This is not yet a
generated-relation column count. -/
def persistentFields (state : State) : List Nat :=
  bindingFields state.binding ++ [state.familyCursor]

@[simp] theorem persistentFields_length (state : State) :
    (persistentFields state).length = 10 := by
  simp [persistentFields]

/-- Named collision boundary for later output replay. A digest is checked
compression, not independent authority. -/
def OutputReplayCollision
    (prior : BindingState) (authoritative supplied : List Nat) : Prop :=
  supplied ≠ authoritative ∧
    Poseidon2Duplex.absorbSlice ProductPoseidon2.constants supplied prior =
      Poseidon2Duplex.absorbSlice ProductPoseidon2.constants authoritative prior

theorem replay_eq_authoritative_or_collision
    (prior : BindingState) (authoritative supplied : List Nat)
    (bindingEqual :
      Poseidon2Duplex.absorbSlice ProductPoseidon2.constants supplied prior =
        Poseidon2Duplex.absorbSlice ProductPoseidon2.constants authoritative prior) :
    supplied = authoritative ∨
      OutputReplayCollision prior authoritative supplied := by
  by_cases exact : supplied = authoritative
  · exact Or.inl exact
  · exact Or.inr ⟨exact, bindingEqual⟩

/-! ## Exact local-row soundness -/

/-- The existing 49,626-row single-family relation proves exactly the value
computed by one narrow phase. This theorem does not yet connect generated
selective-CCS artifact rows to this handwritten row family. -/
theorem local_rows_imply_combineOne
    {layout : ProductPiRlcRingCombinationRows.Layout}
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (range : ∀ source lane,
      assignment (layout.challengeSymbol source lane) < 5)
    (satisfied : Satisfies
      (ProductPiRlcRingCombinationRows.rows layout) assignment) :
    outputRing layout assignment canonical =
      combineOne (challengeRing layout assignment range)
        (inputRing layout assignment canonical) := by
  exact rows_imply_ring_combination canonical one range satisfied

/-! ## Exact typed PiRLC output -/

/-- Complete typed inputs after PiCCS and before PiRLC at one selected
augmented-relation exponent. -/
structure TypedInputs
    (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) where
  commitments : Source → ProductCommitmentAlgebra.BundleValue
  publicInputs : Source →
    ProductPaperAlgebraFor.PublicInput rowVariables logicalWidth publicFits
  evaluations : Source → ProductPaperAlgebraFor.Evaluation rowVariables

def evaluationLimb (value : RingK) (limb : Fin 2) : RingF :=
  if limb.val = 0 then RingKModule.component0 value
  else RingKModule.component1 value

/-- Coordinate-family view consumed by the narrow family machine. -/
def typedInputRings
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (inputs : TypedInputs rowVariables logicalWidth publicFits) : InputRings :=
  fun source family =>
    match family with
    | .commitment component row => inputs.commitments source component row
    | .publicInput block =>
        PiRLCAlgebra.PublicInput.publicBlock
          (inputs.publicInputs source) block
    | .evaluation matrix limb =>
        evaluationLimb (inputs.evaluations source matrix) limb

def outputBundle
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (challenges : Source → RingF)
    (inputs : TypedInputs rowVariables logicalWidth publicFits) :
    ProductCommitmentAlgebra.BundleValue :=
  fun component row =>
    familyOutput challenges (typedInputRings inputs) (.commitment component row)

def outputPublic
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (challenges : Source → RingF)
    (inputs : TypedInputs rowVariables logicalWidth publicFits) :
    ProductPaperAlgebraFor.PublicInput rowVariables logicalWidth publicFits :=
  fun column =>
    familyOutput challenges (typedInputRings inputs)
      (.publicInput
        (PiRLCAlgebra.PublicInput.publicBlockIndex
          (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth
            publicFits) column))
      (PiRLCAlgebra.PublicInput.publicLaneIndex column)

def outputEvaluation
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (challenges : Source → RingF)
    (inputs : TypedInputs rowVariables logicalWidth publicFits) :
    ProductPaperAlgebraFor.Evaluation rowVariables :=
  fun matrix lane =>
    ⟨familyOutput challenges (typedInputRings inputs)
        (.evaluation matrix 0) lane,
      familyOutput challenges (typedInputRings inputs)
        (.evaluation matrix 1) lane⟩

theorem outputBundle_eq_combineBundles
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (challenges : Source → RingF)
    (inputs : TypedInputs rowVariables logicalWidth publicFits) :
    outputBundle challenges inputs =
      ProductCommitmentAlgebra.combineBundles challenges inputs.commitments := by
  funext component row lane
  unfold outputBundle ProductCommitmentAlgebra.combineBundles familyOutput
    combineOne
  rw [ProductPiRlcAlgebraSound.combineCommitments_eq_productSum]
  rfl

theorem outputPublic_eq_combinePublicInputs
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (challenges : Source → RingF)
    (inputs : TypedInputs rowVariables logicalWidth publicFits) :
    outputPublic challenges inputs =
      PiRLCAlgebra.PublicInput.combinePublicInputs challenges
        inputs.publicInputs := by
  funext column
  let block := PiRLCAlgebra.PublicInput.publicBlockIndex
    (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits)
    column
  let lane := PiRLCAlgebra.PublicInput.publicLaneIndex column
  have combined :=
    ProductPiRlcAlgebraSound.combinePublicInputs_coordinate challenges
      inputs.publicInputs column
  change
    familyOutput challenges (typedInputRings inputs) (.publicInput block) lane = _
  simp [familyOutput, combineOne, typedInputRings] at *
  exact combined.symm

private theorem k_eq_of_components
    {left right : K} (low : left.c0 = right.c0)
    (high : left.c1 = right.c1) : left = right := by
  cases left
  cases right
  simp_all

theorem outputEvaluation_eq_combineEvaluationFamily
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (challenges : Source → RingF)
    (inputs : TypedInputs rowVariables logicalWidth publicFits) :
    outputEvaluation challenges inputs =
      ProductPaperAlgebraFor.combineEvaluationFamily challenges
        inputs.evaluations := by
  funext matrix lane
  apply k_eq_of_components
  · have combined := congrFun
      (ProductPiRlcAlgebraSound.combineEvaluation_component0 challenges
        (fun source => inputs.evaluations source matrix)) lane
    simpa [outputEvaluation, familyOutput, combineOne, typedInputRings,
      evaluationLimb] using combined.symm
  · have combined := congrFun
      (ProductPiRlcAlgebraSound.combineEvaluation_component1 challenges
        (fun source => inputs.evaluations source matrix)) lane
    simpa [outputEvaluation, familyOutput, combineOne, typedInputRings,
      evaluationLimb] using combined.symm

/-- The 110 narrow family phases reconstruct all three exact typed fields of
the current PiRLC algebra. -/
theorem typedOutput_exact
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (challenges : Source → RingF)
    (inputs : TypedInputs rowVariables logicalWidth publicFits) :
    outputBundle challenges inputs =
        ProductCommitmentAlgebra.combineBundles challenges inputs.commitments ∧
      outputPublic challenges inputs =
        PiRLCAlgebra.PublicInput.combinePublicInputs challenges
          inputs.publicInputs ∧
      outputEvaluation challenges inputs =
        ProductPaperAlgebraFor.combineEvaluationFamily challenges
          inputs.evaluations := by
  exact
    ⟨outputBundle_eq_combineBundles challenges inputs,
      outputPublic_eq_combinePublicInputs challenges inputs,
      outputEvaluation_eq_combineEvaluationFamily challenges inputs⟩

/-! ## Exact bounded-work geometry -/

def familyCount : Nat := familySchedule.length

theorem familyCount_eq : familyCount = 110 := by
  exact familySchedule_length

/-- One phase owns the current 17 challenge rings, 17 source rings, and one
output ring. These are semantic fields, not generated column aliases. -/
def perFamilyVisibleFieldCount : Nat := (17 + 17 + 1) * ringDegree

theorem perFamilyVisibleFieldCount_eq :
    perFamilyVisibleFieldCount = 1890 := by
  decide

/-- One phase uses one schoolbook product coordinate for each source and
pair of ring lanes. -/
def perFamilyAuxiliaryColumnCount : Nat :=
  ProductPiRlcRingCombinationRows.auxiliaryCount

theorem perFamilyAuxiliaryColumnCount_eq :
    perFamilyAuxiliaryColumnCount = 49572 := by
  exact ProductPiRlcRingCombinationRows.productCount_eq

/-- Exact handwritten R1CS row count for one narrow family phase. -/
def perFamilyArithmeticRowCount : Nat :=
  (ProductPiRlcRingCombinationRows.rows
    { base := 0,
      challengeSymbol := fun _ _ => 0,
      input := fun _ _ => 0,
      output := fun _ => 0 }).length

theorem perFamilyArithmeticRowCount_eq :
    perFamilyArithmeticRowCount = 49626 := by
  exact ProductPiRlcRingCombinationRows.rows_length _

/-- Exact local algebra footprint before selectors, transcript rows, and
recursive glue: visible fields plus owned product columns. -/
def perFamilyAlgebraFieldCount : Nat :=
  perFamilyVisibleFieldCount + perFamilyAuxiliaryColumnCount

theorem perFamilyAlgebraFieldCount_eq :
    perFamilyAlgebraFieldCount = 51462 := by
  simp [perFamilyAlgebraFieldCount, perFamilyVisibleFieldCount_eq,
    perFamilyAuxiliaryColumnCount_eq]

/-- Total arithmetic work remains equal to the monolithic 110-family row
census. The work is spread across 110 recursive phases. -/
def totalStreamingArithmeticRowCount : Nat :=
  familyCount * perFamilyArithmeticRowCount

theorem totalStreamingArithmeticRowCount_eq :
    totalStreamingArithmeticRowCount = 5458860 := by
  simp [totalStreamingArithmeticRowCount, familyCount_eq,
    perFamilyArithmeticRowCount_eq]

end Nightstream.Implementation.Nebula.ProductionStreamingPiRlc
