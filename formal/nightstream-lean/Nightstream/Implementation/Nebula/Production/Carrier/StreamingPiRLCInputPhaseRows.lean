import Mathlib.Algebra.BigOperators.Fin
import Nightstream.Implementation.Nebula.Commitment.Lanes.ShiftedTernaryEncodingBridge
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCInputBindingSetup
import Nightstream.Implementation.R1CS.Canonical.GoldilocksField
import Nightstream.Implementation.R1CS.Core.AffinePins
import Nightstream.Implementation.R1CS.Core.SeededPhi81RingRefinement
import Nightstream.Implementation.R1CS.Core.SeededPhi81SamplerRefinement

/-!
Contract: exact fixed-position source and compact seeded rows for one
production PiRLC input family.

Assurance tier: generated source-row soundness.

Owns the shared constrained-zero word, the 810 canonical family openings at
their global positions in the 89,100-field input, the fixed rank-two Phi81
shape, the 108 compact seeded output rows, and their refinement to the exact
local family commitment.

Does not own a fixed production seed, Rust sampler conformance, Poseidon2
replay, PiRLC arithmetic rows, the 108 residual-link rows, family-state glue,
telescoping, or the terminal zero check.

Emits constraints: 100,591 R1CS rows.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows

open scoped BigOperators
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBinding
open Nightstream.Implementation.Nebula.ShiftedTernaryEncodingBridge
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.ShiftedTernaryCanonicalWord
open Nightstream.Implementation.R1CS.ShiftedTernarySound
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.AjtaiBinding
open Nightstream.Protocol.Nebula.ShiftedTernary41V1
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.MatrixCoefficientSource

private abbrev integerResidue :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingSetup.integerResidue

private abbrev signedDigit :=
  Nightstream.Protocol.Nebula.CompactCommit.signedDigit

private abbrev CanonicalOpening :=
  Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.CanonicalOpening

private abbrev canonicalInput :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputResidual.canonicalInput

private abbrev coordinateField :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputResidual.coordinateField

private abbrev coordinateDigit :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputResidual.coordinateDigit

private abbrev phaseWitness :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputResidual.phaseWitness

/-- Columns allocated by one family commitment call. Input and digit columns
are local, while the compact selector keeps their fixed global positions. -/
structure Layout where
  inputColumn : Source → Fin laneCount → Nat
  digitStart : Source → Fin laneCount → Nat
  zeroDigitStart : Nat
  dColumn : Nat
  kappaColumn : Nat
  outputColumn : Fin (shape.rows * shape.degree) → Nat
  seededRowStart : Nat

def Layout.selected (family : Family) (position : Fin fieldCount) : Bool :=
  decide (positionOrdinal position = familyIndex family)

/-- Selected family fields read their canonical digit words. All other
global fields read one constrained zero word. -/
def Layout.wordStart
    (layout : Layout) (family : Family) (position : Fin fieldCount) : Nat :=
  if positionOrdinal position = familyIndex family then
    layout.digitStart (positionSource position) (positionLane position)
  else
    layout.zeroDigitStart

def Layout.wordStarts (layout : Layout) (family : Family) : List Nat :=
  List.ofFn (layout.wordStart family)

theorem Layout.wordStarts_length (layout : Layout) (family : Family) :
    (layout.wordStarts family).length = fieldCount := by
  unfold Layout.wordStarts
  rw [List.length_ofFn]

theorem Layout.wordStarts_getD
    (layout : Layout) (family : Family) (position : Fin fieldCount) :
    (layout.wordStarts family).getD position.val 0 =
      layout.wordStart family position := by
  unfold Layout.wordStarts
  have bound : position.val < (List.ofFn (layout.wordStart family)).length := by
    rw [List.length_ofFn]
    exact position.isLt
  rw [List.getD_eq_getElem _ _ bound]
  exact List.getElem_ofFn bound

theorem Layout.wordStart_selected
    (layout : Layout) (family : Family) (position : Fin fieldCount)
    (selected : positionOrdinal position = familyIndex family) :
    layout.wordStart family position =
      layout.digitStart (positionSource position) (positionLane position) := by
  simp [Layout.wordStart, selected]

theorem Layout.wordStart_unselected
    (layout : Layout) (family : Family) (position : Fin fieldCount)
    (unselected : positionOrdinal position ≠ familyIndex family) :
    layout.wordStart family position = layout.zeroDigitStart := by
  simp [Layout.wordStart, unselected]

/-- Compact seeded block constructor with an opaque selector vector. -/
private def coordinateBlockFromWords
    (setup : SeededAjtai.Setup verifierRows messageColumnCount)
    (layout : Layout) (wordStarts : List Nat) : SeededPhi81.Block where
  rowStart := layout.seededRowStart
  wordStarts := wordStarts
  wordWidth := digitCount
  kappa := verifierRows
  messageCols := messageColumnCount
  outputColumns := List.ofFn layout.outputColumn
  superneoTransformedColumns := false
  schedule := SeededAjtai.schedule setup.seed.bytes verifierRows
    messageColumnCount setup.rejectionFuel

private theorem coordinateBlockFromWords_wordStarts
    (setup : SeededAjtai.Setup verifierRows messageColumnCount)
    (layout : Layout) (wordStarts : List Nat) :
    (coordinateBlockFromWords setup layout wordStarts).wordStarts =
      wordStarts := rfl

/-- Compact seeded block for the full fixed-position message. -/
def coordinateBlock
    (setup : SeededAjtai.Setup verifierRows messageColumnCount)
    (layout : Layout) (family : Family) : SeededPhi81.Block :=
  coordinateBlockFromWords setup layout (layout.wordStarts family)

theorem coordinateBlock_wordStarts
    (setup : SeededAjtai.Setup verifierRows messageColumnCount)
    (layout : Layout) (family : Family) :
    (coordinateBlock setup layout family).wordStarts =
      layout.wordStarts family := by
  unfold coordinateBlock
  exact coordinateBlockFromWords_wordStarts setup layout
    (layout.wordStarts family)

theorem coordinateBlock_messageCols
    (setup : SeededAjtai.Setup verifierRows messageColumnCount)
    (layout : Layout) (family : Family) :
    (coordinateBlock setup layout family).messageCols =
      messageColumnCount := rfl

theorem coordinateBlock_wordWidth
    (setup : SeededAjtai.Setup verifierRows messageColumnCount)
    (layout : Layout) (family : Family) :
    (coordinateBlock setup layout family).wordWidth = digitCount := rfl

theorem coordinateBlock_kappa
    (setup : SeededAjtai.Setup verifierRows messageColumnCount)
    (layout : Layout) (family : Family) :
    (coordinateBlock setup layout family).kappa = verifierRows := rfl

theorem coordinateBlock_outputColumns
    (setup : SeededAjtai.Setup verifierRows messageColumnCount)
    (layout : Layout) (family : Family) :
    (coordinateBlock setup layout family).outputColumns =
      List.ofFn layout.outputColumn := rfl

theorem exact_chunk_geometry :
    SeededAjtai.chunkSize messageColumnCount = 32768 /\
      SeededAjtai.chunkCount messageColumnCount = 3 := by
  decide

theorem coordinateBlock_exact_geometry
    (setup : SeededAjtai.Setup verifierRows messageColumnCount)
    (layout : Layout) (family : Family) :
    (coordinateBlock setup layout family).rowStart = layout.seededRowStart /\
      (coordinateBlock setup layout family).wordStarts.length = 89100 /\
      (coordinateBlock setup layout family).wordWidth = 41 /\
      (coordinateBlock setup layout family).kappa = 2 /\
      (coordinateBlock setup layout family).messageCols = 67650 /\
      (coordinateBlock setup layout family).outputColumns.length = 108 /\
      (coordinateBlock setup layout family).superneoTransformedColumns = false /\
      (coordinateBlock setup layout family).schedule.chunkSize = 32768 := by
  constructor
  · rfl
  constructor
  · rw [coordinateBlock_wordStarts, layout.wordStarts_length]
    decide
  constructor
  · rfl
  constructor
  · rfl
  constructor
  · rfl
  constructor
  · rw [coordinateBlock_outputColumns, List.length_ofFn]
    decide
  constructor
  · rfl
  · exact exact_chunk_geometry.1

/-- The compact block reads the successful pure sampler output selected by
the explicit verifier-owned setup. -/
theorem coordinateBlock_baseRotations
    (setup : SeededAjtai.Setup verifierRows messageColumnCount)
    (layout : Layout) (family : Family) :
    (coordinateBlock setup layout family).baseRotations = setup.outputs := by
  rw [SeededPhi81SamplerRefinement.blockBaseRotations_eq_pure]
  rfl

private theorem ringOfList_baseRotation
    (setup : SeededAjtai.Setup verifierRows messageColumnCount)
    (output : Fin verifierRows) (messageCol : Fin messageColumnCount) :
    SeededPhi81RingRefinement.ringOfList
        ((setup.outputs.getD output.val []).getD messageCol.val []) =
      setup.verifierKey output messageCol := by
  funext lane
  apply Fin.ext
  rfl

/-- Every compact coefficient is the matching coefficient of the explicit
verifier-owned Phi81 matrix. -/
theorem coordinateBlock_coefficient_residue
    (setup : SeededAjtai.Setup verifierRows messageColumnCount)
    (layout : Layout) (family : Family)
    (output : Fin verifierRows) (messageCol : Fin messageColumnCount)
    (messageRow coordinate : Fin ringDegree) :
    SeededPhi81RingRefinement.residueNat
        ((coordinateBlock setup layout family).coefficient
          output.val messageCol.val messageRow.val coordinate.val) =
      CarrierAction.rightCoefficient
        (setup.verifierKey output messageCol) coordinate messageRow := by
  let base := ((setup.outputs.getD output.val []).getD messageCol.val [])
  have rotated := SeededPhi81RingRefinement.ringOfList_rotatePow
    messageRow.val messageRow.isLt base
  rw [ringOfList_baseRotation setup output messageCol] at rotated
  have atCoordinate := congrFun rotated coordinate
  unfold SeededPhi81.Block.coefficient
  rw [coordinateBlock_baseRotations]
  change SeededPhi81RingRefinement.residueNat
      ((SeededPhi81.rotatePow messageRow.val base).getD coordinate.val 0) =
    CarrierAction.rightCoefficient
      (setup.verifierKey output messageCol) coordinate messageRow
  simpa only [SeededPhi81RingRefinement.ringOfList,
    CarrierAction.rightCoefficient] using atCoordinate

private theorem wordIndex_quotient
    (field : Fin fieldCount) (digit : Fin digitCount) :
    wordIndex field digit / digitCount = field.val := by
  unfold wordIndex
  rw [Nat.mul_comm field.val digitCount]
  rw [Nat.mul_add_div (by decide : 0 < digitCount),
    Nat.div_eq_of_lt digit.isLt, Nat.add_zero]

private theorem wordIndex_remainder
    (field : Fin fieldCount) (digit : Fin digitCount) :
    wordIndex field digit % digitCount = digit.val := by
  unfold wordIndex
  exact Nat.mul_add_mod_of_lt digit.isLt

/-- Every matrix coordinate selects the named global field and digit. -/
theorem coordinateBlock_bitColumn
    (setup : SeededAjtai.Setup verifierRows messageColumnCount)
    (layout : Layout) (family : Family)
    (field : Fin fieldCount) (digit : Fin digitCount) :
    (coordinateBlock setup layout family).bitColumn (wordIndex field digit) =
      some (layout.wordStart family field + digit.val) := by
  have bound := wordIndex_lt field digit
  have selectorBound :
      wordIndex field digit <
        (layout.wordStarts family).length * digitCount := by
    rw [layout.wordStarts_length]
    exact bound
  unfold SeededPhi81.Block.bitColumn
  rw [coordinateBlock_wordWidth, coordinateBlock_wordStarts]
  rw [if_neg (by decide : digitCount ≠ 0)]
  rw [if_pos selectorBound]
  rw [wordIndex_quotient, wordIndex_remainder]
  rw [layout.wordStarts_getD]

/-! ## Exact source rows -/

def sourceLanes : List (Source × Fin laneCount) :=
  (List.ofFn fun source : Source => source).flatMap fun source =>
    List.ofFn fun lane : Fin laneCount => (source, lane)

theorem sourceLanes_length : sourceLanes.length = fieldsPerFamily := by
  simp [sourceLanes, fieldsPerFamily, sourceCount, laneCount]

theorem sourceLane_mem
    (source : Source) (lane : Fin laneCount) :
    (source, lane) ∈ sourceLanes := by
  unfold sourceLanes
  apply List.mem_flatMap.mpr
  exact ⟨source, List.mem_ofFn.mpr ⟨source, rfl⟩,
    List.mem_ofFn.mpr ⟨lane, rfl⟩⟩

def zeroPins (layout : Layout) : List AffinePins.Pin :=
  List.ofFn fun digit : Fin digitCount =>
    .zero (layout.zeroDigitStart + digit.val)

def zeroRows (layout : Layout) : List Row :=
  AffinePins.rows (zeroPins layout)

def openingBlockRows
    (layout : Layout) (source : Source) (lane : Fin laneCount) : List Row :=
  ShiftedTernaryCompiler.canonicalRows.map (Relabel.row
    (OwnerCertificate.shiftedTernaryColumnMap
      (layout.inputColumn source lane) (layout.digitStart source lane)))

def openingRows (layout : Layout) : List Row :=
  sourceLanes.flatMap fun pair => openingBlockRows layout pair.1 pair.2

def sourceRows (layout : Layout) : List Row :=
  zeroRows layout ++ openingRows layout

theorem zeroRows_length (layout : Layout) :
    (zeroRows layout).length = 41 := by
  simp [zeroRows, AffinePins.rows, zeroPins, digitCount]

theorem openingBlockRows_length
    (layout : Layout) (source : Source) (lane : Fin laneCount) :
    (openingBlockRows layout source lane).length = 124 := by
  simp [openingBlockRows]
  decide

private theorem openingRowsFor_length
    (layout : Layout) (pairs : List (Source × Fin laneCount)) :
    (pairs.flatMap fun pair => openingBlockRows layout pair.1 pair.2).length =
      pairs.length * 124 := by
  induction pairs with
  | nil => rfl
  | cons pair rest inductionHypothesis =>
      simp [openingBlockRows_length, inductionHypothesis]
      omega

theorem openingRows_length (layout : Layout) :
    (openingRows layout).length = fieldsPerFamily * 124 := by
  unfold openingRows
  rw [openingRowsFor_length, sourceLanes_length]

theorem sourceRows_length (layout : Layout) :
    (sourceRows layout).length = 100481 := by
  rw [sourceRows, List.length_append, zeroRows_length,
    openingRows_length]
  decide

def InputsPlaced
    (layout : Layout) (assignment : Nat → Nat)
    (inputs : Source → RingF) : Prop :=
  ∀ source lane,
    assignment (layout.inputColumn source lane) = (inputs source lane).val

def SourceColumnsExact
    (layout : Layout) (assignment : Nat → Nat)
    (inputs : Source → RingF) : Prop :=
  (∀ source lane digit,
    assignment (layout.digitStart source lane + digit.val) =
      (integerResidue
        (signedDigit (canonicalInput (inputs source lane)) digit)).val) /\
  (∀ digit : Fin digitCount,
    assignment (layout.zeroDigitStart + digit.val) =
      (integerResidue 0).val)

def localAssignment
    (layout : Layout) (assignment : Nat → Nat)
    (source : Source) (lane : Fin laneCount) : Nat → Nat :=
  ShiftedTernaryCanonicalWord.localAssignment assignment
    (layout.inputColumn source lane) (layout.digitStart source lane)

private theorem zero_holds_of_source
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (sourceRows layout) assignment) :
    Satisfies (zeroRows layout) assignment := by
  intro row member
  exact holds row (List.mem_append_left _ member)

private theorem opening_holds_of_source
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (sourceRows layout) assignment) :
    Satisfies (openingRows layout) assignment := by
  intro row member
  exact holds row (List.mem_append_right _ member)

theorem zeroPins_canonical (layout : Layout) :
    AffinePins.PinsCanonical (zeroPins layout) := by
  intro pin member
  rcases List.mem_ofFn.mp member with ⟨digit, rfl⟩
  trivial

theorem zero_word_exact
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (sourceRows layout) assignment)
    (digit : Fin digitCount) :
    assignment (layout.zeroDigitStart + digit.val) =
      (integerResidue 0).val := by
  have pinFacts := AffinePins.rows_sound
    (zeroPins_canonical layout) canonical one (zero_holds_of_source holds)
  have zeroFact := pinFacts
    (.zero (layout.zeroDigitStart + digit.val))
    (List.mem_ofFn.mpr ⟨digit, rfl⟩)
  simpa [AffinePins.Pin.Holds] using zeroFact

private theorem openingBlock_holds
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (openingRows layout) assignment)
    (source : Source) (lane : Fin laneCount) :
    Satisfies (openingBlockRows layout source lane) assignment := by
  intro row member
  apply holds row
  unfold openingRows
  exact List.mem_flatMap.mpr
    ⟨(source, lane), sourceLane_mem source lane, member⟩

theorem opening_of_rows
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (sourceRows layout) assignment)
    (source : Source) (lane : Fin laneCount) :
    CanonicalOpening (localAssignment layout assignment source lane) := by
  apply canonicalOpening_of_canonicalRows
    Nightstream.Implementation.R1CS.Canonical.GoldilocksField.goldilocks_euclidPrime
    (Relabel.canonical canonical)
  · calc
      Relabel.assignment
          (OwnerCertificate.shiftedTernaryColumnMap
            (layout.inputColumn source lane) (layout.digitStart source lane))
          assignment 0 = assignment 0 :=
        ShiftedTernaryCanonicalWord.localAssignment_zero assignment
          (layout.inputColumn source lane) (layout.digitStart source lane)
      _ = 1 := one
  · exact (Relabel.satisfies_mapped_iff _ _ _).mp
      (openingBlock_holds (opening_holds_of_source holds) source lane)

theorem active_digit_exact
    {layout : Layout} {assignment : Nat → Nat}
    {inputs : Source → RingF}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : InputsPlaced layout assignment inputs)
    (holds : Satisfies (sourceRows layout) assignment)
    (source : Source) (lane : Fin laneCount) (digit : Fin digitCount) :
    assignment (layout.digitStart source lane + digit.val) =
      (integerResidue
        (signedDigit (canonicalInput (inputs source lane)) digit)).val := by
  rw [Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingSetup.integerResidue_signedDigit]
  exact productionDigit_eq_protocolDigit
    (canonicalInput (inputs source lane)) (placed source lane)
    (opening_of_rows canonical one holds source lane) digit

theorem sourceColumnsExact_of_rows
    {layout : Layout} {assignment : Nat → Nat}
    {inputs : Source → RingF}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : InputsPlaced layout assignment inputs)
    (holds : Satisfies (sourceRows layout) assignment) :
    SourceColumnsExact layout assignment inputs := by
  constructor
  · intro source lane digit
    exact active_digit_exact canonical one placed holds source lane digit
  · intro digit
    exact zero_word_exact canonical one holds digit

theorem selected_word_exact
    {layout : Layout} {assignment : Nat → Nat}
    {family : Family} {inputs : Source → RingF}
    (exact : SourceColumnsExact layout assignment inputs)
    (position : Fin fieldCount) (digit : Fin digitCount) :
    assignment (layout.wordStart family position + digit.val) =
      (integerResidue
        (if positionOrdinal position = familyIndex family then
          signedDigit
            (canonicalInput
              (inputs (positionSource position) (positionLane position)))
            digit
        else 0)).val := by
  by_cases selected : positionOrdinal position = familyIndex family
  · rw [layout.wordStart_selected family position selected]
    rw [if_pos selected]
    exact exact.1 (positionSource position) (positionLane position) digit
  · rw [layout.wordStart_unselected family position selected]
    rw [if_neg selected]
    exact exact.2 digit

private theorem wordIndex_coordinate
    (column : Fin messageColumnCount) (coefficient : Fin ringDegree) :
    wordIndex (coordinateField column coefficient)
        (coordinateDigit column coefficient) =
      flatIndex column coefficient := by
  unfold wordIndex coordinateField coordinateDigit
  simpa only [Nat.mul_comm] using
    Nat.div_add_mod (flatIndex column coefficient) digitCount

/-- Every compact input coordinate is the exact fixed-position coefficient
of the selected family witness. -/
theorem coordinateBlock_inputValue_exact
    {setup : SeededAjtai.Setup verifierRows messageColumnCount}
    {layout : Layout} {assignment : Nat → Nat}
    {family : Family} {inputs : Source → RingF}
    (exact : SourceColumnsExact layout assignment inputs)
    (messageCol : Fin messageColumnCount) (messageRow : Fin ringDegree) :
    SeededPhi81RingRefinement.residueNat
        ((coordinateBlock setup layout family).inputValue assignment
          messageCol.val messageRow.val) =
      integerResidue (phaseWitness family inputs messageCol messageRow) := by
  let field := coordinateField messageCol messageRow
  let digit := coordinateDigit messageCol messageRow
  have bitColumn :
      (coordinateBlock setup layout family).bitColumn
          (flatIndex messageCol messageRow) =
        some (layout.wordStart family field + digit.val) := by
    rw [← wordIndex_coordinate messageCol messageRow]
    exact coordinateBlock_bitColumn setup layout family field digit
  have nativeIndex :
      messageRow.val * (coordinateBlock setup layout family).messageCols +
          messageCol.val =
        flatIndex messageCol messageRow := by
    rw [coordinateBlock_messageCols]
    rfl
  have nativeBitColumn :
      (coordinateBlock setup layout family).bitColumn
          (messageRow.val *
              (coordinateBlock setup layout family).messageCols +
            messageCol.val) =
        some (layout.wordStart family field + digit.val) := by
    rw [nativeIndex]
    exact bitColumn
  rw [SeededPhi81.Block.inputValue_eq_of_bitColumn_some nativeBitColumn]
  calc
    SeededPhi81RingRefinement.residueNat
        (assignment (layout.wordStart family field + digit.val)) =
        SeededPhi81RingRefinement.residueNat
          (integerResidue
            (if positionOrdinal field = familyIndex family then
              signedDigit
                (canonicalInput
                  (inputs (positionSource field) (positionLane field)))
                digit
            else 0)).val :=
      congrArg SeededPhi81RingRefinement.residueNat
        (selected_word_exact exact field digit)
    _ = integerResidue
        (if positionOrdinal field = familyIndex family then
          signedDigit
            (canonicalInput
              (inputs (positionSource field) (positionLane field))) digit
        else 0) :=
      SeededPhi81RingRefinement.residueNat_fin_val _
    _ = integerResidue (phaseWitness family inputs messageCol messageRow) := by
      rfl

/-! ## Exact compact output -/

private abbrev Phi81Ring :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingSetup.Phi81Ring

private abbrev phaseCoefficientMap :
    CoefficientVector shape →+ Phi81Ring :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingSetup.coefficientMap

private abbrev phaseSeededMatrix
    (setup : SeededAjtai.Setup verifierRows messageColumnCount) :
    Matrix Phi81Ring shape :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingSetup.seededMatrix
    setup

def residualOutputIndex
    (output : Fin shape.rows) (coordinate : Fin shape.degree) :
    Fin (shape.rows * shape.degree) :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingSetup.outputIndex
    output coordinate

private theorem residualOutputIndex_val
    (output : Fin shape.rows) (coordinate : Fin shape.degree) :
    (residualOutputIndex output coordinate).val =
      output.val * shape.degree + coordinate.val := by
  exact
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingSetup.outputIndex_val
      output coordinate

private theorem concretePhaseBinding_outputIndex
    (setup : SeededAjtai.Setup verifierRows messageColumnCount)
    (family : Family) (inputs : Source → RingF)
    (output : Fin shape.rows) (coordinate : Fin shape.degree) :
    (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingSetup.concretePhaseBinding
        setup family inputs (residualOutputIndex output coordinate)).val =
      ((ProductionStreamingPiRlcInputResidual.phaseBinding
        (phaseSeededMatrix setup) phaseCoefficientMap family inputs output
        ).coefficients coordinate).val := by
  unfold
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingSetup.concretePhaseBinding
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingSetup.flattenCommitment
    residualOutputIndex
  rw [Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingSetup.outputPair_outputIndex]

local instance : CommRing F :=
  CommRing.ofMinimalAxioms
    ConcreteCarrier.baseLaws.add_assoc
    ConcreteCarrier.baseLaws.zero_add
    Lean.Grind.Fin.neg_add_cancel
    ConcreteCarrier.baseLaws.mul_assoc
    ConcreteCarrier.baseLaws.mul_comm
    ConcreteCarrier.baseLaws.one_mul
    ConcreteCarrier.baseLaws.left_distrib

private theorem foldRange_residue
    (count initial : Nat) (term : Nat → Nat) :
    SeededPhi81RingRefinement.residueNat
        ((List.range count).foldl
          (fun accumulated index => accumulated + term index) initial) =
      SeededPhi81RingRefinement.residueNat initial +
        sumRange ConcreteCarrier.baseOps count
          (fun index => SeededPhi81RingRefinement.residueNat (term index)) := by
  induction count generalizing initial with
  | zero =>
      exact (ConcreteCarrier.baseLaws.add_zero _).symm
  | succ count inductionHypothesis =>
      rw [List.range_succ, List.foldl_append]
      simp only [List.foldl_cons, List.foldl_nil]
      rw [SeededPhi81RingRefinement.residueNat_add,
        inductionHypothesis, sumRange]
      exact ConcreteCarrier.baseLaws.add_assoc _ _ _

private theorem nestedFold_residue
    (outerCount innerCount : Nat) (term : Nat → Nat → Nat) :
    SeededPhi81RingRefinement.residueNat
        ((List.range outerCount).foldl (fun outer outerIndex =>
          (List.range innerCount).foldl (fun inner innerIndex =>
            inner + term outerIndex innerIndex) outer) 0) =
      sumRange ConcreteCarrier.baseOps outerCount fun outerIndex =>
        sumRange ConcreteCarrier.baseOps innerCount fun innerIndex =>
          SeededPhi81RingRefinement.residueNat
            (term outerIndex innerIndex) := by
  induction outerCount with
  | zero => rfl
  | succ outerCount inductionHypothesis =>
      rw [List.range_succ, List.foldl_append]
      simp only [List.foldl_cons, List.foldl_nil]
      rw [foldRange_residue, inductionHypothesis, sumRange]
      rfl

theorem linearValue_residue
    (block : SeededPhi81.Block) (assignment : Nat → Nat)
    (output coordinate : Nat) :
    SeededPhi81RingRefinement.residueNat
        (block.linearValue assignment output coordinate) =
      sumRange ConcreteCarrier.baseOps block.messageCols fun messageCol =>
        sumRange ConcreteCarrier.baseOps SeededPhi81.dimension fun messageRow =>
          SeededPhi81RingRefinement.residueNat
            (block.termValue assignment output coordinate
              messageCol messageRow) := by
  unfold SeededPhi81.Block.linearValue
  rw [SeededPhi81RingRefinement.residueNat_mod]
  exact nestedFold_residue block.messageCols SeededPhi81.dimension _

theorem coordinateBlock_linearValue_eq_ring_products
    {setup : SeededAjtai.Setup verifierRows messageColumnCount}
    {layout : Layout} {assignment : Nat → Nat}
    {family : Family} {inputs : Source → RingF}
    (exact : SourceColumnsExact layout assignment inputs)
    (output : Fin verifierRows) (coordinate : Fin ringDegree) :
    SeededPhi81RingRefinement.residueNat
        ((coordinateBlock setup layout family).linearValue assignment
          output.val coordinate.val) =
      sumRange ConcreteCarrier.baseOps messageColumnCount fun messageCol =>
        if messageColLt : messageCol < messageColumnCount then
          ringFMul
            (setup.verifierKey output ⟨messageCol, messageColLt⟩)
            (phaseCoefficientMap
              (phaseWitness family inputs
                ⟨messageCol, messageColLt⟩)).coefficients
            coordinate
        else 0 := by
  rw [linearValue_residue, coordinateBlock_messageCols]
  apply sumRange_congr
  intro messageCol messageColLt
  rw [dif_pos messageColLt]
  let column : Fin messageColumnCount := ⟨messageCol, messageColLt⟩
  rw [CarrierAction.ringFMul_apply_eq_rightLinear]
  apply sumRange_congr
  intro messageRow messageRowLt
  have messageRowLtRing : messageRow < ringDegree := by
    simpa [SeededPhi81.dimension, SeededPhi81Sampler.dimension, ringDegree]
      using messageRowLt
  rw [dif_pos messageRowLtRing]
  let row : Fin ringDegree := ⟨messageRow, messageRowLtRing⟩
  unfold SeededPhi81.Block.termValue
  rw [SeededPhi81RingRefinement.residueNat_mul]
  rw [coordinateBlock_coefficient_residue setup layout family output column
    row coordinate]
  rw [coordinateBlock_inputValue_exact exact column row]
  rfl

private theorem finSum_eq_sumRange :
    ∀ {count : Nat} (term : Fin count → F),
      (∑ index, term index) =
        sumRange ConcreteCarrier.baseOps count fun index =>
          if indexLt : index < count then term ⟨index, indexLt⟩ else 0
  | 0, term => by
      rw [Fin.sum_univ_zero]
      rfl
  | count + 1, term => by
      rw [Fin.sum_univ_castSucc, sumRange]
      rw [finSum_eq_sumRange (fun index : Fin count => term index.castSucc)]
      congr 1
      · apply sumRange_congr
        intro index indexLt
        rw [dif_pos indexLt,
          dif_pos (Nat.lt_trans indexLt (Nat.lt_succ_self count))]
        congr 1
      · rw [dif_pos (Nat.lt_succ_self count)]
        congr 1

private def ringCoordinate (coordinate : Fin ringDegree) :
    Phi81Ring →+ F where
  toFun value := value.coefficients coordinate
  map_zero' := rfl
  map_add' := by
    intro left right
    rfl

private theorem commit_coordinate_generic
    {sourceShape : Nightstream.Protocol.Nebula.AjtaiBinding.Shape}
    (matrix : Matrix Phi81Ring sourceShape)
    (map : CoefficientVector sourceShape →+ Phi81Ring)
    (witness : Witness sourceShape)
    (output : Fin sourceShape.rows) (coordinate : Fin ringDegree) :
    ((commit matrix map witness output).coefficients coordinate) =
      ∑ messageCol,
        ringFMul (map (witness messageCol)).coefficients
          (matrix output messageCol).coefficients coordinate := by
  unfold commit
  calc
    ((∑ messageCol,
        map (witness messageCol) * matrix output messageCol) :
        Phi81Ring).coefficients coordinate =
        ∑ messageCol,
          (map (witness messageCol) *
            matrix output messageCol).coefficients coordinate := by
      simpa only [ringCoordinate] using
        (map_sum (ringCoordinate coordinate)
          (fun messageCol : Fin sourceShape.columns =>
            map (witness messageCol) * matrix output messageCol) Finset.univ)
    _ = ∑ messageCol,
        ringFMul (map (witness messageCol)).coefficients
          (matrix output messageCol).coefficients coordinate := by
      rfl

private theorem concretePhaseCommitment_coordinate
    (setup : SeededAjtai.Setup verifierRows messageColumnCount)
    (family : Family) (inputs : Source → RingF)
    (output : Fin verifierRows) (coordinate : Fin ringDegree) :
    ((ProductionStreamingPiRlcInputResidual.phaseBinding
        (phaseSeededMatrix setup) phaseCoefficientMap family inputs output
        ).coefficients coordinate) =
      ∑ messageCol,
        ringFMul (setup.verifierKey output messageCol)
          (phaseCoefficientMap
            (phaseWitness family inputs messageCol)).coefficients
          coordinate := by
  unfold ProductionStreamingPiRlcInputResidual.phaseBinding
  rw [commit_coordinate_generic]
  apply Finset.sum_congr rfl
  intro messageCol _
  change ringFMul
      (phaseCoefficientMap
        (phaseWitness family inputs messageCol)).coefficients
      (setup.verifierKey output messageCol) coordinate = _
  exact congrFun (RingFLaws.ringFMul_comm _ _) coordinate

/-- The exact local family commitment coordinate is the residue of the dense
compact-row value on the same 810 authoritative inputs. -/
theorem phaseCommitment_coordinate_eq_linearValue
    {setup : SeededAjtai.Setup verifierRows messageColumnCount}
    {layout : Layout} {assignment : Nat → Nat}
    {family : Family} {inputs : Source → RingF}
    (exact : SourceColumnsExact layout assignment inputs)
    (output : Fin verifierRows) (coordinate : Fin ringDegree) :
    ((ProductionStreamingPiRlcInputResidual.phaseBinding
        (phaseSeededMatrix setup) phaseCoefficientMap family inputs output
        ).coefficients coordinate) =
      SeededPhi81RingRefinement.residueNat
        ((coordinateBlock setup layout family).linearValue assignment
          output.val coordinate.val) := by
  rw [concretePhaseCommitment_coordinate]
  rw [finSum_eq_sumRange]
  exact
    (coordinateBlock_linearValue_eq_ring_products exact output coordinate).symm

private theorem getD_ofFn
    {alpha : Type} {count : Nat} (function : Fin count → alpha)
    (index : Nat) (fallback : alpha) (bound : index < count) :
    (List.ofFn function).getD index fallback = function ⟨index, bound⟩ := by
  have listBound : index < (List.ofFn function).length := by
    simpa using bound
  rw [List.getD_eq_getElem _ _ listBound]
  exact List.getElem_ofFn listBound

private theorem coordinateBlock_outputColumn
    (setup : SeededAjtai.Setup verifierRows messageColumnCount)
    (layout : Layout) (family : Family)
    (output : Fin verifierRows) (coordinate : Fin ringDegree) :
    (coordinateBlock setup layout family).outputColumns.getD
        (output.val * SeededPhi81.dimension + coordinate.val) 0 =
      layout.outputColumn (residualOutputIndex output coordinate) := by
  rw [coordinateBlock_outputColumns]
  have bound : output.val * SeededPhi81.dimension + coordinate.val <
      shape.rows * shape.degree := by
    have outputLt := output.isLt
    have coordinateLt := coordinate.isLt
    change output.val < 2 at outputLt
    change coordinate.val < 54 at coordinateLt
    change output.val * 54 + coordinate.val < 108
    omega
  rw [getD_ofFn layout.outputColumn _ 0 bound]
  congr 1
  apply Fin.ext
  rw [residualOutputIndex_val]
  rfl

/-- Accepted compact rows determine the exact 108-field local commitment. -/
theorem compact_output_exact_of_rows
    {setup : SeededAjtai.Setup verifierRows messageColumnCount}
    {layout : Layout} {assignment : Nat → Nat}
    {family : Family} {inputs : Source → RingF}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (sourceExact : SourceColumnsExact layout assignment inputs)
    (satisfies : Satisfies
      (coordinateBlock setup layout family).rows assignment)
    (output : Fin verifierRows) (coordinate : Fin ringDegree) :
    assignment (layout.outputColumn (residualOutputIndex output coordinate)) =
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingSetup.concretePhaseBinding
        setup family inputs (residualOutputIndex output coordinate)).val := by
  have outputValue :=
    (coordinateBlock setup layout family).output_eq_linearValue
      (SeededPhi81.sound canonical one satisfies) output coordinate
  rw [coordinateBlock_outputColumn setup layout family output coordinate]
    at outputValue
  rw [outputValue]
  let value :=
    (coordinateBlock setup layout family).linearValue assignment
      output.val coordinate.val
  have commitmentEqual :=
    phaseCommitment_coordinate_eq_linearValue (setup := setup)
      (family := family)
      sourceExact output coordinate
  calc
    value = (SeededPhi81RingRefinement.residueNat value).val := by
      rw [SeededPhi81RingRefinement.residueNat_val,
        Nat.mod_eq_of_lt
          ((coordinateBlock setup layout family).linearValue_lt assignment
            output.val coordinate.val)]
    _ =
        ((ProductionStreamingPiRlcInputResidual.phaseBinding
          (phaseSeededMatrix setup) phaseCoefficientMap family inputs output
          ).coefficients coordinate).val := by
      simpa [value] using congrArg Fin.val commitmentEqual.symm
    _ =
        (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingSetup.concretePhaseBinding
          setup family inputs (residualOutputIndex output coordinate)).val :=
      (concretePhaseBinding_outputIndex setup family inputs output
        coordinate).symm

/-! ## Complete ordered family commitment rows -/

def shapePins (layout : Layout) : List AffinePins.Pin :=
  [.constant layout.dColumn ringDegree,
   .constant layout.kappaColumn verifierRows]

def shapeRows (layout : Layout) : List Row :=
  AffinePins.rows (shapePins layout)

theorem shapePins_canonical (layout : Layout) :
    AffinePins.PinsCanonical (shapePins layout) := by
  intro pin member
  simp [shapePins] at member
  rcases member with rfl | rfl
  · change 0 < ringDegree ∧ ringDegree < goldilocksP
    decide
  · change 0 < verifierRows ∧ verifierRows < goldilocksP
    decide

theorem shapeRows_length (layout : Layout) :
    (shapeRows layout).length = 2 := by
  simp [shapeRows, AffinePins.rows, shapePins]

theorem shape_exact
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (shapeRows layout) assignment) :
    assignment layout.dColumn = ringDegree /\
      assignment layout.kappaColumn = verifierRows := by
  have facts := AffinePins.rows_sound
    (shapePins_canonical layout) canonical one satisfies
  constructor
  · simpa [AffinePins.Pin.Holds] using
      facts (.constant layout.dColumn ringDegree) (by simp [shapePins])
  · simpa [AffinePins.Pin.Holds] using
      facts (.constant layout.kappaColumn verifierRows) (by simp [shapePins])

theorem coordinateRows_length
    (setup : SeededAjtai.Setup verifierRows messageColumnCount)
    (layout : Layout) (family : Family) :
    (coordinateBlock setup layout family).rows.length = 108 := by
  rw [SeededPhi81.Block.rows_length, coordinateBlock_kappa]
  decide

/-- Exact row order: zero word, 810 canonical openings, shape, seeded map. -/
def rows
    (setup : SeededAjtai.Setup verifierRows messageColumnCount)
    (layout : Layout) (family : Family) : List Row :=
  sourceRows layout ++
    (shapeRows layout ++ (coordinateBlock setup layout family).rows)

theorem rows_length
    (setup : SeededAjtai.Setup verifierRows messageColumnCount)
    (layout : Layout) (family : Family) :
    (rows setup layout family).length = 100591 := by
  simp only [rows, List.length_append, sourceRows_length,
    shapeRows_length, coordinateRows_length]

private theorem source_satisfies
    {setup : SeededAjtai.Setup verifierRows messageColumnCount}
    {layout : Layout} {family : Family} {assignment : Nat → Nat}
    (satisfies : Satisfies (rows setup layout family) assignment) :
    Satisfies (sourceRows layout) assignment := by
  intro row member
  exact satisfies row (List.mem_append_left _ member)

private theorem shape_satisfies
    {setup : SeededAjtai.Setup verifierRows messageColumnCount}
    {layout : Layout} {family : Family} {assignment : Nat → Nat}
    (satisfies : Satisfies (rows setup layout family) assignment) :
    Satisfies (shapeRows layout) assignment := by
  intro row member
  exact satisfies row
    (List.mem_append_right _ (List.mem_append_left _ member))

private theorem coordinate_satisfies
    {setup : SeededAjtai.Setup verifierRows messageColumnCount}
    {layout : Layout} {family : Family} {assignment : Nat → Nat}
    (satisfies : Satisfies (rows setup layout family) assignment) :
    Satisfies (coordinateBlock setup layout family).rows assignment := by
  intro row member
  exact satisfies row
    (List.mem_append_right _ (List.mem_append_right _ member))

structure Exact
    (setup : SeededAjtai.Setup verifierRows messageColumnCount)
    (layout : Layout) (family : Family) (assignment : Nat → Nat)
    (inputs : Source → RingF) : Prop where
  d : assignment layout.dColumn = ringDegree
  kappa : assignment layout.kappaColumn = verifierRows
  source : SourceColumnsExact layout assignment inputs
  output : ∀ output : Fin verifierRows, ∀ coordinate : Fin ringDegree,
    assignment (layout.outputColumn (residualOutputIndex output coordinate)) =
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingSetup.concretePhaseBinding
        setup family inputs (residualOutputIndex output coordinate)).val

/-- Coordinate-pair output soundness covers every one of the 108 flattened
commitment fields. -/
theorem Exact.output_at
    {setup : SeededAjtai.Setup verifierRows messageColumnCount}
    {layout : Layout} {family : Family} {assignment : Nat → Nat}
    {inputs : Source → RingF}
    (exact : Exact setup layout family assignment inputs)
    (index : Fin (shape.rows * shape.degree)) :
    assignment (layout.outputColumn index) =
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingSetup.concretePhaseBinding
        setup family inputs index).val := by
  let pair :=
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingSetup.outputPair
      index
  have atPair := exact.output pair.1 pair.2
  simpa [pair, residualOutputIndex] using atPair

/-- Main soundness theorem for one complete 100,591-row family commitment. -/
theorem rows_sound
    {setup : SeededAjtai.Setup verifierRows messageColumnCount}
    {layout : Layout} {family : Family} {assignment : Nat → Nat}
    {inputs : Source → RingF}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : InputsPlaced layout assignment inputs)
    (satisfies : Satisfies (rows setup layout family) assignment) :
    Exact setup layout family assignment inputs := by
  have sourceExact := sourceColumnsExact_of_rows canonical one placed
    (source_satisfies satisfies)
  have shapeExact := shape_exact canonical one (shape_satisfies satisfies)
  refine ⟨shapeExact.1, shapeExact.2, sourceExact, ?_⟩
  intro output coordinate
  exact compact_output_exact_of_rows canonical one sourceExact
    (coordinate_satisfies satisfies) output coordinate

end Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows
