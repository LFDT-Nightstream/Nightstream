import Nightstream.Implementation.R1CS.Canonical.KPointEqualityHonest
import Nightstream.Implementation.R1CS.Canonical.KSparsePolynomialSequentialHonest
import Nightstream.Implementation.R1CS.Canonical.KSplitNcEndpointSupport
import Nightstream.Implementation.R1CS.Canonical.KSplitNcFeInitialHonest

/-!
Contract: constructive completeness for the selected Split-NC FE-terminal
program.

Every arithmetic intermediate is produced by this module's witness.  The only
semantic boundary is the final equality to the authoritative terminal column;
the enclosing selected-NIFS theorem must derive that equality from the frozen
FE relation.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.KSplitNcFeTerminalHonest

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.KCompositeRowSupport
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KSplitNcEndpointSupport
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

private theorem flatMap_congr_local
    {α β : Type} (items : List α) (left right : α → List β)
    (each : ∀ item ∈ items, left item = right item) :
    items.flatMap left = items.flatMap right := by
  induction items with
  | nil => rfl
  | cons item rest inductionHypothesis =>
      rw [List.flatMap_cons, List.flatMap_cons, each item (by simp)]
      congr 1
      exact inductionHypothesis
        (fun value member => each value (by simp [member]))

private def freshPoint
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (fresh : Fin shape.freshCount) (matrix : Fin shape.matrixCount) :
    Carried :=
  input.messageYRing (Data.freshIndex fresh) matrix
    Phi81CoefficientKernel.constant

private theorem freshInputAt_eq
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (fresh : Fin shape.freshCount) :
    KSparsePolynomialSequentialHonest.inputAt
        (Polynomial.Fe.liftedConstraintPolynomial polynomialInput)
        (freshPoint input) input.frameBase fresh.val fresh =
      KSplitNcFeTerminal.freshPolynomialInput input fresh := by
  rfl

private theorem fresh_positions
    {shape : SemanticShape} :
    (canonicalFinIndices shape.freshCount).map
        (fun fresh => fresh.val) =
      List.range' 0 (canonicalFinIndices shape.freshCount).length := by
  rw [canonicalFinIndices_values, canonicalFinIndices_length]
  simp [List.range'_eq_map_range]

private theorem freshRows_eq_sequential
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain) :
    KSplitNcFeTerminal.freshRows input =
      KSparsePolynomialSequentialHonest.rowsFrom
        (Polynomial.Fe.liftedConstraintPolynomial polynomialInput)
        (freshPoint input) input.frameBase
        (canonicalFinIndices shape.freshCount) 0 := by
  unfold KSplitNcFeTerminal.freshRows
  symm
  rw [KSparsePolynomialSequentialHonest.rowsFrom_eq_flatMap
    (canonicalFinIndices shape.freshCount)
    (Polynomial.Fe.liftedConstraintPolynomial polynomialInput)
    (freshPoint input) (fun fresh => fresh.val) input.frameBase 0
    fresh_positions]
  apply flatMap_congr_local
  intro fresh _
  rw [freshInputAt_eq]

def afterFresh
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (assignment : Nat → Nat) : Nat → Nat :=
  KSparsePolynomialSequentialHonest.witnessFrom assignment
    (Polynomial.Fe.liftedConstraintPolynomial polynomialInput)
    (freshPoint input) input.frameBase
    (canonicalFinIndices shape.freshCount) 0

def afterFreshHorner
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (assignment : Nat → Nat) : Nat → Nat :=
  KHornerHonest.hornerWitness (afterFresh input assignment) input.gamma
    (KSplitNcFeTerminal.freshHornerBase input)
    (KSplitNcFeTerminal.freshOutputs input) 0

def afterCarried
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (assignment : Nat → Nat) : Nat → Nat :=
  KSplitNcFeInitialHonest.materializedWitness
    (KSplitNcFeTerminal.carriedInput input)
    (afterFreshHorner input assignment)

private theorem fresh_points_below
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (messageBelow :
      ∀ source matrix lane,
        CarriedBelow
          (input.messageYRing source matrix lane) input.frameBase) :
    ∀ fresh matrix,
      CarriedBelow (freshPoint input fresh matrix) input.frameBase :=
  fun fresh matrix =>
    messageBelow (Data.freshIndex fresh) matrix
      Phi81CoefficientKernel.constant

private theorem freshRows_honest
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (assignment : Nat → Nat)
    (positive : 0 < input.frameBase)
    (messageBelow :
      ∀ source matrix lane,
        CarriedBelow
          (input.messageYRing source matrix lane) input.frameBase) :
    Satisfies (KSplitNcFeTerminal.freshRows input)
      (afterFresh input assignment) := by
  rw [freshRows_eq_sequential]
  exact KSparsePolynomialSequentialHonest.rowsFrom_honest assignment
    (Polynomial.Fe.liftedConstraintPolynomial polynomialInput)
    (freshPoint input) positive (fresh_points_below input messageBelow)
    (canonicalFinIndices shape.freshCount) 0

private theorem freshRows_below_hornerBase
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (positive : 0 < input.frameBase)
    (messageBelow :
      ∀ source matrix lane,
        CarriedBelow
          (input.messageYRing source matrix lane) input.frameBase) :
    RowsBelow (KSplitNcFeTerminal.freshRows input)
      (KSplitNcFeTerminal.freshHornerBase input) := by
  rw [freshRows_eq_sequential]
  have bounded :=
    KSparsePolynomialSequentialHonest.rowsFrom_below_end
      (Polynomial.Fe.liftedConstraintPolynomial polynomialInput)
      (freshPoint input) positive (fresh_points_below input messageBelow)
      (canonicalFinIndices shape.freshCount) 0
  simpa [KSparsePolynomialSequentialHonest.blockWidth,
    KSplitNcFeTerminal.freshHornerBase,
    KSplitNcFeTerminal.sparseRowsPerFresh,
    KSplitNcFeTerminal.polynomialDegreeSum,
    canonicalFinIndices_length, Nat.mul_comm] using bounded

private theorem freshOutput_below_hornerBase
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (positive : 0 < input.frameBase)
    (messageBelow :
      ∀ source matrix lane,
        CarriedBelow
          (input.messageYRing source matrix lane) input.frameBase)
    (fresh : Fin shape.freshCount) :
    CarriedBelow
      (KSparsePolynomial.output
        (KSplitNcFeTerminal.freshPolynomialInput input fresh))
      (KSplitNcFeTerminal.freshHornerBase input) := by
  apply sparsePolynomial_output_below
      (KSplitNcFeTerminal.freshPolynomialInput input fresh)
      (Nat.lt_of_lt_of_le positive (Nat.le_add_right input.frameBase _))
  · intro matrix
    exact carried_mono
      (messageBelow (Data.freshIndex fresh) matrix
        Phi81CoefficientKernel.constant)
      (Nat.le_add_right input.frameBase _)
  · change input.frameBase +
        fresh.val * KSplitNcFeTerminal.sparseRowsPerFresh input +
        KSplitNcFeTerminal.sparseRowsPerFresh input ≤
      KSplitNcFeTerminal.freshHornerBase input
    have bound : fresh.val + 1 ≤ shape.freshCount :=
      Nat.succ_le_iff.mpr fresh.isLt
    have scaled :=
      Nat.mul_le_mul_left
        (KSplitNcFeTerminal.sparseRowsPerFresh input) bound
    rw [Nat.mul_comm fresh.val
      (KSplitNcFeTerminal.sparseRowsPerFresh input)]
    simp only [Nat.mul_succ] at scaled
    unfold KSplitNcFeTerminal.freshHornerBase
    omega

theorem freshOutputs_below_hornerBase
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (positive : 0 < input.frameBase)
    (messageBelow :
      ∀ source matrix lane,
        CarriedBelow
          (input.messageYRing source matrix lane) input.frameBase) :
    ∀ output ∈ KSplitNcFeTerminal.freshOutputs input,
      CarriedBelow output (KSplitNcFeTerminal.freshHornerBase input) := by
  intro output member
  rcases List.mem_map.1 member with ⟨fresh, _, rfl⟩
  exact freshOutput_below_hornerBase input positive messageBelow fresh

private theorem freshHornerRows_honest
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (assignment : Nat → Nat)
    (positive : 0 < input.frameBase)
    (gammaBelow : CarriedBelow input.gamma input.frameBase)
    (messageBelow :
      ∀ source matrix lane,
        CarriedBelow
          (input.messageYRing source matrix lane) input.frameBase) :
    Satisfies (KSplitNcFeTerminal.freshHornerRows input)
      (afterFreshHorner input assignment) := by
  unfold KSplitNcFeTerminal.freshHornerRows afterFreshHorner
  exact KHornerHonest.hornerWitness_satisfies
    (afterFresh input assignment) input.gamma
    (KSplitNcFeTerminal.freshHornerBase input)
    (carried_mono gammaBelow (Nat.le_add_right input.frameBase _)).1
    (carried_mono gammaBelow (Nat.le_add_right input.frameBase _)).2
    (KSplitNcFeTerminal.freshOutputs input) 0
    (freshOutputs_below_hornerBase input positive messageBelow)

private theorem freshPrefix_honest
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (assignment : Nat → Nat)
    (positive : 0 < input.frameBase)
    (gammaBelow : CarriedBelow input.gamma input.frameBase)
    (messageBelow :
      ∀ source matrix lane,
        CarriedBelow
          (input.messageYRing source matrix lane) input.frameBase) :
    Satisfies
      (KSplitNcFeTerminal.freshRows input ++
        KSplitNcFeTerminal.freshHornerRows input)
      (afterFreshHorner input assignment) := by
  have freshSatisfied :=
    freshRows_honest input assignment positive messageBelow
  have freshPreserved :
      Satisfies (KSplitNcFeTerminal.freshRows input)
        (afterFreshHorner input assignment) := by
    apply KHornerSupport.satisfies_extend _
      (afterFresh input assignment) (afterFreshHorner input assignment)
    · intro row member column mentioned
      symm
      apply KHornerHonest.hornerWitness_off_block
      exact freshRows_below_hornerBase input positive messageBelow
        row member column mentioned
    · exact freshSatisfied
  have hornerSatisfied :=
    freshHornerRows_honest input assignment positive gammaBelow messageBelow
  intro row member
  exact (List.mem_append.1 member).elim
    (freshPreserved row) (hornerSatisfied row)

private theorem freshPrefix_below_carriedBase
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (positive : 0 < input.frameBase)
    (gammaBelow : CarriedBelow input.gamma input.frameBase)
    (messageBelow :
      ∀ source matrix lane,
        CarriedBelow
          (input.messageYRing source matrix lane) input.frameBase) :
    RowsBelow
      (KSplitNcFeTerminal.freshRows input ++
        KSplitNcFeTerminal.freshHornerRows input)
      (KSplitNcFeTerminal.carriedBase input) := by
  have freshBelow :
      RowsBelow (KSplitNcFeTerminal.freshRows input)
        (KSplitNcFeTerminal.carriedBase input) := by
    intro row member column mentioned
    exact Nat.lt_of_lt_of_le
      (freshRows_below_hornerBase input positive messageBelow
        row member column mentioned)
      (by
        unfold KSplitNcFeTerminal.carriedBase
        omega)
  have hornerBelow :
      RowsBelow (KSplitNcFeTerminal.freshHornerRows input)
        (KSplitNcFeTerminal.carriedBase input) := by
    unfold KSplitNcFeTerminal.freshHornerRows
    apply horner_rows_below
    · exact carried_mono gammaBelow (by
        unfold KSplitNcFeTerminal.carriedBase
          KSplitNcFeTerminal.freshHornerBase
        omega)
    · intro output member
      exact carried_mono
        (freshOutputs_below_hornerBase input positive messageBelow
          output member)
        (by
          unfold KSplitNcFeTerminal.carriedBase
          omega)
    · rw [KSplitNcFeTerminal.freshOutputs_length]
      unfold KSplitNcFeTerminal.carriedBase
      omega
  intro row member column mentioned
  exact (List.mem_append.1 member).elim
    (fun inFresh => freshBelow row inFresh column mentioned)
    (fun inHorner => hornerBelow row inHorner column mentioned)

private theorem carried_target_is_canonical
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain) :
    (KSplitNcFeTerminal.carriedInput input).initial =
      KSplitNcFeInitialHonest.canonicalTarget
        (KSplitNcFeTerminal.carriedInput input) := by
  rfl

private theorem carriedRows_honest
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (assignment : Nat → Nat)
    (positive : 0 < input.frameBase)
    (constantWire : assignment 0 = 1)
    (gammaBelow : CarriedBelow input.gamma input.frameBase)
    (pointLaneBelow :
      ∀ coordinate,
        CarriedBelow (input.pointLane coordinate) input.frameBase)
    (messageBelow :
      ∀ source matrix lane,
        CarriedBelow
          (input.messageYRing source matrix lane) input.frameBase) :
    Satisfies (KSplitNcFeTerminal.carriedRows input)
      (afterCarried input assignment) := by
  unfold KSplitNcFeTerminal.carriedRows afterCarried
  apply KSplitNcFeInitialHonest.rows_honest_canonicalTarget
  · change 0 < KSplitNcFeTerminal.carriedBase input
    unfold KSplitNcFeTerminal.carriedBase
      KSplitNcFeTerminal.freshHornerBase
    omega
  · have one :
        afterFreshHorner input assignment 0 = 1 := by
      unfold afterFreshHorner
      rw [KHornerHonest.hornerWitness_off_block _ _ _ _ 0 0 (by
          unfold KSplitNcFeTerminal.freshHornerBase
          omega)]
      unfold afterFresh
      rw [KSparsePolynomialSequentialHonest.witnessFrom_off_before
        assignment
        (Polynomial.Fe.liftedConstraintPolynomial polynomialInput)
        (freshPoint input) (canonicalFinIndices shape.freshCount)
        input.frameBase 0 0 (by simpa using positive)]
      exact constantWire
    exact one
  · exact carried_mono gammaBelow (by
      change input.frameBase ≤ KSplitNcFeTerminal.carriedBase input
      unfold KSplitNcFeTerminal.carriedBase
        KSplitNcFeTerminal.freshHornerBase
      omega)
  · intro coordinate
    exact carried_mono (pointLaneBelow coordinate) (by
      change input.frameBase ≤ KSplitNcFeTerminal.carriedBase input
      unfold KSplitNcFeTerminal.carriedBase
        KSplitNcFeTerminal.freshHornerBase
      omega)
  · intro running matrix lane
    exact carried_mono
      (messageBelow (Data.runningIndex running) matrix lane) (by
        change input.frameBase ≤ KSplitNcFeTerminal.carriedBase input
        unfold KSplitNcFeTerminal.carriedBase
          KSplitNcFeTerminal.freshHornerBase
        omega)
  · exact carried_target_is_canonical input

private theorem arithmeticPrefix_honest
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (assignment : Nat → Nat)
    (positive : 0 < input.frameBase)
    (constantWire : assignment 0 = 1)
    (gammaBelow : CarriedBelow input.gamma input.frameBase)
    (pointLaneBelow :
      ∀ coordinate,
        CarriedBelow (input.pointLane coordinate) input.frameBase)
    (messageBelow :
      ∀ source matrix lane,
        CarriedBelow
          (input.messageYRing source matrix lane) input.frameBase) :
    Satisfies
      (KSplitNcFeTerminal.freshRows input ++
        KSplitNcFeTerminal.freshHornerRows input ++
        KSplitNcFeTerminal.carriedRows input)
      (afterCarried input assignment) := by
  have prefixSatisfied :=
    freshPrefix_honest input assignment positive gammaBelow messageBelow
  have prefixPreserved :
      Satisfies
        (KSplitNcFeTerminal.freshRows input ++
          KSplitNcFeTerminal.freshHornerRows input)
        (afterCarried input assignment) := by
    apply KHornerSupport.satisfies_extend _
      (afterFreshHorner input assignment) (afterCarried input assignment)
    · intro row member column mentioned
      symm
      exact KSplitNcFeInitialHonest.materializedWitness_off_block
        (KSplitNcFeTerminal.carriedInput input)
        (afterFreshHorner input assignment) column
        (freshPrefix_below_carriedBase input positive gammaBelow messageBelow
          row member column mentioned)
    · exact prefixSatisfied
  have carriedSatisfied :=
    carriedRows_honest input assignment positive constantWire gammaBelow
      pointLaneBelow messageBelow
  intro row member
  rcases List.mem_append.1 member with inPrefix | inCarried
  · exact prefixPreserved row inPrefix
  · exact carriedSatisfied row inCarried

def arithmeticRows
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain) : List Row :=
  KSplitNcFeTerminal.freshRows input ++
    KSplitNcFeTerminal.freshHornerRows input ++
    KSplitNcFeTerminal.carriedRows input

def afterFreshLane
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (assignment : Nat → Nat) : Nat → Nat :=
  KPointEqualityHonest.witness (afterCarried input assignment)
    (KSplitNcFeTerminal.freshLaneEqualityInput input)

def afterFreshRow
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (assignment : Nat → Nat) : Nat → Nat :=
  KPointEqualityHonest.witness (afterFreshLane input assignment)
    (KSplitNcFeTerminal.freshRowEqualityInput input)

def afterCarriedLane
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (assignment : Nat → Nat) : Nat → Nat :=
  KPointEqualityHonest.witness (afterFreshRow input assignment)
    (KSplitNcFeTerminal.carriedLaneEqualityInput input)

def afterCarriedRow
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (assignment : Nat → Nat) : Nat → Nat :=
  KPointEqualityHonest.witness (afterCarriedLane input assignment)
    (KSplitNcFeTerminal.carriedRowEqualityInput input)

private theorem base_le_equalityBase
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain) :
    input.frameBase ≤ KSplitNcFeTerminal.equalityBase input := by
  unfold KSplitNcFeTerminal.equalityBase
    KSplitNcFeTerminal.carriedTargetBase
    KSplitNcFeTerminal.carriedBase
    KSplitNcFeTerminal.freshHornerBase
  omega

theorem afterCarriedRow_off_source
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (assignment : Nat → Nat) (column : Nat)
    (below : column < input.frameBase) :
    afterCarriedRow input assignment column = assignment column := by
  calc
    afterCarriedRow input assignment column =
        afterCarriedLane input assignment column :=
      KPointEqualityHonest.witness_off_block
        (afterCarriedLane input assignment)
        (KSplitNcFeTerminal.carriedRowEqualityInput input) column (by
          exact Nat.lt_of_lt_of_le below
            (Nat.le_trans (base_le_equalityBase input) (by
              change
                KSplitNcFeTerminal.equalityBase input ≤
                  KSplitNcFeTerminal.equalityBase input +
                    2 * KSplitNcFeTerminal.pointEqualityRows
                      domain.laneVariables +
                    KSplitNcFeTerminal.pointEqualityRows shape.rowVariables
              omega)))
    _ = afterFreshRow input assignment column :=
      KPointEqualityHonest.witness_off_block
        (afterFreshRow input assignment)
        (KSplitNcFeTerminal.carriedLaneEqualityInput input) column (by
          exact Nat.lt_of_lt_of_le below
            (Nat.le_trans (base_le_equalityBase input) (by
              change
                KSplitNcFeTerminal.equalityBase input ≤
                  KSplitNcFeTerminal.equalityBase input +
                    KSplitNcFeTerminal.pointEqualityRows
                      domain.laneVariables +
                    KSplitNcFeTerminal.pointEqualityRows shape.rowVariables
              omega)))
    _ = afterFreshLane input assignment column :=
      KPointEqualityHonest.witness_off_block
        (afterFreshLane input assignment)
        (KSplitNcFeTerminal.freshRowEqualityInput input) column (by
          exact Nat.lt_of_lt_of_le below
            (Nat.le_trans (base_le_equalityBase input) (by
              change
                KSplitNcFeTerminal.equalityBase input ≤
                  KSplitNcFeTerminal.equalityBase input +
                    KSplitNcFeTerminal.pointEqualityRows
                      domain.laneVariables
              omega)))
    _ = afterCarried input assignment column :=
      KPointEqualityHonest.witness_off_block
        (afterCarried input assignment)
        (KSplitNcFeTerminal.freshLaneEqualityInput input) column (by
          exact Nat.lt_of_lt_of_le below
            (base_le_equalityBase input))
    _ = afterFreshHorner input assignment column :=
      KSplitNcFeInitialHonest.materializedWitness_off_block
        (KSplitNcFeTerminal.carriedInput input)
        (afterFreshHorner input assignment) column (by
          change column < KSplitNcFeTerminal.carriedBase input
          unfold KSplitNcFeTerminal.carriedBase
            KSplitNcFeTerminal.freshHornerBase
          omega)
    _ = afterFresh input assignment column :=
      KHornerHonest.hornerWitness_off_block
        (afterFresh input assignment) input.gamma
        (KSplitNcFeTerminal.freshHornerBase input)
        (KSplitNcFeTerminal.freshOutputs input) 0 column (by
          unfold KSplitNcFeTerminal.freshHornerBase
          omega)
    _ = assignment column :=
      KSparsePolynomialSequentialHonest.witnessFrom_off_before
        assignment
        (Polynomial.Fe.liftedConstraintPolynomial polynomialInput)
        (freshPoint input) (canonicalFinIndices shape.freshCount)
        input.frameBase 0 column (by simpa using below)

def freshLanePrefix
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain) : List Row :=
  arithmeticRows input ++
    KPointEquality.rows (KSplitNcFeTerminal.freshLaneEqualityInput input)

def freshRowPrefix
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain) : List Row :=
  freshLanePrefix input ++
    KPointEquality.rows (KSplitNcFeTerminal.freshRowEqualityInput input)

def carriedLanePrefix
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain) : List Row :=
  freshRowPrefix input ++
    KPointEquality.rows (KSplitNcFeTerminal.carriedLaneEqualityInput input)

def pointPrefix
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain) : List Row :=
  carriedLanePrefix input ++
    KPointEquality.rows (KSplitNcFeTerminal.carriedRowEqualityInput input)

private theorem rowsBelow_append
    {left right : List Row} {boundary : Nat}
    (leftBelow : RowsBelow left boundary)
    (rightBelow : RowsBelow right boundary) :
    RowsBelow (left ++ right) boundary := by
  intro row member column mentioned
  exact (List.mem_append.1 member).elim
    (fun inLeft => leftBelow row inLeft column mentioned)
    (fun inRight => rightBelow row inRight column mentioned)

private theorem rowsBelow_mono
    {rows : List Row} {lower upper : Nat}
    (below : RowsBelow rows lower) (ordered : lower ≤ upper) :
    RowsBelow rows upper := by
  intro row member column mentioned
  exact Nat.lt_of_lt_of_le
    (below row member column mentioned) ordered

private theorem equalityBase_le_freshRowBase
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain) :
    KSplitNcFeTerminal.equalityBase input ≤
      (KSplitNcFeTerminal.freshRowEqualityInput input).frameBase := by
  change KSplitNcFeTerminal.equalityBase input ≤
    KSplitNcFeTerminal.equalityBase input +
      KSplitNcFeTerminal.pointEqualityRows domain.laneVariables
  omega

private theorem base_le_freshRowBase
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain) :
    input.frameBase ≤
      (KSplitNcFeTerminal.freshRowEqualityInput input).frameBase :=
  Nat.le_trans (base_le_equalityBase input)
    (equalityBase_le_freshRowBase input)

private theorem satisfies_after_append
    {oldRows newRows : List Row}
    {oldAssignment newAssignment : Nat → Nat}
    {boundary : Nat}
    (oldSatisfied : Satisfies oldRows oldAssignment)
    (oldBelow : RowsBelow oldRows boundary)
    (off :
      ∀ column, column < boundary →
        newAssignment column = oldAssignment column)
    (newSatisfied : Satisfies newRows newAssignment) :
    Satisfies (oldRows ++ newRows) newAssignment := by
  have preserved :
      Satisfies oldRows newAssignment := by
    apply KHornerSupport.satisfies_extend _
      oldAssignment newAssignment
    · intro row member column mentioned
      exact (off column (oldBelow row member column mentioned)).symm
    · exact oldSatisfied
  intro row member
  exact (List.mem_append.1 member).elim
    (preserved row) (newSatisfied row)

theorem target_below_equalityBase
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain) :
    CarriedBelow (KSplitNcFeTerminal.carriedTarget input)
      (KSplitNcFeTerminal.equalityBase input) := by
  constructor <;> intro column mentioned
  · have same : column = KSplitNcFeTerminal.carriedTargetBase input := by
      simpa [KSplitNcFeTerminal.carriedTarget, Mentions] using mentioned
    subst column
    unfold KSplitNcFeTerminal.equalityBase
    omega
  · have same :
        column = KSplitNcFeTerminal.carriedTargetBase input + 1 := by
      simpa [KSplitNcFeTerminal.carriedTarget, Mentions] using mentioned
    subst column
    unfold KSplitNcFeTerminal.equalityBase
    omega

private theorem arithmeticRows_below_equalityBase
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (positive : 0 < input.frameBase)
    (gammaBelow : CarriedBelow input.gamma input.frameBase)
    (pointLaneBelow :
      ∀ coordinate,
        CarriedBelow (input.pointLane coordinate) input.frameBase)
    (messageBelow :
      ∀ source matrix lane,
        CarriedBelow
          (input.messageYRing source matrix lane) input.frameBase) :
    RowsBelow (arithmeticRows input)
      (KSplitNcFeTerminal.equalityBase input) := by
  have freshBelow :=
    freshPrefix_below_carriedBase input positive gammaBelow messageBelow
  have freshAtEnd :
      RowsBelow
        (KSplitNcFeTerminal.freshRows input ++
          KSplitNcFeTerminal.freshHornerRows input)
        (KSplitNcFeTerminal.equalityBase input) :=
    rowsBelow_mono freshBelow (by
      unfold KSplitNcFeTerminal.equalityBase
        KSplitNcFeTerminal.carriedTargetBase
      omega)
  have carriedBelow :
      RowsBelow (KSplitNcFeTerminal.carriedRows input)
        (KSplitNcFeTerminal.equalityBase input) := by
    unfold KSplitNcFeTerminal.carriedRows
    apply feInitial_rows_below
        (KSplitNcFeTerminal.carriedInput input)
        (KSplitNcFeTerminal.equalityBase input)
        (Nat.lt_of_lt_of_le positive (base_le_equalityBase input))
    · exact carried_mono gammaBelow (by
        change input.frameBase ≤ KSplitNcFeTerminal.carriedBase input
        unfold KSplitNcFeTerminal.carriedBase
          KSplitNcFeTerminal.freshHornerBase
        omega)
    · intro coordinate
      exact carried_mono (pointLaneBelow coordinate) (by
        change input.frameBase ≤ KSplitNcFeTerminal.carriedBase input
        unfold KSplitNcFeTerminal.carriedBase
          KSplitNcFeTerminal.freshHornerBase
        omega)
    · intro running matrix lane
      exact carried_mono
        (messageBelow (Data.runningIndex running) matrix lane) (by
          change input.frameBase ≤ KSplitNcFeTerminal.carriedBase input
          unfold KSplitNcFeTerminal.carriedBase
            KSplitNcFeTerminal.freshHornerBase
          omega)
    · exact target_below_equalityBase input
    · change
        KSplitNcFeTerminal.carriedBase input +
            KSplitNcFeInitial.allocationWidth
              (KSplitNcFeTerminal.carriedInput input) ≤
          KSplitNcFeTerminal.equalityBase input
      unfold KSplitNcFeTerminal.equalityBase
        KSplitNcFeTerminal.carriedTargetBase
        KSplitNcFeTerminal.carriedInternalWidth
        KSplitNcFeInitial.allocationWidth
      omega
  exact rowsBelow_append freshAtEnd carriedBelow

private theorem pointRows_below
    {variables : Nat}
    (pointInput : KPointEquality.Input variables)
    (boundary : Nat) (positive : 0 < boundary)
    (leftBelow :
      ∀ coordinate, CarriedBelow (pointInput.left coordinate) boundary)
    (rightBelow :
      ∀ coordinate, CarriedBelow (pointInput.right coordinate) boundary)
    (endBound :
      pointInput.frameBase +
          KSplitNcFeTerminal.pointEqualityRows variables ≤ boundary) :
    RowsBelow (KPointEquality.rows pointInput) boundary := by
  apply pointEquality_rows_below pointInput positive leftBelow rightBelow
  unfold KSplitNcFeTerminal.pointEqualityRows at endBound
  simpa [Nat.add_assoc] using endBound

private theorem freshLanePrefix_honest
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (assignment : Nat → Nat)
    (positive : 0 < input.frameBase)
    (constantWire : assignment 0 = 1)
    (gammaBelow : CarriedBelow input.gamma input.frameBase)
    (pointLaneBelow :
      ∀ coordinate,
        CarriedBelow (input.pointLane coordinate) input.frameBase)
    (betaABelow :
      ∀ coordinate,
        CarriedBelow (input.betaA coordinate) input.frameBase)
    (messageBelow :
      ∀ source matrix lane,
        CarriedBelow
          (input.messageYRing source matrix lane) input.frameBase) :
    Satisfies (freshLanePrefix input) (afterFreshLane input assignment) := by
  apply satisfies_after_append
      (arithmeticPrefix_honest input assignment positive constantWire
        gammaBelow pointLaneBelow messageBelow)
      (arithmeticRows_below_equalityBase input positive gammaBelow
        pointLaneBelow messageBelow)
  · intro column below
    exact KPointEqualityHonest.witness_off_block
      (afterCarried input assignment)
      (KSplitNcFeTerminal.freshLaneEqualityInput input) column
      (by simpa [KSplitNcFeTerminal.freshLaneEqualityInput] using below)
  · apply KPointEqualityHonest.rows_honest
    · change 0 < KSplitNcFeTerminal.equalityBase input
      exact Nat.lt_of_lt_of_le positive
        (base_le_equalityBase input)
    · intro coordinate
      exact carried_mono (pointLaneBelow coordinate)
        (base_le_equalityBase input)
    · intro coordinate
      exact carried_mono (betaABelow coordinate)
        (base_le_equalityBase input)

private theorem freshLanePrefix_below_freshRowBase
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (positive : 0 < input.frameBase)
    (gammaBelow : CarriedBelow input.gamma input.frameBase)
    (pointLaneBelow :
      ∀ coordinate,
        CarriedBelow (input.pointLane coordinate) input.frameBase)
    (betaABelow :
      ∀ coordinate,
        CarriedBelow (input.betaA coordinate) input.frameBase)
    (messageBelow :
      ∀ source matrix lane,
        CarriedBelow
          (input.messageYRing source matrix lane) input.frameBase) :
    RowsBelow (freshLanePrefix input)
      (KSplitNcFeTerminal.freshRowEqualityInput input).frameBase := by
  have arithmeticBelow :=
    rowsBelow_mono
      (arithmeticRows_below_equalityBase input positive gammaBelow
        pointLaneBelow messageBelow)
      (equalityBase_le_freshRowBase input)
  have equalityBelow :
      RowsBelow
        (KPointEquality.rows
          (KSplitNcFeTerminal.freshLaneEqualityInput input))
        (KSplitNcFeTerminal.freshRowEqualityInput input).frameBase := by
    apply pointRows_below
    · exact Nat.lt_of_lt_of_le positive
        (base_le_freshRowBase input)
    · intro coordinate
      exact carried_mono (pointLaneBelow coordinate)
        (base_le_freshRowBase input)
    · intro coordinate
      exact carried_mono (betaABelow coordinate)
        (base_le_freshRowBase input)
    · exact Nat.le_refl _
  exact rowsBelow_append arithmeticBelow equalityBelow

private theorem freshRowPrefix_honest
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (assignment : Nat → Nat)
    (positive : 0 < input.frameBase)
    (constantWire : assignment 0 = 1)
    (gammaBelow : CarriedBelow input.gamma input.frameBase)
    (pointLaneBelow :
      ∀ coordinate,
        CarriedBelow (input.pointLane coordinate) input.frameBase)
    (pointRowBelow :
      ∀ coordinate,
        CarriedBelow (input.pointRow coordinate) input.frameBase)
    (betaABelow :
      ∀ coordinate,
        CarriedBelow (input.betaA coordinate) input.frameBase)
    (betaRBelow :
      ∀ coordinate,
        CarriedBelow (input.betaR coordinate) input.frameBase)
    (messageBelow :
      ∀ source matrix lane,
        CarriedBelow
          (input.messageYRing source matrix lane) input.frameBase) :
    Satisfies (freshRowPrefix input) (afterFreshRow input assignment) := by
  apply satisfies_after_append
      (freshLanePrefix_honest input assignment positive constantWire
        gammaBelow pointLaneBelow betaABelow messageBelow)
      (freshLanePrefix_below_freshRowBase input positive gammaBelow
        pointLaneBelow betaABelow messageBelow)
  · intro column below
    exact KPointEqualityHonest.witness_off_block
      (afterFreshLane input assignment)
      (KSplitNcFeTerminal.freshRowEqualityInput input) column below
  · apply KPointEqualityHonest.rows_honest
    · exact Nat.lt_of_lt_of_le positive
        (base_le_freshRowBase input)
    · intro coordinate
      exact carried_mono (pointRowBelow coordinate)
        (base_le_freshRowBase input)
    · intro coordinate
      exact carried_mono (betaRBelow coordinate)
        (base_le_freshRowBase input)

private theorem freshRowPrefix_below_carriedLaneBase
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (positive : 0 < input.frameBase)
    (gammaBelow : CarriedBelow input.gamma input.frameBase)
    (pointLaneBelow :
      ∀ coordinate,
        CarriedBelow (input.pointLane coordinate) input.frameBase)
    (pointRowBelow :
      ∀ coordinate,
        CarriedBelow (input.pointRow coordinate) input.frameBase)
    (betaABelow :
      ∀ coordinate,
        CarriedBelow (input.betaA coordinate) input.frameBase)
    (betaRBelow :
      ∀ coordinate,
        CarriedBelow (input.betaR coordinate) input.frameBase)
    (messageBelow :
      ∀ source matrix lane,
        CarriedBelow
          (input.messageYRing source matrix lane) input.frameBase) :
    RowsBelow (freshRowPrefix input)
      (KSplitNcFeTerminal.carriedLaneEqualityInput input).frameBase := by
  have prefixBelow :
      RowsBelow (freshLanePrefix input)
        (KSplitNcFeTerminal.carriedLaneEqualityInput input).frameBase :=
    rowsBelow_mono
      (freshLanePrefix_below_freshRowBase input positive gammaBelow
        pointLaneBelow betaABelow messageBelow)
      (by
        change
          KSplitNcFeTerminal.equalityBase input +
              KSplitNcFeTerminal.pointEqualityRows domain.laneVariables ≤
            KSplitNcFeTerminal.equalityBase input +
              KSplitNcFeTerminal.pointEqualityRows domain.laneVariables +
              KSplitNcFeTerminal.pointEqualityRows shape.rowVariables
        omega)
  have equalityBelow :
      RowsBelow
        (KPointEquality.rows
          (KSplitNcFeTerminal.freshRowEqualityInput input))
        (KSplitNcFeTerminal.carriedLaneEqualityInput input).frameBase := by
    apply pointRows_below
    · exact Nat.lt_of_lt_of_le positive (by
        change input.frameBase ≤
          KSplitNcFeTerminal.equalityBase input +
            KSplitNcFeTerminal.pointEqualityRows domain.laneVariables +
            KSplitNcFeTerminal.pointEqualityRows shape.rowVariables
        have := base_le_equalityBase input
        omega)
    · intro coordinate
      exact carried_mono (pointRowBelow coordinate) (by
        change input.frameBase ≤
          (KSplitNcFeTerminal.carriedLaneEqualityInput input).frameBase
        have := base_le_equalityBase input
        change input.frameBase ≤
          KSplitNcFeTerminal.equalityBase input +
            KSplitNcFeTerminal.pointEqualityRows domain.laneVariables +
            KSplitNcFeTerminal.pointEqualityRows shape.rowVariables
        omega)
    · intro coordinate
      exact carried_mono (betaRBelow coordinate) (by
        change input.frameBase ≤
          (KSplitNcFeTerminal.carriedLaneEqualityInput input).frameBase
        have := base_le_equalityBase input
        change input.frameBase ≤
          KSplitNcFeTerminal.equalityBase input +
            KSplitNcFeTerminal.pointEqualityRows domain.laneVariables +
            KSplitNcFeTerminal.pointEqualityRows shape.rowVariables
        omega)
    · change
        KSplitNcFeTerminal.equalityBase input +
              KSplitNcFeTerminal.pointEqualityRows domain.laneVariables +
              KSplitNcFeTerminal.pointEqualityRows shape.rowVariables ≤
          KSplitNcFeTerminal.equalityBase input +
              KSplitNcFeTerminal.pointEqualityRows domain.laneVariables +
              KSplitNcFeTerminal.pointEqualityRows shape.rowVariables
      exact Nat.le_refl _
  exact rowsBelow_append prefixBelow equalityBelow

private theorem carriedLanePrefix_honest
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (assignment : Nat → Nat)
    (positive : 0 < input.frameBase)
    (constantWire : assignment 0 = 1)
    (gammaBelow : CarriedBelow input.gamma input.frameBase)
    (pointLaneBelow :
      ∀ coordinate,
        CarriedBelow (input.pointLane coordinate) input.frameBase)
    (pointRowBelow :
      ∀ coordinate,
        CarriedBelow (input.pointRow coordinate) input.frameBase)
    (betaABelow :
      ∀ coordinate,
        CarriedBelow (input.betaA coordinate) input.frameBase)
    (betaRBelow :
      ∀ coordinate,
        CarriedBelow (input.betaR coordinate) input.frameBase)
    (alphaBelow :
      ∀ coordinate,
        CarriedBelow (input.alpha coordinate) input.frameBase)
    (messageBelow :
      ∀ source matrix lane,
        CarriedBelow
          (input.messageYRing source matrix lane) input.frameBase) :
    Satisfies (carriedLanePrefix input)
      (afterCarriedLane input assignment) := by
  apply satisfies_after_append
      (freshRowPrefix_honest input assignment positive constantWire
        gammaBelow pointLaneBelow pointRowBelow betaABelow betaRBelow
        messageBelow)
      (freshRowPrefix_below_carriedLaneBase input positive gammaBelow
        pointLaneBelow pointRowBelow betaABelow betaRBelow messageBelow)
  · intro column below
    exact KPointEqualityHonest.witness_off_block
      (afterFreshRow input assignment)
      (KSplitNcFeTerminal.carriedLaneEqualityInput input) column below
  · apply KPointEqualityHonest.rows_honest
    · exact Nat.lt_of_lt_of_le positive (by
        change input.frameBase ≤
          (KSplitNcFeTerminal.carriedLaneEqualityInput input).frameBase
        have := base_le_equalityBase input
        change input.frameBase ≤
          KSplitNcFeTerminal.equalityBase input +
            KSplitNcFeTerminal.pointEqualityRows domain.laneVariables +
            KSplitNcFeTerminal.pointEqualityRows shape.rowVariables
        omega)
    · intro coordinate
      exact carried_mono (pointLaneBelow coordinate) (by
        change input.frameBase ≤
          (KSplitNcFeTerminal.carriedLaneEqualityInput input).frameBase
        have := base_le_equalityBase input
        change input.frameBase ≤
          KSplitNcFeTerminal.equalityBase input +
            KSplitNcFeTerminal.pointEqualityRows domain.laneVariables +
            KSplitNcFeTerminal.pointEqualityRows shape.rowVariables
        omega)
    · intro coordinate
      exact carried_mono (alphaBelow coordinate) (by
        change input.frameBase ≤
          (KSplitNcFeTerminal.carriedLaneEqualityInput input).frameBase
        have := base_le_equalityBase input
        change input.frameBase ≤
          KSplitNcFeTerminal.equalityBase input +
            KSplitNcFeTerminal.pointEqualityRows domain.laneVariables +
            KSplitNcFeTerminal.pointEqualityRows shape.rowVariables
        omega)

private theorem carriedLanePrefix_below_carriedRowBase
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (positive : 0 < input.frameBase)
    (gammaBelow : CarriedBelow input.gamma input.frameBase)
    (pointLaneBelow :
      ∀ coordinate,
        CarriedBelow (input.pointLane coordinate) input.frameBase)
    (pointRowBelow :
      ∀ coordinate,
        CarriedBelow (input.pointRow coordinate) input.frameBase)
    (betaABelow :
      ∀ coordinate,
        CarriedBelow (input.betaA coordinate) input.frameBase)
    (betaRBelow :
      ∀ coordinate,
        CarriedBelow (input.betaR coordinate) input.frameBase)
    (alphaBelow :
      ∀ coordinate,
        CarriedBelow (input.alpha coordinate) input.frameBase)
    (messageBelow :
      ∀ source matrix lane,
        CarriedBelow
          (input.messageYRing source matrix lane) input.frameBase) :
    RowsBelow (carriedLanePrefix input)
      (KSplitNcFeTerminal.carriedRowEqualityInput input).frameBase := by
  have prefixBelow :
      RowsBelow (freshRowPrefix input)
        (KSplitNcFeTerminal.carriedRowEqualityInput input).frameBase :=
    rowsBelow_mono
      (freshRowPrefix_below_carriedLaneBase input positive gammaBelow
        pointLaneBelow pointRowBelow betaABelow betaRBelow messageBelow)
      (by
        change
          KSplitNcFeTerminal.equalityBase input +
              KSplitNcFeTerminal.pointEqualityRows domain.laneVariables +
              KSplitNcFeTerminal.pointEqualityRows shape.rowVariables ≤
            KSplitNcFeTerminal.equalityBase input +
              2 * KSplitNcFeTerminal.pointEqualityRows domain.laneVariables +
              KSplitNcFeTerminal.pointEqualityRows shape.rowVariables
        omega)
  have equalityBelow :
      RowsBelow
        (KPointEquality.rows
          (KSplitNcFeTerminal.carriedLaneEqualityInput input))
        (KSplitNcFeTerminal.carriedRowEqualityInput input).frameBase := by
    apply pointRows_below
    · exact Nat.lt_of_lt_of_le positive (by
        change input.frameBase ≤
          (KSplitNcFeTerminal.carriedRowEqualityInput input).frameBase
        have := base_le_equalityBase input
        change input.frameBase ≤
          KSplitNcFeTerminal.equalityBase input +
            2 * KSplitNcFeTerminal.pointEqualityRows domain.laneVariables +
            KSplitNcFeTerminal.pointEqualityRows shape.rowVariables
        omega)
    · intro coordinate
      exact carried_mono (pointLaneBelow coordinate) (by
        change input.frameBase ≤
          (KSplitNcFeTerminal.carriedRowEqualityInput input).frameBase
        have := base_le_equalityBase input
        change input.frameBase ≤
          KSplitNcFeTerminal.equalityBase input +
            2 * KSplitNcFeTerminal.pointEqualityRows domain.laneVariables +
            KSplitNcFeTerminal.pointEqualityRows shape.rowVariables
        omega)
    · intro coordinate
      exact carried_mono (alphaBelow coordinate) (by
        change input.frameBase ≤
          (KSplitNcFeTerminal.carriedRowEqualityInput input).frameBase
        have := base_le_equalityBase input
        change input.frameBase ≤
          KSplitNcFeTerminal.equalityBase input +
            2 * KSplitNcFeTerminal.pointEqualityRows domain.laneVariables +
            KSplitNcFeTerminal.pointEqualityRows shape.rowVariables
        omega)
    · change
        KSplitNcFeTerminal.equalityBase input +
              KSplitNcFeTerminal.pointEqualityRows domain.laneVariables +
              KSplitNcFeTerminal.pointEqualityRows shape.rowVariables +
              KSplitNcFeTerminal.pointEqualityRows domain.laneVariables ≤
          KSplitNcFeTerminal.equalityBase input +
              2 * KSplitNcFeTerminal.pointEqualityRows domain.laneVariables +
              KSplitNcFeTerminal.pointEqualityRows shape.rowVariables
      omega
  exact rowsBelow_append prefixBelow equalityBelow

theorem pointPrefix_honest
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (assignment : Nat → Nat)
    (positive : 0 < input.frameBase)
    (constantWire : assignment 0 = 1)
    (gammaBelow : CarriedBelow input.gamma input.frameBase)
    (pointLaneBelow :
      ∀ coordinate,
        CarriedBelow (input.pointLane coordinate) input.frameBase)
    (pointRowBelow :
      ∀ coordinate,
        CarriedBelow (input.pointRow coordinate) input.frameBase)
    (betaABelow :
      ∀ coordinate,
        CarriedBelow (input.betaA coordinate) input.frameBase)
    (betaRBelow :
      ∀ coordinate,
        CarriedBelow (input.betaR coordinate) input.frameBase)
    (alphaBelow :
      ∀ coordinate,
        CarriedBelow (input.alpha coordinate) input.frameBase)
    (priorPointBelow :
      ∀ coordinate,
        CarriedBelow (input.priorPoint coordinate) input.frameBase)
    (messageBelow :
      ∀ source matrix lane,
        CarriedBelow
          (input.messageYRing source matrix lane) input.frameBase) :
    Satisfies (pointPrefix input) (afterCarriedRow input assignment) := by
  apply satisfies_after_append
      (carriedLanePrefix_honest input assignment positive constantWire
        gammaBelow pointLaneBelow pointRowBelow betaABelow betaRBelow
        alphaBelow messageBelow)
      (carriedLanePrefix_below_carriedRowBase input positive gammaBelow
        pointLaneBelow pointRowBelow betaABelow betaRBelow alphaBelow
        messageBelow)
  · intro column below
    exact KPointEqualityHonest.witness_off_block
      (afterCarriedLane input assignment)
      (KSplitNcFeTerminal.carriedRowEqualityInput input) column below
  · apply KPointEqualityHonest.rows_honest
    · exact Nat.lt_of_lt_of_le positive (by
        change input.frameBase ≤
          (KSplitNcFeTerminal.carriedRowEqualityInput input).frameBase
        have := base_le_equalityBase input
        change input.frameBase ≤
          KSplitNcFeTerminal.equalityBase input +
            2 * KSplitNcFeTerminal.pointEqualityRows domain.laneVariables +
            KSplitNcFeTerminal.pointEqualityRows shape.rowVariables
        omega)
    · intro coordinate
      exact carried_mono (pointRowBelow coordinate) (by
        change input.frameBase ≤
          (KSplitNcFeTerminal.carriedRowEqualityInput input).frameBase
        have := base_le_equalityBase input
        change input.frameBase ≤
          KSplitNcFeTerminal.equalityBase input +
            2 * KSplitNcFeTerminal.pointEqualityRows domain.laneVariables +
            KSplitNcFeTerminal.pointEqualityRows shape.rowVariables
        omega)
    · intro coordinate
      exact carried_mono (priorPointBelow coordinate) (by
        change input.frameBase ≤
          (KSplitNcFeTerminal.carriedRowEqualityInput input).frameBase
        have := base_le_equalityBase input
        change input.frameBase ≤
          KSplitNcFeTerminal.equalityBase input +
            2 * KSplitNcFeTerminal.pointEqualityRows domain.laneVariables +
            KSplitNcFeTerminal.pointEqualityRows shape.rowVariables
        omega)

theorem pointPrefix_below_productBase
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (positive : 0 < input.frameBase)
    (gammaBelow : CarriedBelow input.gamma input.frameBase)
    (pointLaneBelow :
      ∀ coordinate,
        CarriedBelow (input.pointLane coordinate) input.frameBase)
    (pointRowBelow :
      ∀ coordinate,
        CarriedBelow (input.pointRow coordinate) input.frameBase)
    (betaABelow :
      ∀ coordinate,
        CarriedBelow (input.betaA coordinate) input.frameBase)
    (betaRBelow :
      ∀ coordinate,
        CarriedBelow (input.betaR coordinate) input.frameBase)
    (alphaBelow :
      ∀ coordinate,
        CarriedBelow (input.alpha coordinate) input.frameBase)
    (priorPointBelow :
      ∀ coordinate,
        CarriedBelow (input.priorPoint coordinate) input.frameBase)
    (messageBelow :
      ∀ source matrix lane,
        CarriedBelow
          (input.messageYRing source matrix lane) input.frameBase) :
    RowsBelow (pointPrefix input)
      (KSplitNcFeTerminal.productBase input) := by
  have prefixBelow :
      RowsBelow (carriedLanePrefix input)
        (KSplitNcFeTerminal.productBase input) :=
    rowsBelow_mono
      (carriedLanePrefix_below_carriedRowBase input positive gammaBelow
        pointLaneBelow pointRowBelow betaABelow betaRBelow alphaBelow
        messageBelow)
      (by
        change
          KSplitNcFeTerminal.equalityBase input +
              2 * KSplitNcFeTerminal.pointEqualityRows domain.laneVariables +
              KSplitNcFeTerminal.pointEqualityRows shape.rowVariables ≤
            KSplitNcFeTerminal.equalityBase input +
              2 * KSplitNcFeTerminal.pointEqualityRows domain.laneVariables +
              2 * KSplitNcFeTerminal.pointEqualityRows shape.rowVariables
        omega)
  have equalityBelow :
      RowsBelow
        (KPointEquality.rows
          (KSplitNcFeTerminal.carriedRowEqualityInput input))
        (KSplitNcFeTerminal.productBase input) := by
    apply pointRows_below
    · exact Nat.lt_of_lt_of_le positive (by
        change input.frameBase ≤ KSplitNcFeTerminal.productBase input
        have := base_le_equalityBase input
        unfold KSplitNcFeTerminal.productBase
        omega)
    · intro coordinate
      exact carried_mono (pointRowBelow coordinate) (by
        change input.frameBase ≤ KSplitNcFeTerminal.productBase input
        have := base_le_equalityBase input
        unfold KSplitNcFeTerminal.productBase
        omega)
    · intro coordinate
      exact carried_mono (priorPointBelow coordinate) (by
        change input.frameBase ≤ KSplitNcFeTerminal.productBase input
        have := base_le_equalityBase input
        unfold KSplitNcFeTerminal.productBase
        omega)
    · change
        KSplitNcFeTerminal.equalityBase input +
              2 * KSplitNcFeTerminal.pointEqualityRows domain.laneVariables +
              KSplitNcFeTerminal.pointEqualityRows shape.rowVariables +
              KSplitNcFeTerminal.pointEqualityRows shape.rowVariables ≤
          KSplitNcFeTerminal.equalityBase input +
              2 * KSplitNcFeTerminal.pointEqualityRows domain.laneVariables +
              2 * KSplitNcFeTerminal.pointEqualityRows shape.rowVariables
      omega
  exact rowsBelow_append prefixBelow equalityBelow

end Nightstream.Implementation.R1CS.Canonical.KSplitNcFeTerminalHonest
