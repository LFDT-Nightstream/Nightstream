import Nightstream.Implementation.R1CS.Canonical.KBooleanMleCarriedPadded
import Nightstream.Implementation.R1CS.Canonical.KConcreteHorner
import Nightstream.Implementation.R1CS.Canonical.KEquality
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe

/-!
Contract: physical computation of the selected Split-NC FE initial scalar.

Every public running `yRing` claim is evaluated at the transcript-derived
lane point by a canonical Boolean-MLE program.  The resulting values are then
placed in the exact constant-first gamma vector:

* one leading zero block of width `sourceCount`;
* for each matrix, `freshCount` zero slots followed by all running values.

The dense Horner chain therefore implements the verifier's
`gamma^sourceCount` outer shift and the exact
`freshCount + running + matrix * sourceCount` inner exponents without caller
supplied powers or an exponent list.

Only the 54 live lanes are physical inputs.  Padded lanes are the canonical
row-free zero from `KBooleanMleCarriedPadded`.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KSplitNcFeInitial

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

private abbrev concreteOps := ConcreteCarrier.extensionOps

/-- Physical inputs of the FE-initial computation.  The gamma and lane-point
coordinates will be supplied by the selected transcript replay; the claimed
lanes will be supplied by the authoritative call-frame decoder. -/
structure Input
    (shape : SemanticShape) (domain : FlatNcDomain) where
  gamma : Carried
  alpha : Fin domain.laneVariables → Carried
  claimedYRing :
    Fin shape.runningCount → Fin shape.matrixCount →
      Fin ringDegree → Carried
  initial : Carried
  frameBase : Nat

abbrev Index (shape : SemanticShape) :=
  Fin shape.matrixCount × Fin shape.runningCount

private theorem sum_map_const
    {α : Type} (items : List α) (value : Nat) :
    (items.map fun _ => value).sum = value * items.length := by
  induction items with
  | nil => rfl
  | cons item rest inductionHypothesis =>
      simp only [List.map_cons, List.sum_cons, List.length_cons,
        inductionHypothesis, Nat.mul_succ]
      omega

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

def indices (shape : SemanticShape) : List (Index shape) :=
  (canonicalFinIndices shape.matrixCount).flatMap fun matrix =>
    (canonicalFinIndices shape.runningCount).map fun running =>
      (matrix, running)

theorem indices_length (shape : SemanticShape) :
    (indices shape).length = shape.matrixCount * shape.runningCount := by
  unfold indices
  rw [List.length_flatMap]
  simp only [List.length_map, canonicalFinIndices_length]
  rw [sum_map_const, canonicalFinIndices_length, Nat.mul_comm]

def ordinal
    {shape : SemanticShape} (index : Index shape) : Nat :=
  index.1.val * shape.runningCount + index.2.val

def rowsPerMle (domain : FlatNcDomain) : Nat :=
  3 * KBooleanMle.frameCount domain.laneVariables

def mleBase
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : Input shape domain) (index : Index shape) : Nat :=
  input.frameBase + rowsPerMle domain * ordinal index

def alphaCoordinates
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : Input shape domain) : List Carried :=
  List.ofFn input.alpha

@[simp] theorem alphaCoordinates_length
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : Input shape domain) :
    (alphaCoordinates input).length = domain.laneVariables := by
  simp [alphaCoordinates]

def table
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : Input shape domain) (index : Index shape) :
    BooleanTable Carried domain.laneVariables :=
  KBooleanMleCarriedPadded.carriedTable
    (fun lane => input.claimedYRing index.2 index.1 lane)

def mleRows
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : Input shape domain) : List Row :=
  (indices shape).flatMap fun index =>
    KBooleanMle.rows (KFrames.frameAt (mleBase input index))
      (table input index) (alphaCoordinates input) 0

def mleOutput
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : Input shape domain) (index : Index shape) : Carried :=
  KBooleanMle.carried (KFrames.frameAt (mleBase input index))
    (table input index) (alphaCoordinates input) 0

def matrixBlock
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : Input shape domain) (matrix : Fin shape.matrixCount) :
    List Carried :=
  List.replicate shape.freshCount KLinear.zeroCarried ++
    (canonicalFinIndices shape.runningCount).map fun running =>
      mleOutput input (matrix, running)

def coefficients
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : Input shape domain) : List Carried :=
  List.replicate shape.sourceCount KLinear.zeroCarried ++
    (canonicalFinIndices shape.matrixCount).flatMap (matrixBlock input)

theorem matrixBlock_length
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : Input shape domain) (matrix : Fin shape.matrixCount) :
    (matrixBlock input matrix).length = shape.sourceCount := by
  rw [matrixBlock, List.length_append, List.length_replicate,
    List.length_map, canonicalFinIndices_length]
  rfl

theorem coefficients_length
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : Input shape domain) :
    (coefficients input).length =
      (shape.matrixCount + 1) * shape.sourceCount := by
  unfold coefficients
  rw [List.length_append, List.length_replicate, List.length_flatMap]
  simp only [matrixBlock_length]
  rw [sum_map_const, canonicalFinIndices_length]
  rw [Nat.add_mul, Nat.one_mul,
    Nat.mul_comm shape.matrixCount shape.sourceCount]
  exact Nat.add_comm _ _

/-- Exact contiguous auxiliary width.  The terminal `K` equality emits two
rows but allocates no column, so this is intentionally two smaller than
`rows input).length`. -/
def allocationWidth
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : Input shape domain) : Nat :=
  rowsPerMle domain * (shape.matrixCount * shape.runningCount) +
    3 * ((shape.matrixCount + 1) * shape.sourceCount - 1)

def hornerBase
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : Input shape domain) : Nat :=
  input.frameBase +
    rowsPerMle domain * (shape.matrixCount * shape.runningCount)

def evaluated
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : Input shape domain) : Carried :=
  hornerCarried input.gamma (KFrames.frameAt (hornerBase input))
    (coefficients input) 0

def hornerRows
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : Input shape domain) : List Row :=
  KHorner.hornerRows input.gamma (KFrames.frameAt (hornerBase input))
    (coefficients input) 0

def rows
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : Input shape domain) : List Row :=
  mleRows input ++ hornerRows input ++
    KEquality.rows (evaluated input) input.initial

private theorem constantFlatMapLength
    {α β : Type} (items : List α) (program : α → List β) (count : Nat)
    (each : ∀ item ∈ items, (program item).length = count) :
    (items.flatMap program).length = count * items.length := by
  induction items with
  | nil => rfl
  | cons item rest inductionHypothesis =>
      rw [List.flatMap_cons, List.length_append, each item (by simp)]
      have tailEach : ∀ value ∈ rest, (program value).length = count := by
        intro value member
        exact each value (by simp [member])
      rw [inductionHypothesis tailEach]
      simp only [List.length_cons, Nat.mul_succ]
      omega

theorem mleRows_length
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : Input shape domain) :
    (mleRows input).length =
      rowsPerMle domain * (shape.matrixCount * shape.runningCount) := by
  unfold mleRows
  rw [constantFlatMapLength _ _ _ (fun index _ => by
    rw [KBooleanMle.rows_length]), indices_length]
  rfl

theorem rows_length
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : Input shape domain) :
    (rows input).length =
      rowsPerMle domain * (shape.matrixCount * shape.runningCount) +
        3 * ((shape.matrixCount + 1) * shape.sourceCount - 1) + 2 := by
  unfold rows hornerRows
  rw [List.length_append, List.length_append, mleRows_length,
    KHorner.hornerRows_length, coefficients_length, KEquality.rows_length]

theorem rows_length_eq_allocationWidth_add_two
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : Input shape domain) :
    (rows input).length = allocationWidth input + 2 := by
  rw [rows_length]
  rfl

/-- Exact ordered auxiliary interval used by the MLE and Horner frames. -/
def columns
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : Input shape domain) : List Nat :=
  (List.range (allocationWidth input)).map
    (fun offset => input.frameBase + offset)

theorem columns_length
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : Input shape domain) :
    (columns input).length = allocationWidth input := by
  simp [columns]

theorem columns_nodup
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : Input shape domain) :
    (columns input).Nodup := by
  unfold columns
  exact LinCombNormal.nodup_map _ _ (fun left right equal => by omega)
    List.nodup_range

def decoded (assignment : Nat → Nat) (value : Carried) : K :=
  KBooleanMleSemantics.decodeCarried assignment value

def decodedAlpha
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : Input shape domain) (assignment : Nat → Nat) :
    CubePoint K domain.laneVariables :=
  KBooleanMleSemantics.decodePoint assignment (alphaCoordinates input)
    (alphaCoordinates_length input)

theorem mleRows_satisfied
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : Input shape domain) (assignment : Nat → Nat)
    (satisfied : Satisfies (rows input) assignment)
    (index : Index shape) :
    Satisfies
      (KBooleanMle.rows (KFrames.frameAt (mleBase input index))
        (table input index) (alphaCoordinates input) 0)
      assignment := by
  have allMle : Satisfies (mleRows input) assignment :=
    fun row member =>
      satisfied row (List.mem_append_left _ (List.mem_append_left _ member))
  intro row member
  exact allMle row
    (List.mem_flatMap.2
      ⟨index, by
        rcases index with ⟨matrix, running⟩
        unfold indices
        apply List.mem_flatMap.2
        exact ⟨matrix, List.mem_ofFn.mpr ⟨matrix, rfl⟩,
          List.mem_map.2
            ⟨running, List.mem_ofFn.mpr ⟨running, rfl⟩, rfl⟩⟩,
        member⟩)

theorem mleOutput_sound
    {shape : SemanticShape} {domain : FlatNcDomain}
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (input : Input shape domain) (assignment : Nat → Nat)
    (satisfied : Satisfies (rows input) assignment)
    (index : Index shape) :
    decoded assignment (mleOutput input index) =
      Polynomial.Fe.paddedLaneEvaluation profile.laneCovers
        (fun lane =>
          decoded assignment
            (input.claimedYRing index.2 index.1 lane))
        (decodedAlpha input assignment) := by
  exact KBooleanMleCarriedPadded.rows_compute_paddedLaneEvaluation
    assignment (mleBase input index) profile.laneCovers
    (fun lane => input.claimedYRing index.2 index.1 lane)
    (alphaCoordinates input) (alphaCoordinates_length input)
    (mleRows_satisfied input assignment satisfied index)

def semanticMatrixBlock
    {shape : SemanticShape} {domain : FlatNcDomain}
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (input : Input shape domain) (assignment : Nat → Nat)
    (matrix : Fin shape.matrixCount) : List K :=
  List.replicate shape.freshCount K.zero ++
    (canonicalFinIndices shape.runningCount).map fun running =>
      Polynomial.Fe.paddedLaneEvaluation profile.laneCovers
        (fun lane =>
          decoded assignment (input.claimedYRing running matrix lane))
        (decodedAlpha input assignment)

def semanticCoefficients
    {shape : SemanticShape} {domain : FlatNcDomain}
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (input : Input shape domain) (assignment : Nat → Nat) : List K :=
  List.replicate shape.sourceCount K.zero ++
    (canonicalFinIndices shape.matrixCount).flatMap
      (semanticMatrixBlock profile input assignment)

theorem decoded_coefficients
    {shape : SemanticShape} {domain : FlatNcDomain}
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (input : Input shape domain) (assignment : Nat → Nat)
    (satisfied : Satisfies (rows input) assignment) :
    (coefficients input).map (decoded assignment) =
      semanticCoefficients profile input assignment := by
  unfold coefficients semanticCoefficients
  rw [List.map_append, List.map_replicate]
  have zeroDecoded :
      decoded assignment KLinear.zeroCarried = K.zero := by
    rfl
  rw [zeroDecoded, List.map_flatMap]
  congr 1
  apply flatMap_congr_local
  intro matrix matrixMember
  unfold matrixBlock semanticMatrixBlock
  rw [List.map_append, List.map_replicate, zeroDecoded, List.map_map]
  apply congrArg (fun tail =>
    List.replicate shape.freshCount K.zero ++ tail)
  apply List.map_congr_left
  intro running runningMember
  exact mleOutput_sound profile input assignment satisfied (matrix, running)

theorem horner_satisfied
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : Input shape domain) (assignment : Nat → Nat)
    (satisfied : Satisfies (rows input) assignment) :
    Satisfies (hornerRows input) assignment := by
  intro row member
  exact satisfied row
    (List.mem_append_left _
      (List.mem_append_right (mleRows input) member))

theorem evaluated_sound
    {shape : SemanticShape} {domain : FlatNcDomain}
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (input : Input shape domain) (assignment : Nat → Nat)
    (satisfied : Satisfies (rows input) assignment) :
    decoded assignment (evaluated input) =
      SumCheck.Finite.Message.evaluateCoefficients concreteOps.toOps
        (decoded assignment input.gamma)
      (semanticCoefficients profile input assignment) := by
  have computed :=
    KConcreteHorner.rows_sound assignment input.gamma
      (KFrames.frameAt (hornerBase input)) (coefficients input) 0
      (horner_satisfied input assignment satisfied)
  have decoderEqual :
      ∀ value,
        decoded assignment value =
          KConcreteHorner.decoded assignment value := by
    intro value
    rfl
  calc
    decoded assignment (evaluated input) =
        KConcreteHorner.decoded assignment (evaluated input) :=
      decoderEqual _
    _ = SumCheck.Finite.Message.evaluateCoefficients concreteOps.toOps
          (KConcreteHorner.decoded assignment input.gamma)
          ((coefficients input).map
            (KConcreteHorner.decoded assignment)) := by
      simpa only [evaluated] using computed
    _ = SumCheck.Finite.Message.evaluateCoefficients concreteOps.toOps
          (decoded assignment input.gamma)
          (semanticCoefficients profile input assignment) := by
      rw [← decoderEqual input.gamma]
      congr 1
      rw [← decoded_coefficients profile input assignment satisfied]
      apply List.map_congr_left
      intro coefficient _
      exact (decoderEqual coefficient).symm

theorem equality_satisfied
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : Input shape domain) (assignment : Nat → Nat)
    (satisfied : Satisfies (rows input) assignment) :
    Satisfies (KEquality.rows (evaluated input) input.initial) assignment := by
  intro row member
  exact satisfied row
    (List.mem_append_right _ member)

theorem initial_eq_evaluated
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : Input shape domain) (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows input) assignment) :
    decoded assignment input.initial =
      decoded assignment (evaluated input) := by
  have bound :=
    KEquality.rows_sound assignment (evaluated input) input.initial
      constantWire (equality_satisfied input assignment satisfied)
  apply KConcreteBridge.ofConcrete_injective
  have reversed := And.intro bound.1.symm bound.2.symm
  simpa [decoded, KBooleanMleSemantics.ofConcrete_decodeCarried,
    KHorner.carriedValue, KHorner.Pair.mk.injEq] using reversed

private def shiftLaws :
    TargetPolynomial.ShiftLaws concreteOps.toOps where
  one_mul := ConcreteCarrier.extensionLaws.one_mul
  mul_assoc := ConcreteCarrier.extensionLaws.mul_assoc
  mul_zero := ConcreteCarrier.extensionLaws.mul_zero
  mul_add := ConcreteCarrier.extensionLaws.left_distrib

private theorem evaluate_zeros (gamma : K) :
    ∀ count,
      SumCheck.Finite.Message.evaluateCoefficients concreteOps.toOps gamma
          (List.replicate count K.zero) =
        K.zero
  | 0 => rfl
  | count + 1 => by
      simp only [List.replicate_succ,
        SumCheck.Finite.Message.evaluateCoefficients]
      rw [evaluate_zeros gamma count]
      change
        ConcreteCarrier.extensionOps.add K.zero
            (ConcreteCarrier.extensionOps.mul gamma K.zero) =
          K.zero
      calc
        ConcreteCarrier.extensionOps.add K.zero
            (ConcreteCarrier.extensionOps.mul gamma K.zero) =
            ConcreteCarrier.extensionOps.add K.zero K.zero := by
          apply congrArg (ConcreteCarrier.extensionOps.add K.zero)
          exact ConcreteCarrier.extensionLaws.mul_zero gamma
        _ = K.zero := ConcreteCarrier.extensionLaws.zero_add K.zero

/-- Multiplying a canonical gamma sum by one verifier-owned power shifts every
explicit exponent. -/
private theorem gamma_shift_sum
    {Index : Type}
    (gamma : K) (offset : Nat) (items : List Index)
    (position : Index → Nat) (value : Index → K) :
    K.mul (TargetPolynomial.power concreteOps.toOps gamma offset)
        (FiniteSumAlgebra.sumMap concreteOps items fun item =>
          SignedJointIdentity.gammaTerm concreteOps gamma
            (position item) (value item)) =
      FiniteSumAlgebra.sumMap concreteOps items fun item =>
        SignedJointIdentity.gammaTerm concreteOps gamma
          (offset + position item) (value item) := by
  calc
    K.mul (TargetPolynomial.power concreteOps.toOps gamma offset)
        (FiniteSumAlgebra.sumMap concreteOps items fun item =>
          SignedJointIdentity.gammaTerm concreteOps gamma
            (position item) (value item)) =
        FiniteSumAlgebra.sumMap concreteOps items fun item =>
          K.mul (TargetPolynomial.power concreteOps.toOps gamma offset)
            (SignedJointIdentity.gammaTerm concreteOps gamma
              (position item) (value item)) :=
      (FiniteSumAlgebra.sumMap_mul_left concreteOps
        ConcreteCarrier.extensionLaws _ _ _).symm
    _ = _ := by
      apply FiniteSumAlgebra.sumMap_congr
      intro item member
      unfold SignedJointIdentity.gammaTerm
      calc
        K.mul (TargetPolynomial.power concreteOps.toOps gamma offset)
            (K.mul
              (TargetPolynomial.power concreteOps.toOps gamma (position item))
              (value item)) =
            K.mul
              (K.mul
                (TargetPolynomial.power concreteOps.toOps gamma offset)
                (TargetPolynomial.power concreteOps.toOps gamma
                  (position item)))
              (value item) :=
          (ConcreteCarrier.extensionLaws.mul_assoc _ _ _).symm
        _ = K.mul
              (TargetPolynomial.power concreteOps.toOps gamma
                (offset + position item))
              (value item) := by
          apply congrArg (fun power => K.mul power (value item))
          exact
            (TargetPolynomial.power_add concreteOps.toOps shiftLaws
              gamma offset (position item)).symm

/-- Exponentiating a verifier-owned power is exponent multiplication. -/
private theorem power_power (gamma : K) (width : Nat) :
    ∀ count,
      TargetPolynomial.power concreteOps.toOps
          (TargetPolynomial.power concreteOps.toOps gamma width) count =
        TargetPolynomial.power concreteOps.toOps gamma (width * count)
  | 0 => rfl
  | count + 1 => by
      simp only [TargetPolynomial.power, Nat.mul_succ]
      rw [power_power gamma width count,
        TargetPolynomial.power_add concreteOps.toOps shiftLaws]
      exact ConcreteCarrier.extensionLaws.mul_comm _ _

/-- Flattening equal-width coefficient blocks is Horner evaluation of their
individual evaluations at `gamma^width`. -/
private theorem evaluate_flatten
    (gamma : K) (width : Nat) :
    ∀ blocks : List (List K),
      (∀ block ∈ blocks, block.length = width) →
      SumCheck.Finite.Message.evaluateCoefficients concreteOps.toOps gamma
          blocks.flatten =
        SumCheck.Finite.Message.evaluateCoefficients concreteOps.toOps
          (TargetPolynomial.power concreteOps.toOps gamma width)
          (blocks.map fun block =>
            SumCheck.Finite.Message.evaluateCoefficients
              concreteOps.toOps gamma block)
  | [], _ => rfl
  | block :: blocks, sized => by
      have blockSized := sized block (by simp)
      have tailSized : ∀ tail ∈ blocks, tail.length = width :=
        fun tail member => sized tail (by simp [member])
      simp only [List.flatten_cons, List.map_cons,
        SumCheck.Finite.Message.evaluateCoefficients]
      rw [SignedCoefficientPolynomial.evaluate_append concreteOps
        ConcreteCarrier.extensionLaws, blockSized,
        evaluate_flatten gamma width blocks tailSized]

private theorem semanticMatrixBlock_evaluate
    {shape : SemanticShape} {domain : FlatNcDomain}
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (input : Input shape domain) (assignment : Nat → Nat)
    (matrix : Fin shape.matrixCount) :
    SumCheck.Finite.Message.evaluateCoefficients concreteOps.toOps
        (decoded assignment input.gamma)
        (semanticMatrixBlock profile input assignment matrix) =
      FiniteSumAlgebra.sumMap concreteOps
        (canonicalFinIndices shape.runningCount) fun running =>
          SignedJointIdentity.gammaTerm concreteOps
            (decoded assignment input.gamma)
            (shape.freshCount + running.val)
            (Polynomial.Fe.paddedLaneEvaluation profile.laneCovers
              (fun lane =>
                decoded assignment
                  (input.claimedYRing running matrix lane))
              (decodedAlpha input assignment)) := by
  unfold semanticMatrixBlock
  rw [SignedCoefficientPolynomial.evaluate_append concreteOps
    ConcreteCarrier.extensionLaws,
    evaluate_zeros]
  change
    K.add K.zero
        (K.mul
          (TargetPolynomial.power concreteOps.toOps
            (decoded assignment input.gamma)
            (List.replicate shape.freshCount K.zero).length)
          (SumCheck.Finite.Message.evaluateCoefficients concreteOps.toOps
            (decoded assignment input.gamma)
            ((canonicalFinIndices shape.runningCount).map fun running =>
              Polynomial.Fe.paddedLaneEvaluation profile.laneCovers
                (fun lane =>
                  decoded assignment
                    (input.claimedYRing running matrix lane))
                (decodedAlpha input assignment)))) =
      _
  rw [List.length_replicate,
    SignedCoefficientPolynomial.evaluate_canonicalFinMap_eq_gammaSum
      concreteOps ConcreteCarrier.extensionLaws]
  calc
    K.add K.zero
        (K.mul
          (TargetPolynomial.power concreteOps.toOps
            (decoded assignment input.gamma) shape.freshCount)
          (FiniteSumAlgebra.sumMap concreteOps
            (canonicalFinIndices shape.runningCount) fun running =>
              SignedJointIdentity.gammaTerm concreteOps
                (decoded assignment input.gamma) running.val
                (Polynomial.Fe.paddedLaneEvaluation profile.laneCovers
                  (fun lane =>
                    decoded assignment
                      (input.claimedYRing running matrix lane))
                  (decodedAlpha input assignment)))) =
        K.mul
          (TargetPolynomial.power concreteOps.toOps
            (decoded assignment input.gamma) shape.freshCount)
          (FiniteSumAlgebra.sumMap concreteOps
            (canonicalFinIndices shape.runningCount) fun running =>
              SignedJointIdentity.gammaTerm concreteOps
                (decoded assignment input.gamma) running.val
                (Polynomial.Fe.paddedLaneEvaluation profile.laneCovers
                  (fun lane =>
                    decoded assignment
                      (input.claimedYRing running matrix lane))
                  (decodedAlpha input assignment))) :=
      ConcreteCarrier.extensionLaws.zero_add _
    _ = _ := gamma_shift_sum
      (decoded assignment input.gamma) shape.freshCount
      (canonicalFinIndices shape.runningCount)
      (fun running => running.val)
      (fun running =>
        Polynomial.Fe.paddedLaneEvaluation profile.laneCovers
          (fun lane =>
            decoded assignment (input.claimedYRing running matrix lane))
          (decodedAlpha input assignment))

private theorem shifted_matrix_evaluate
    {shape : SemanticShape} {domain : FlatNcDomain}
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (input : Input shape domain) (assignment : Nat → Nat)
    (matrix : Fin shape.matrixCount) :
    K.mul
        (TargetPolynomial.power concreteOps.toOps
          (TargetPolynomial.power concreteOps.toOps
            (decoded assignment input.gamma) shape.sourceCount)
          matrix.val)
        (SumCheck.Finite.Message.evaluateCoefficients concreteOps.toOps
          (decoded assignment input.gamma)
          (semanticMatrixBlock profile input assignment matrix)) =
      FiniteSumAlgebra.sumMap concreteOps
        (canonicalFinIndices shape.runningCount) fun running =>
          SignedJointIdentity.gammaTerm concreteOps
            (decoded assignment input.gamma)
            (Polynomial.Fe.carriedGammaExponent shape running matrix)
            (Polynomial.Fe.paddedLaneEvaluation profile.laneCovers
              (fun lane =>
                decoded assignment
                  (input.claimedYRing running matrix lane))
              (decodedAlpha input assignment)) := by
  rw [power_power, semanticMatrixBlock_evaluate]
  rw [gamma_shift_sum
    (decoded assignment input.gamma)
    (shape.sourceCount * matrix.val)
    (canonicalFinIndices shape.runningCount)
    (fun running => shape.freshCount + running.val)
    (fun running =>
      Polynomial.Fe.paddedLaneEvaluation profile.laneCovers
        (fun lane =>
          decoded assignment (input.claimedYRing running matrix lane))
        (decodedAlpha input assignment))]
  apply FiniteSumAlgebra.sumMap_congr
  intro running member
  unfold Polynomial.Fe.carriedGammaExponent
  rw [Nat.mul_comm shape.sourceCount matrix.val]
  apply congrArg (fun exponent =>
    SignedJointIdentity.gammaTerm concreteOps
      (decoded assignment input.gamma) exponent
      (Polynomial.Fe.paddedLaneEvaluation profile.laneCovers
        (fun lane =>
          decoded assignment (input.claimedYRing running matrix lane))
        (decodedAlpha input assignment)))
  exact Nat.add_comm _ _

def decodedCoins
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : Input shape domain) (assignment : Nat → Nat) :
    Polynomial.Fe.Coins shape domain where
  alpha := decodedAlpha input assignment
  betaA := decodedAlpha input assignment
  betaR := {
    coordinates := List.replicate shape.rowVariables K.zero
    dimension := by simp
  }
  gamma := decoded assignment input.gamma

/-- The exact dense coefficient vector evaluates to the unchanged verifier
definition of the FE initial claim.  The only source bridge is pointwise
decoding of the authoritative running claims; the enclosing call-frame
decoder must construct it rather than the caller of the headline theorem. -/
theorem semanticCoefficients_evaluate
    {shape : SemanticShape} {domain : FlatNcDomain}
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (polynomialInput : PublicInput shape)
    (input : Input shape domain) (assignment : Nat → Nat)
    (claimedBound :
      ∀ running matrix lane,
        decoded assignment (input.claimedYRing running matrix lane) =
          polynomialInput.claimedYRing running matrix lane) :
    SumCheck.Finite.Message.evaluateCoefficients concreteOps.toOps
        (decoded assignment input.gamma)
        (semanticCoefficients profile input assignment) =
      Polynomial.Fe.initial profile polynomialInput
        (decodedCoins input assignment) := by
  unfold semanticCoefficients
  rw [SignedCoefficientPolynomial.evaluate_append concreteOps
    ConcreteCarrier.extensionLaws,
    evaluate_zeros]
  change
    K.add K.zero
        (K.mul
          (TargetPolynomial.power concreteOps.toOps
            (decoded assignment input.gamma)
            (List.replicate shape.sourceCount K.zero).length)
          (SumCheck.Finite.Message.evaluateCoefficients concreteOps.toOps
            (decoded assignment input.gamma)
            ((canonicalFinIndices shape.matrixCount).flatMap
              (semanticMatrixBlock profile input assignment)))) =
      _
  rw [List.length_replicate]
  refine (ConcreteCarrier.extensionLaws.zero_add _).trans ?_
  have flattened :
      (canonicalFinIndices shape.matrixCount).flatMap
          (semanticMatrixBlock profile input assignment) =
        ((canonicalFinIndices shape.matrixCount).map
          (semanticMatrixBlock profile input assignment)).flatten := by
    rfl
  rw [flattened,
    evaluate_flatten (decoded assignment input.gamma) shape.sourceCount]
  · rw [
      List.map_map,
      SignedCoefficientPolynomial.evaluate_canonicalFinMap_eq_gammaSum
        concreteOps ConcreteCarrier.extensionLaws]
    unfold Polynomial.Fe.initial
    apply congrArg
      (K.mul
        (TargetPolynomial.power concreteOps.toOps
          (decoded assignment input.gamma) shape.sourceCount))
    apply FiniteSumAlgebra.sumMap_congr
    intro matrix member
    calc
      K.mul
          (TargetPolynomial.power concreteOps.toOps
            (TargetPolynomial.power concreteOps.toOps
              (decoded assignment input.gamma) shape.sourceCount)
            matrix.val)
          (SumCheck.Finite.Message.evaluateCoefficients concreteOps.toOps
            (decoded assignment input.gamma)
            (semanticMatrixBlock profile input assignment matrix)) =
          FiniteSumAlgebra.sumMap concreteOps
            (canonicalFinIndices shape.runningCount) (fun running =>
              SignedJointIdentity.gammaTerm concreteOps
                (decoded assignment input.gamma)
                (Polynomial.Fe.carriedGammaExponent shape running matrix)
                (Polynomial.Fe.paddedLaneEvaluation profile.laneCovers
                  (fun lane =>
                    decoded assignment
                      (input.claimedYRing running matrix lane))
                  (decodedAlpha input assignment))) :=
        shifted_matrix_evaluate profile input assignment matrix
      _ = _ := by
        apply FiniteSumAlgebra.sumMap_congr
        intro running runningMember
        apply congrArg
          (SignedJointIdentity.gammaTerm concreteOps
            (decoded assignment input.gamma)
            (Polynomial.Fe.carriedGammaExponent shape running matrix))
        unfold decodedCoins
        apply congrArg
          (fun values =>
            Polynomial.Fe.paddedLaneEvaluation profile.laneCovers
              values (decodedAlpha input assignment))
        funext lane
        exact claimedBound running matrix lane
  · intro block member
    rcases List.mem_map.1 member with ⟨matrix, matrixMember, rfl⟩
    unfold semanticMatrixBlock
    rw [List.length_append, List.length_replicate, List.length_map,
      canonicalFinIndices_length]
    rfl

/-- Satisfying FE-initial rows derive the first `EndpointAgrees` equation;
the equation is not an input to the row program. -/
theorem rows_sound
    {shape : SemanticShape} {domain : FlatNcDomain}
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (polynomialInput : PublicInput shape)
    (input : Input shape domain) (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (claimedBound :
      ∀ running matrix lane,
        decoded assignment (input.claimedYRing running matrix lane) =
          polynomialInput.claimedYRing running matrix lane)
    (satisfied : Satisfies (rows input) assignment) :
    decoded assignment input.initial =
      Polynomial.Fe.initial profile polynomialInput
        (decodedCoins input assignment) := by
  rw [initial_eq_evaluated input assignment constantWire satisfied,
    evaluated_sound profile input assignment satisfied]
  exact semanticCoefficients_evaluate
    profile polynomialInput input assignment claimedBound

end Nightstream.Implementation.R1CS.Canonical.KSplitNcFeInitial
