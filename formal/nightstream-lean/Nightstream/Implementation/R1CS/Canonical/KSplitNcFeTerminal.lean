import Nightstream.Implementation.R1CS.Canonical.KSplitNcFeInitial
import Nightstream.Implementation.R1CS.Canonical.KSparsePolynomial
import Nightstream.Implementation.R1CS.Canonical.KPointEquality

/-!
Contract: physical computation of the selected Split-NC FE terminal scalar.

The program emits exactly the arithmetic in `Polynomial.Fe.terminalFromMessage`:

* one sparse lifted-CCS evaluation for every fresh source;
* one dense gamma fold of those fresh evaluations;
* the proved zero-padded lane-MLE/gamma program for every running claim;
* the four row/lane equality selectors;
* the two selector products and two selected branch products; and
* a two-row binding to the SumCheck terminal columns.

The constraint polynomial is taken from the typed verifier input.  Challenge
coordinates, output-message lanes, prior-point coordinates and the terminal
are physical carried values.  This module derives a semantic message and
verifier input from those same values; the enclosing call-frame decoder later
proves that they are the selected authoritative values.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.KSplitNcFeTerminal

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

private abbrev concreteOps := ConcreteCarrier.extensionOps

/-- Physical values consumed by the FE terminal.  The verifier-owned sparse
polynomial is an index of the structure, so no row input can replace it. -/
structure Input
    {shape : SemanticShape}
    (polynomialInput : PublicInput shape)
    (domain : FlatNcDomain) where
  gamma : Carried
  alpha : Fin domain.laneVariables → Carried
  betaA : Fin domain.laneVariables → Carried
  betaR : Fin shape.rowVariables → Carried
  pointLane : Fin domain.laneVariables → Carried
  pointRow : Fin shape.rowVariables → Carried
  priorPoint : Fin shape.rowVariables → Carried
  messageYRing :
    Fin shape.sourceCount → Fin shape.matrixCount →
      Fin ringDegree → Carried
  terminal : Carried
  frameBase : Nat

def polynomialDegreeSum
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain) : Nat :=
  KSparsePolynomial.totalDegreeSum
    (Polynomial.Fe.liftedConstraintPolynomial
      polynomialInput).terms

def sparseRowsPerFresh
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain) : Nat :=
  3 * polynomialDegreeSum input

def freshPolynomialInput
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain)
    (fresh : Fin shape.freshCount) :
    KSparsePolynomial.Input shape.matrixCount where
  polynomial := Polynomial.Fe.liftedConstraintPolynomial polynomialInput
  point := fun matrix =>
    input.messageYRing (Data.freshIndex fresh) matrix
      Phi81CoefficientKernel.constant
  frameBase := input.frameBase + fresh.val * sparseRowsPerFresh input

def freshRows
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain) : List Row :=
  (canonicalFinIndices shape.freshCount).flatMap fun fresh =>
    KSparsePolynomial.rows (freshPolynomialInput input fresh)

def freshOutputs
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain) : List Carried :=
  (canonicalFinIndices shape.freshCount).map fun fresh =>
    KSparsePolynomial.output (freshPolynomialInput input fresh)

def freshHornerBase
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain) : Nat :=
  input.frameBase + sparseRowsPerFresh input * shape.freshCount

def freshHornerRows
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain) : List Row :=
  KHorner.hornerRows input.gamma (KFrames.frameAt (freshHornerBase input))
    (freshOutputs input) 0

def freshOutput
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain) : Carried :=
  KHorner.hornerCarried input.gamma
    (KFrames.frameAt (freshHornerBase input)) (freshOutputs input) 0

def carriedBase
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain) : Nat :=
  freshHornerBase input + 3 * (shape.freshCount - 1)

def carriedInternalWidth
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain) : Nat :=
  KSplitNcFeInitial.rowsPerMle domain *
      (shape.matrixCount * shape.runningCount) +
    3 * ((shape.matrixCount + 1) * shape.sourceCount - 1)

def carriedTargetBase
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain) : Nat :=
  carriedBase input + carriedInternalWidth input

/-- Two columns materializing the shifted carried branch. -/
def carriedTarget
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain) : Carried :=
  {
    low := [(carriedTargetBase input, 1)]
    high := [(carriedTargetBase input + 1, 1)]
  }

def carriedInput
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain) :
    KSplitNcFeInitial.Input shape domain where
  gamma := input.gamma
  alpha := input.pointLane
  claimedYRing := fun running matrix lane =>
    input.messageYRing (Data.runningIndex running) matrix lane
  initial := carriedTarget input
  frameBase := carriedBase input

def carriedRows
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain) : List Row :=
  KSplitNcFeInitial.rows (carriedInput input)

def pointEqualityRows (variables : Nat) : Nat :=
  3 * variables + 3 * (variables - 1)

/-- Exact contiguous auxiliary width.  Equality rows bind carried values but
allocate nothing; the two carried-target coordinates are included explicitly. -/
def allocationWidth
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain) : Nat :=
  sparseRowsPerFresh input * shape.freshCount +
    3 * (shape.freshCount - 1) +
    carriedInternalWidth input + 2 +
    2 * pointEqualityRows domain.laneVariables +
    2 * pointEqualityRows shape.rowVariables + 12

def equalityBase
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain) : Nat :=
  carriedTargetBase input + 2

def freshLaneEqualityInput
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain) :
    KPointEquality.Input domain.laneVariables where
  left := input.pointLane
  right := input.betaA
  frameBase := equalityBase input

def freshRowEqualityInput
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain) :
    KPointEquality.Input shape.rowVariables where
  left := input.pointRow
  right := input.betaR
  frameBase := equalityBase input + pointEqualityRows domain.laneVariables

def carriedLaneEqualityInput
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain) :
    KPointEquality.Input domain.laneVariables where
  left := input.pointLane
  right := input.alpha
  frameBase :=
    equalityBase input + pointEqualityRows domain.laneVariables +
      pointEqualityRows shape.rowVariables

def carriedRowEqualityInput
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain) :
    KPointEquality.Input shape.rowVariables where
  left := input.pointRow
  right := input.priorPoint
  frameBase :=
    equalityBase input + 2 * pointEqualityRows domain.laneVariables +
      pointEqualityRows shape.rowVariables

def productBase
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain) : Nat :=
  equalityBase input +
    2 * pointEqualityRows domain.laneVariables +
    2 * pointEqualityRows shape.rowVariables

def freshSelectorFrame
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain) : Frame :=
  KFrames.frameAt (productBase input) 0

def freshContributionFrame
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain) : Frame :=
  KFrames.frameAt (productBase input) 1

def carriedSelectorFrame
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain) : Frame :=
  KFrames.frameAt (productBase input) 2

def carriedContributionFrame
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain) : Frame :=
  KFrames.frameAt (productBase input) 3

def freshSelector
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain) : Carried :=
  KMulChain.frameOutput (freshSelectorFrame input)

def freshContribution
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain) : Carried :=
  KMulChain.frameOutput (freshContributionFrame input)

def carriedSelector
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain) : Carried :=
  KMulChain.frameOutput (carriedSelectorFrame input)

def carriedContribution
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain) : Carried :=
  KMulChain.frameOutput (carriedContributionFrame input)

def terminalExpression
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain) : Carried :=
  KLinear.addCarried (freshContribution input) (carriedContribution input)

def rowGroups
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain) : List (List Row) :=
  [
    freshRows input,
    freshHornerRows input,
    carriedRows input,
    KPointEquality.rows (freshLaneEqualityInput input),
    KPointEquality.rows (freshRowEqualityInput input),
    KPointEquality.rows (carriedLaneEqualityInput input),
    KPointEquality.rows (carriedRowEqualityInput input),
    KMul.rows
      (KPointEquality.equalityCarried (freshLaneEqualityInput input))
      (KPointEquality.equalityCarried (freshRowEqualityInput input))
      (freshSelectorFrame input),
    KMul.rows (freshSelector input) (freshOutput input)
      (freshContributionFrame input),
    KMul.rows
      (KPointEquality.equalityCarried (carriedLaneEqualityInput input))
      (KPointEquality.equalityCarried (carriedRowEqualityInput input))
      (carriedSelectorFrame input),
    KMul.rows (carriedSelector input) (carriedTarget input)
      (carriedContributionFrame input),
    KEquality.rows (terminalExpression input) input.terminal
  ]

def rows
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain) : List Row :=
  (rowGroups input).flatten

private theorem constantFlatMapLength
    {α β : Type} (items : List α) (program : α → List β) (count : Nat)
    (each : ∀ item ∈ items, (program item).length = count) :
    (items.flatMap program).length = count * items.length := by
  induction items with
  | nil => rfl
  | cons item rest inductionHypothesis =>
      rw [List.flatMap_cons, List.length_append, each item (by simp)]
      have tailEach :
          ∀ value ∈ rest, (program value).length = count := by
        intro value member
        exact each value (by simp [member])
      rw [inductionHypothesis tailEach]
      simp only [List.length_cons, Nat.mul_succ]
      omega

theorem freshRows_length
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain) :
    (freshRows input).length =
      sparseRowsPerFresh input * shape.freshCount := by
  unfold freshRows
  calc
    ((canonicalFinIndices shape.freshCount).flatMap fun fresh =>
        KSparsePolynomial.rows (freshPolynomialInput input fresh)).length =
        sparseRowsPerFresh input *
          (canonicalFinIndices shape.freshCount).length := by
      apply constantFlatMapLength _ _ (sparseRowsPerFresh input)
      intro fresh _
      rw [KSparsePolynomial.rows_length]
      rfl
    _ = sparseRowsPerFresh input * shape.freshCount := by
      rw [canonicalFinIndices_length]

theorem freshOutputs_length
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain) :
    (freshOutputs input).length = shape.freshCount := by
  unfold freshOutputs
  rw [List.length_map, canonicalFinIndices_length]

theorem carriedRows_length
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain) :
    (carriedRows input).length =
      KSplitNcFeInitial.rowsPerMle domain *
          (shape.matrixCount * shape.runningCount) +
        3 * ((shape.matrixCount + 1) * shape.sourceCount - 1) + 2 := by
  unfold carriedRows
  exact KSplitNcFeInitial.rows_length (carriedInput input)

theorem rows_length
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain) :
    (rows input).length =
      sparseRowsPerFresh input * shape.freshCount +
        3 * (shape.freshCount - 1) +
        (KSplitNcFeInitial.rowsPerMle domain *
            (shape.matrixCount * shape.runningCount) +
          3 * ((shape.matrixCount + 1) * shape.sourceCount - 1) + 2) +
        2 * pointEqualityRows domain.laneVariables +
        2 * pointEqualityRows shape.rowVariables +
        14 := by
  unfold rows rowGroups freshHornerRows pointEqualityRows
  simp only [List.flatten_cons, List.flatten_nil, List.nil_append,
    List.length_append, List.length_nil, freshRows_length,
    KHorner.hornerRows_length, freshOutputs_length,
    carriedRows_length, KPointEquality.rows_length,
    KMul.rows_length, KEquality.rows_length]
  omega

theorem rows_length_eq_allocationWidth_add_two
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain) :
    (rows input).length = allocationWidth input + 2 := by
  rw [rows_length]
  unfold allocationWidth carriedInternalWidth pointEqualityRows
  omega

/-- Exact ordered auxiliary interval used by every endpoint sub-gadget. -/
def columns
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain) : List Nat :=
  (List.range (allocationWidth input)).map
    (fun offset => input.frameBase + offset)

theorem columns_length
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain) :
    (columns input).length = allocationWidth input := by
  simp [columns]

theorem columns_nodup
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain) :
    (columns input).Nodup := by
  unfold columns
  exact LinCombNormal.nodup_map _ _ (fun left right equal => by omega)
    List.nodup_range

theorem satisfies_group
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain)
    (assignment : Nat → Nat)
    (satisfied : Satisfies (rows input) assignment)
    (group : List Row) (member : group ∈ rowGroups input) :
    Satisfies group assignment := by
  intro row rowMember
  exact satisfied row (List.mem_flatten.2 ⟨group, member, rowMember⟩)

def decoded (assignment : Nat → Nat) (value : Carried) : K :=
  KPointEquality.decoded assignment value

def decodedMessage
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain)
    (assignment : Nat → Nat) : OutputMessage shape where
  yRing := fun source matrix lane =>
    decoded assignment (input.messageYRing source matrix lane)
  yZcol := fun _ _ => K.zero

def decodedPublicInput
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain)
    (assignment : Nat → Nat) : PublicInput shape where
  constraintPolynomial := polynomialInput.constraintPolynomial
  priorPoint :=
    KPointEquality.decodedRight
      (carriedRowEqualityInput input) assignment
  claimedYRing := polynomialInput.claimedYRing

def decodedCoins
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain)
    (assignment : Nat → Nat) :
    Polynomial.Fe.Coins shape domain where
  alpha :=
    KPointEquality.decodedRight
      (carriedLaneEqualityInput input) assignment
  betaA :=
    KPointEquality.decodedRight
      (freshLaneEqualityInput input) assignment
  betaR :=
    KPointEquality.decodedRight
      (freshRowEqualityInput input) assignment
  gamma := decoded assignment input.gamma

def decodedPoint
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain)
    (assignment : Nat → Nat) :
    Polynomial.Fe.Point shape domain where
  row :=
    KPointEquality.decodedLeft
      (freshRowEqualityInput input) assignment
  lane :=
    KPointEquality.decodedLeft
      (freshLaneEqualityInput input) assignment

theorem fresh_source_satisfied
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain)
    (assignment : Nat → Nat)
    (satisfied : Satisfies (rows input) assignment)
    (fresh : Fin shape.freshCount) :
    Satisfies
      (KSparsePolynomial.rows (freshPolynomialInput input fresh))
      assignment := by
  have group : Satisfies (freshRows input) assignment :=
    satisfies_group input assignment satisfied _ (by simp [rowGroups])
  intro row member
  exact group row
    (List.mem_flatMap.2
      ⟨fresh, List.mem_ofFn.mpr ⟨fresh, rfl⟩, member⟩)

theorem freshOutputs_decoded
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain)
    (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows input) assignment) :
    (freshOutputs input).map (decoded assignment) =
      (canonicalFinIndices shape.freshCount).map fun fresh =>
        CCSResidualTable.evaluatePolynomial concreteOps
          (Polynomial.Fe.liftedConstraintPolynomial polynomialInput)
          (fun matrix =>
            decoded assignment
              (input.messageYRing (Data.freshIndex fresh) matrix
                Phi81CoefficientKernel.constant)) := by
  unfold freshOutputs
  rw [List.map_map]
  apply List.map_congr_left
  intro fresh _
  exact KSparsePolynomial.rows_sound
    (freshPolynomialInput input fresh) assignment constantWire
    (fresh_source_satisfied input assignment satisfied fresh)

theorem freshOutput_sound
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain)
    (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows input) assignment) :
    decoded assignment (freshOutput input) =
      Polynomial.Fe.freshTermFromYRing
        (decodedPublicInput input assignment)
        (decoded assignment input.gamma)
        (decodedMessage input assignment).yRing := by
  have group : Satisfies (freshHornerRows input) assignment :=
    satisfies_group input assignment satisfied _ (by simp [rowGroups])
  have computed :=
    KConcreteHorner.rows_sound assignment input.gamma
      (KFrames.frameAt (freshHornerBase input))
      (freshOutputs input) 0 group
  rw [show decoded assignment (freshOutput input) =
      SumCheck.Finite.Message.evaluateCoefficients concreteOps.toOps
        (decoded assignment input.gamma)
        ((freshOutputs input).map (decoded assignment)) by
          simpa [freshOutput] using computed]
  rw [freshOutputs_decoded input assignment constantWire satisfied,
    SignedCoefficientPolynomial.evaluate_canonicalFinMap_eq_gammaSum
      concreteOps ConcreteCarrier.extensionLaws]
  rfl

private theorem decoder_eq
    (assignment : Nat → Nat) (value : Carried) :
    KSplitNcFeInitial.decoded assignment value =
      decoded assignment value := by
  unfold KSplitNcFeInitial.decoded decoded
  apply KConcreteBridge.ofConcrete_injective
  rw [KBooleanMleSemantics.ofConcrete_decodeCarried,
    KPointEquality.ofConcrete_decoded]

private theorem cubePoint_eq_of_coordinates_eq
    {variables : Nat}
    (left right : CubePoint K variables)
    (coordinates : left.coordinates = right.coordinates) :
    left = right := by
  cases left
  cases right
  simp_all

def carriedPublicInput
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain)
    (assignment : Nat → Nat) : PublicInput shape where
  constraintPolynomial := polynomialInput.constraintPolynomial
  priorPoint := polynomialInput.priorPoint
  claimedYRing := fun running matrix lane =>
    decoded assignment
      (input.messageYRing (Data.runningIndex running) matrix lane)

private theorem decodedAlpha_carriedInput
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain)
    (assignment : Nat → Nat) :
    KSplitNcFeInitial.decodedAlpha
        (carriedInput input) assignment =
      (decodedPoint input assignment).lane := by
  apply cubePoint_eq_of_coordinates_eq
  simp only [KSplitNcFeInitial.decodedAlpha,
    KSplitNcFeInitial.alphaCoordinates,
    KBooleanMleSemantics.decodePoint, decodedPoint,
    KPointEquality.decodedLeft, KPointEquality.indices]
  simp only [List.map_ofFn, Function.comp_apply]
  apply congrArg List.ofFn
  funext coordinate
  exact decoder_eq assignment (input.pointLane coordinate)

theorem carriedTarget_sound
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (input : Input polynomialInput domain)
    (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows input) assignment) :
    decoded assignment (carriedTarget input) =
      SignedJointIdentity.gammaTerm concreteOps
        (decoded assignment input.gamma) shape.sourceCount
        (Polynomial.Fe.carriedTermFromYRing profile.laneCovers
          (decoded assignment input.gamma)
          (decodedPoint input assignment).lane
          (decodedMessage input assignment).yRing) := by
  have group : Satisfies (carriedRows input) assignment :=
    satisfies_group input assignment satisfied _ (by simp [rowGroups])
  have computed :=
    KSplitNcFeInitial.rows_sound profile
      (carriedPublicInput input assignment)
      (carriedInput input) assignment constantWire
      (fun running matrix lane => decoder_eq assignment
        (input.messageYRing (Data.runningIndex running) matrix lane))
      group
  change
    KSplitNcFeInitial.decoded assignment (carriedTarget input) =
      _ at computed
  rw [decoder_eq assignment (carriedTarget input)] at computed
  rw [computed]
  unfold Polynomial.Fe.initial
  unfold Polynomial.Fe.carriedTermFromYRing
  unfold KSplitNcFeInitial.decodedCoins
  rw [decodedAlpha_carriedInput input assignment]
  rfl

private theorem decoded_mul
    (assignment : Nat → Nat)
    (left right : Carried) (frame : Frame)
    (satisfied : Satisfies (KMul.rows left right frame) assignment) :
    decoded assignment (KMulChain.frameOutput frame) =
      K.mul (decoded assignment left) (decoded assignment right) := by
  unfold decoded
  apply KConcreteBridge.ofConcrete_injective
  rw [KPointEquality.ofConcrete_decoded,
    KMulChain.frameOutput_sound assignment left right frame satisfied,
    KConcreteBridge.ofConcrete_mul,
    KPointEquality.ofConcrete_decoded,
    KPointEquality.ofConcrete_decoded]

private theorem terminal_eq_expression
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input polynomialInput domain)
    (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows input) assignment) :
    decoded assignment input.terminal =
      decoded assignment (terminalExpression input) := by
  have group :
      Satisfies (KEquality.rows (terminalExpression input) input.terminal)
        assignment :=
    satisfies_group input assignment satisfied _ (by simp [rowGroups])
  have bound :=
    KEquality.rows_sound assignment (terminalExpression input) input.terminal
      constantWire group
  unfold decoded
  apply KConcreteBridge.ofConcrete_injective
  rw [KPointEquality.ofConcrete_decoded,
    KPointEquality.ofConcrete_decoded]
  unfold carriedValue
  simp only [Pair.mk.injEq]
  exact And.intro bound.1.symm bound.2.symm

private theorem decoded_add
    (assignment : Nat → Nat) (left right : Carried) :
    decoded assignment (KLinear.addCarried left right) =
      K.add (decoded assignment left) (decoded assignment right) := by
  unfold decoded
  exact KSparsePolynomial.decoded_add assignment left right

/-- Satisfying FE-terminal rows compute exactly the unchanged operational
message terminal from values decoded out of the same assignment. -/
theorem rows_sound
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (input : Input polynomialInput domain)
    (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows input) assignment) :
    decoded assignment input.terminal =
      Polynomial.Fe.terminalFromMessage profile
        (decodedPublicInput input assignment)
        (decodedCoins input assignment)
        (decodedPoint input assignment)
        (decodedMessage input assignment) := by
  have freshLaneSatisfied :
      Satisfies (KPointEquality.rows (freshLaneEqualityInput input))
        assignment :=
    satisfies_group input assignment satisfied _ (by simp [rowGroups])
  have freshRowSatisfied :
      Satisfies (KPointEquality.rows (freshRowEqualityInput input))
        assignment :=
    satisfies_group input assignment satisfied _ (by simp [rowGroups])
  have carriedLaneSatisfied :
      Satisfies (KPointEquality.rows (carriedLaneEqualityInput input))
        assignment :=
    satisfies_group input assignment satisfied _ (by simp [rowGroups])
  have carriedRowSatisfied :
      Satisfies (KPointEquality.rows (carriedRowEqualityInput input))
        assignment :=
    satisfies_group input assignment satisfied _ (by simp [rowGroups])
  have freshSelectorSatisfied :
      Satisfies
        (KMul.rows
          (KPointEquality.equalityCarried (freshLaneEqualityInput input))
          (KPointEquality.equalityCarried (freshRowEqualityInput input))
          (freshSelectorFrame input)) assignment :=
    satisfies_group input assignment satisfied _ (by simp [rowGroups])
  have freshContributionSatisfied :
      Satisfies
        (KMul.rows (freshSelector input) (freshOutput input)
          (freshContributionFrame input)) assignment :=
    satisfies_group input assignment satisfied _ (by simp [rowGroups])
  have carriedSelectorSatisfied :
      Satisfies
        (KMul.rows
          (KPointEquality.equalityCarried (carriedLaneEqualityInput input))
          (KPointEquality.equalityCarried (carriedRowEqualityInput input))
          (carriedSelectorFrame input)) assignment :=
    satisfies_group input assignment satisfied _ (by simp [rowGroups])
  have carriedContributionSatisfied :
      Satisfies
        (KMul.rows (carriedSelector input) (carriedTarget input)
          (carriedContributionFrame input)) assignment :=
    satisfies_group input assignment satisfied _ (by simp [rowGroups])
  have freshLane :=
    KPointEquality.rows_sound (freshLaneEqualityInput input) assignment
      constantWire freshLaneSatisfied
  have freshRow :=
    KPointEquality.rows_sound (freshRowEqualityInput input) assignment
      constantWire freshRowSatisfied
  have carriedLane :=
    KPointEquality.rows_sound (carriedLaneEqualityInput input) assignment
      constantWire carriedLaneSatisfied
  have carriedRow :=
    KPointEquality.rows_sound (carriedRowEqualityInput input) assignment
      constantWire carriedRowSatisfied
  change
    decoded assignment
        (KPointEquality.equalityCarried (freshLaneEqualityInput input)) =
      _ at freshLane
  change
    decoded assignment
        (KPointEquality.equalityCarried (freshRowEqualityInput input)) =
      _ at freshRow
  change
    decoded assignment
        (KPointEquality.equalityCarried (carriedLaneEqualityInput input)) =
      _ at carriedLane
  change
    decoded assignment
        (KPointEquality.equalityCarried (carriedRowEqualityInput input)) =
      _ at carriedRow
  have freshSelectorValue :=
    decoded_mul assignment
      (KPointEquality.equalityCarried (freshLaneEqualityInput input))
      (KPointEquality.equalityCarried (freshRowEqualityInput input))
      (freshSelectorFrame input) freshSelectorSatisfied
  have freshContributionValue :=
    decoded_mul assignment (freshSelector input) (freshOutput input)
      (freshContributionFrame input) freshContributionSatisfied
  have carriedSelectorValue :=
    decoded_mul assignment
      (KPointEquality.equalityCarried (carriedLaneEqualityInput input))
      (KPointEquality.equalityCarried (carriedRowEqualityInput input))
      (carriedSelectorFrame input) carriedSelectorSatisfied
  have carriedContributionValue :=
    decoded_mul assignment (carriedSelector input) (carriedTarget input)
      (carriedContributionFrame input) carriedContributionSatisfied
  change decoded assignment (freshSelector input) = _ at freshSelectorValue
  change decoded assignment (carriedSelector input) = _ at carriedSelectorValue
  change
    decoded assignment (freshContribution input) = _
      at freshContributionValue
  change
    decoded assignment (carriedContribution input) = _
      at carriedContributionValue
  rw [terminal_eq_expression input assignment constantWire satisfied]
  unfold terminalExpression
  rw [decoded_add]
  rw [freshContributionValue,
    carriedContributionValue, freshSelectorValue, carriedSelectorValue,
    freshLane, freshRow, carriedLane, carriedRow,
    freshOutput_sound input assignment constantWire satisfied,
    carriedTarget_sound profile input assignment constantWire satisfied]
  rfl

end Nightstream.Implementation.R1CS.Canonical.KSplitNcFeTerminal
