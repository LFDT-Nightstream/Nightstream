import Nightstream.Implementation.R1CS.Canonical.KBooleanMleCarriedPadded
import Nightstream.Implementation.R1CS.Canonical.KStrictNorm
import Nightstream.Implementation.R1CS.Canonical.KConcreteHorner
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.InitialSum
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Terminal

/-!
Contract: physical computation of the selected block×lane NC endpoints.

The initial endpoint is bound to the verifier-owned zero.  The terminal
program zero-extends each source's 54 `yZcol` lanes, evaluates the resulting
Boolean MLE at the transcript-derived lane point, applies the strict `b = 2`
cubic, gamma-compresses sources in the paper-relative order, applies the
block/lane equality selectors, and binds the result to the SumCheck terminal.

No padded lane, gamma power, selector value, endpoint scalar, or acceptance
result is supplied as a semantic premise.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.KSplitNcNcEndpoint

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane

private abbrev concreteOps := ConcreteCarrier.extensionOps

structure Input
    (shape : SemanticShape) (domain : BlockNcDomain) where
  gamma : Carried
  betaBlock : Fin domain.blockVariables → Carried
  betaA : Fin domain.laneVariables → Carried
  pointBlock : Fin domain.blockVariables → Carried
  pointLane : Fin domain.laneVariables → Carried
  messageYZcol : Fin shape.sourceCount → Fin ringDegree → Carried
  initial : Carried
  terminal : Carried
  frameBase : Nat

def laneCoordinates
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) : List Carried :=
  List.ofFn input.pointLane

@[simp] theorem laneCoordinates_length
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) :
    (laneCoordinates input).length = domain.laneVariables := by
  simp [laneCoordinates]

def sourceTable
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) (source : Fin shape.sourceCount) :
    BooleanTable Carried domain.laneVariables :=
  KBooleanMleCarriedPadded.carriedTable (input.messageYZcol source)

def rowsPerMle (domain : BlockNcDomain) : Nat :=
  3 * KBooleanMle.frameCount domain.laneVariables

def mleBase
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) (source : Fin shape.sourceCount) : Nat :=
  input.frameBase + rowsPerMle domain * source.val

def mleRows
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) : List Row :=
  (canonicalFinIndices shape.sourceCount).flatMap fun source =>
    KBooleanMle.rows (KFrames.frameAt (mleBase input source))
      (sourceTable input source) (laneCoordinates input) 0

def mleOutput
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) (source : Fin shape.sourceCount) : Carried :=
  KBooleanMle.carried (KFrames.frameAt (mleBase input source))
    (sourceTable input source) (laneCoordinates input) 0

def normBase
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) : Nat :=
  input.frameBase + rowsPerMle domain * shape.sourceCount

def normInput
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) (source : Fin shape.sourceCount) :
    KStrictNorm.Input where
  value := mleOutput input source
  frameBase := normBase input + 6 * source.val

def normRows
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) : List Row :=
  (canonicalFinIndices shape.sourceCount).flatMap fun source =>
    KStrictNorm.rows (normInput input source)

def normOutputs
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) : List Carried :=
  (canonicalFinIndices shape.sourceCount).map fun source =>
    KStrictNorm.output (normInput input source)

def mixedBase
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) : Nat :=
  normBase input + 6 * shape.sourceCount

def mixedRows
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) : List Row :=
  KHorner.hornerRows input.gamma (KFrames.frameAt (mixedBase input))
    (normOutputs input) 0

def mixedOutput
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) : Carried :=
  KHorner.hornerCarried input.gamma (KFrames.frameAt (mixedBase input))
    (normOutputs input) 0

def pointEqualityRows (variables : Nat) : Nat :=
  3 * variables + 3 * (variables - 1)

/-- Exact contiguous auxiliary width.  The initial and terminal `K`
equalities emit four rows in total and allocate no column. -/
def allocationWidth
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) : Nat :=
  rowsPerMle domain * shape.sourceCount +
    6 * shape.sourceCount +
    3 * (shape.sourceCount - 1) +
    pointEqualityRows domain.blockVariables +
    pointEqualityRows domain.laneVariables + 6

def equalityBase
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) : Nat :=
  mixedBase input + 3 * (shape.sourceCount - 1)

def blockEqualityInput
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) :
    KPointEquality.Input domain.blockVariables where
  left := input.pointBlock
  right := input.betaBlock
  frameBase := equalityBase input

def laneEqualityInput
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) :
    KPointEquality.Input domain.laneVariables where
  left := input.pointLane
  right := input.betaA
  frameBase := equalityBase input + pointEqualityRows domain.blockVariables

def productBase
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) : Nat :=
  equalityBase input + pointEqualityRows domain.blockVariables +
    pointEqualityRows domain.laneVariables

def selectorFrame
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) : Frame :=
  KFrames.frameAt (productBase input) 0

def terminalFrame
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) : Frame :=
  KFrames.frameAt (productBase input) 1

def selector
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) : Carried :=
  KMulChain.frameOutput (selectorFrame input)

def terminalExpression
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) : Carried :=
  KMulChain.frameOutput (terminalFrame input)

def initialRows
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) : List Row :=
  KEquality.rows KLinear.zeroCarried input.initial

def rowGroups
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) : List (List Row) :=
  [
    initialRows input,
    mleRows input,
    normRows input,
    mixedRows input,
    KPointEquality.rows (blockEqualityInput input),
    KPointEquality.rows (laneEqualityInput input),
    KMul.rows
      (KPointEquality.equalityCarried (blockEqualityInput input))
      (KPointEquality.equalityCarried (laneEqualityInput input))
      (selectorFrame input),
    KMul.rows (selector input) (mixedOutput input) (terminalFrame input),
    KEquality.rows (terminalExpression input) input.terminal
  ]

def rows
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) : List Row :=
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

theorem mleRows_length
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) :
    (mleRows input).length =
      rowsPerMle domain * shape.sourceCount := by
  unfold mleRows
  calc
    ((canonicalFinIndices shape.sourceCount).flatMap fun source =>
        KBooleanMle.rows (KFrames.frameAt (mleBase input source))
          (sourceTable input source) (laneCoordinates input) 0).length =
        rowsPerMle domain *
          (canonicalFinIndices shape.sourceCount).length := by
      apply constantFlatMapLength _ _ (rowsPerMle domain)
      intro source _
      rw [KBooleanMle.rows_length]
      rfl
    _ = rowsPerMle domain * shape.sourceCount := by
      rw [canonicalFinIndices_length]

theorem normRows_length
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) :
    (normRows input).length = 6 * shape.sourceCount := by
  unfold normRows
  rw [constantFlatMapLength _ _ _ (fun source _ =>
    KStrictNorm.rows_length (normInput input source)),
    canonicalFinIndices_length]

@[simp] theorem normOutputs_length
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) :
    (normOutputs input).length = shape.sourceCount := by
  unfold normOutputs
  rw [List.length_map, canonicalFinIndices_length]

theorem rows_length
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) :
    (rows input).length =
      rowsPerMle domain * shape.sourceCount +
        6 * shape.sourceCount +
        3 * (shape.sourceCount - 1) +
        pointEqualityRows domain.blockVariables +
        pointEqualityRows domain.laneVariables + 10 := by
  unfold rows rowGroups initialRows mixedRows pointEqualityRows
  simp only [List.flatten_cons, List.flatten_nil, List.length_append,
    List.length_nil, KEquality.rows_length, mleRows_length, normRows_length,
    KHorner.hornerRows_length, normOutputs_length,
    KPointEquality.rows_length, KMul.rows_length]
  omega

theorem rows_length_eq_allocationWidth_add_four
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) :
    (rows input).length = allocationWidth input + 4 := by
  rw [rows_length]
  rfl

/-- Exact ordered auxiliary interval used by the MLE, norm, Horner,
point-equality, and selector frames. -/
def columns
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) : List Nat :=
  (List.range (allocationWidth input)).map
    (fun offset => input.frameBase + offset)

theorem columns_length
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) :
    (columns input).length = allocationWidth input := by
  simp [columns]

theorem columns_nodup
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) :
    (columns input).Nodup := by
  unfold columns
  exact LinCombNormal.nodup_map _ _ (fun left right equal => by omega)
    List.nodup_range

theorem satisfies_group
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) (assignment : Nat → Nat)
    (satisfied : Satisfies (rows input) assignment)
    (group : List Row) (member : group ∈ rowGroups input) :
    Satisfies group assignment := by
  intro row rowMember
  exact satisfied row (List.mem_flatten.2 ⟨group, member, rowMember⟩)

def decoded (assignment : Nat → Nat) (value : Carried) : K :=
  KPointEquality.decoded assignment value

def decodedLanePoint
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) (assignment : Nat → Nat) :
    CubePoint K domain.laneVariables :=
  KPointEquality.decodedLeft (laneEqualityInput input) assignment

def decodedPoint
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) (assignment : Nat → Nat) :
    Point domain where
  block := KPointEquality.decodedLeft
    (blockEqualityInput input) assignment
  lane := decodedLanePoint input assignment

def decodedCoins
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) (assignment : Nat → Nat) :
    Mixing.Coins domain where
  betaBlock := KPointEquality.decodedRight
    (blockEqualityInput input) assignment
  betaA := KPointEquality.decodedRight
    (laneEqualityInput input) assignment
  gamma := decoded assignment input.gamma

def decodedMessage
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) (assignment : Nat → Nat) :
    Claims shape where
  yRing := fun _ _ _ => K.zero
  yZcol := fun source lane =>
    decoded assignment (input.messageYZcol source lane)

theorem initial_sound
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows input) assignment) :
    decoded assignment input.initial =
      InitialSum.claimedInitial := by
  have group : Satisfies (initialRows input) assignment :=
    satisfies_group input assignment satisfied _ (by simp [rowGroups])
  have bound :=
    KEquality.rows_sound assignment KLinear.zeroCarried input.initial
      constantWire group
  unfold decoded InitialSum.claimedInitial
  apply KConcreteBridge.ofConcrete_injective
  rw [KPointEquality.ofConcrete_decoded,
    KConcreteBridge.ofConcrete_zero]
  unfold carriedValue
  simp only [Pair.mk.injEq]
  exact And.intro bound.1.symm bound.2.symm

theorem mle_source_satisfied
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) (assignment : Nat → Nat)
    (satisfied : Satisfies (rows input) assignment)
    (source : Fin shape.sourceCount) :
    Satisfies
      (KBooleanMle.rows (KFrames.frameAt (mleBase input source))
        (sourceTable input source) (laneCoordinates input) 0)
      assignment := by
  have group : Satisfies (mleRows input) assignment :=
    satisfies_group input assignment satisfied _ (by simp [rowGroups])
  intro row member
  exact group row
    (List.mem_flatMap.2
      ⟨source, List.mem_ofFn.mpr ⟨source, rfl⟩, member⟩)

private theorem cubePoint_eq_of_coordinates_eq
    {variables : Nat}
    (left right : CubePoint K variables)
    (coordinates : left.coordinates = right.coordinates) :
    left = right := by
  cases left
  cases right
  simp_all

private theorem decodedLanePoint_eq_boolean
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) (assignment : Nat → Nat) :
    KBooleanMleSemantics.decodePoint assignment
        (laneCoordinates input) (laneCoordinates_length input) =
      decodedLanePoint input assignment := by
  apply cubePoint_eq_of_coordinates_eq
  simp only [KBooleanMleSemantics.decodePoint, laneCoordinates,
    decodedLanePoint, KPointEquality.decodedLeft,
    KPointEquality.indices]
  simp only [List.map_ofFn]
  apply congrArg List.ofFn
  funext coordinate
  rfl

private theorem semanticTable_eq_laneTable
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) (assignment : Nat → Nat)
    (source : Fin shape.sourceCount) :
    KBooleanMlePadded.semanticTable
        (fun lane =>
          KBooleanMleSemantics.decodeCarried assignment
            (input.messageYZcol source lane)) =
      Terminal.laneTable (domain := domain)
        (decodedMessage input assignment) source := by
  unfold KBooleanMlePadded.semanticTable Terminal.laneTable
    Terminal.paddedYZcol decodedMessage
  apply congrArg BooleanTable.tabulate
  funext vertex
  rfl

theorem mleOutput_sound
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) (assignment : Nat → Nat)
    (satisfied : Satisfies (rows input) assignment)
    (source : Fin shape.sourceCount) :
    decoded assignment (mleOutput input source) =
      Terminal.valueAt (domain := domain)
        (decodedMessage input assignment) source
        (decodedLanePoint input assignment) := by
  have computed :=
    KBooleanMleSemantics.rows_compute_evaluate assignment
      (mleBase input source) (sourceTable input source)
      (laneCoordinates input) (laneCoordinates_length input)
      (mle_source_satisfied input assignment satisfied source)
  unfold decoded
  apply KConcreteBridge.ofConcrete_injective
  rw [KPointEquality.ofConcrete_decoded]
  change
    carriedValue assignment
        (KBooleanMle.carried (KFrames.frameAt (mleBase input source))
          (sourceTable input source) (laneCoordinates input) 0) =
      _
  rw [computed]
  apply congrArg KConcreteBridge.ofConcrete
  rw [sourceTable,
    KBooleanMleCarriedPadded.decodeTable_carriedTable,
    semanticTable_eq_laneTable input assignment source,
    decodedLanePoint_eq_boolean input assignment]
  rfl

theorem norm_source_satisfied
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) (assignment : Nat → Nat)
    (satisfied : Satisfies (rows input) assignment)
    (source : Fin shape.sourceCount) :
    Satisfies (KStrictNorm.rows (normInput input source)) assignment := by
  have group : Satisfies (normRows input) assignment :=
    satisfies_group input assignment satisfied _ (by simp [rowGroups])
  intro row member
  exact group row
    (List.mem_flatMap.2
      ⟨source, List.mem_ofFn.mpr ⟨source, rfl⟩, member⟩)

theorem normOutputs_decoded
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows input) assignment) :
    (normOutputs input).map (decoded assignment) =
      (canonicalFinIndices shape.sourceCount).map fun source =>
        Terminal.rangeAt (domain := domain)
          (decodedMessage input assignment) source
          (decodedLanePoint input assignment) := by
  unfold normOutputs
  rw [List.map_map]
  apply List.map_congr_left
  intro source _
  change decoded assignment (KStrictNorm.output (normInput input source)) = _
  have strict :=
    KStrictNorm.rows_sound (normInput input source) assignment
      constantWire (norm_source_satisfied input assignment satisfied source)
  change decoded assignment (KStrictNorm.output (normInput input source)) =
    _ at strict
  rw [strict]
  change
    ProtocolPolynomial.strictNormResidual concreteOps
        (decoded assignment (mleOutput input source)) =
      _
  rw [mleOutput_sound input assignment satisfied source]
  unfold ProtocolPolynomial.strictNormResidual Terminal.rangeAt
  rw [ConcreteCarrier.derived_sub_eq_concrete_sub]
  rfl

theorem mixedOutput_sound
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows input) assignment) :
    decoded assignment (mixedOutput input) =
      Terminal.mixedRangeAt (decodedMessage input assignment)
        (decodedCoins input assignment)
        (decodedLanePoint input assignment) := by
  have group : Satisfies (mixedRows input) assignment :=
    satisfies_group input assignment satisfied _ (by simp [rowGroups])
  have computed :=
    KConcreteHorner.rows_sound assignment input.gamma
      (KFrames.frameAt (mixedBase input)) (normOutputs input) 0 group
  change
    decoded assignment (mixedOutput input) =
      SumCheck.Finite.Message.evaluateCoefficients concreteOps.toOps
        (decoded assignment input.gamma)
        ((normOutputs input).map (decoded assignment)) at computed
  rw [computed,
    normOutputs_decoded input assignment constantWire satisfied,
    SignedCoefficientPolynomial.evaluate_canonicalFinMap_eq_gammaSum
      concreteOps ConcreteCarrier.extensionLaws]
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
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) (assignment : Nat → Nat)
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

/-- Satisfying NC endpoint rows derive both verifier-owned endpoint
equalities, including the complete message-derived terminal. -/
theorem rows_sound
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : Input shape domain) (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows input) assignment) :
    decoded assignment input.initial = InitialSum.claimedInitial ∧
      decoded assignment input.terminal =
        Terminal.terminalFromMessage (decodedMessage input assignment)
          (decodedCoins input assignment)
          (decodedPoint input assignment) := by
  have blockSatisfied :
      Satisfies (KPointEquality.rows (blockEqualityInput input)) assignment :=
    satisfies_group input assignment satisfied _ (by simp [rowGroups])
  have laneSatisfied :
      Satisfies (KPointEquality.rows (laneEqualityInput input)) assignment :=
    satisfies_group input assignment satisfied _ (by simp [rowGroups])
  have selectorSatisfied :
      Satisfies
        (KMul.rows
          (KPointEquality.equalityCarried (blockEqualityInput input))
          (KPointEquality.equalityCarried (laneEqualityInput input))
          (selectorFrame input)) assignment :=
    satisfies_group input assignment satisfied _ (by simp [rowGroups])
  have productSatisfied :
      Satisfies
        (KMul.rows (selector input) (mixedOutput input)
          (terminalFrame input)) assignment :=
    satisfies_group input assignment satisfied _ (by simp [rowGroups])
  have blockEquality :=
    KPointEquality.rows_sound (blockEqualityInput input) assignment
      constantWire blockSatisfied
  have laneEquality :=
    KPointEquality.rows_sound (laneEqualityInput input) assignment
      constantWire laneSatisfied
  change
    decoded assignment
        (KPointEquality.equalityCarried (blockEqualityInput input)) =
      _ at blockEquality
  change
    decoded assignment
        (KPointEquality.equalityCarried (laneEqualityInput input)) =
      _ at laneEquality
  have selectorValue :=
    decoded_mul assignment
      (KPointEquality.equalityCarried (blockEqualityInput input))
      (KPointEquality.equalityCarried (laneEqualityInput input))
      (selectorFrame input) selectorSatisfied
  have terminalValue :=
    decoded_mul assignment (selector input) (mixedOutput input)
      (terminalFrame input) productSatisfied
  change decoded assignment (selector input) = _ at selectorValue
  change decoded assignment (terminalExpression input) = _ at terminalValue
  constructor
  · exact initial_sound input assignment constantWire satisfied
  · rw [terminal_eq_expression input assignment constantWire satisfied,
      terminalValue, selectorValue, blockEquality, laneEquality,
      mixedOutput_sound input assignment constantWire satisfied]
    rfl

end Nightstream.Implementation.R1CS.Canonical.KSplitNcNcEndpoint
