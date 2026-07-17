import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Terminal

/-!
Necessity of binding the Split-NC `yZcol` message before the NC terminal.

Protocol: SuperNeo `Pi_CCS`, split NC branch.
Phase: verifier-output binding at the terminal SumCheck point.
Constraint family: `yZcol` source binding and scalar terminal equality; this
file emits no rows.

Owns: two closed, typed counterexamples over the smallest useful Phi81
carrier. The first changes only an honestly projected assignment-two `yZcol`
to zero and flips the scalar terminal equality. The second changes the zero
assignment's lane-zero claim to one; its cubic still vanishes, so the scalar
terminal equality accepts even though source binding fails.

Does not own: an executable verifier, transcript derivation, SumCheck
soundness, production behavior, Rust, R1CS, row removal, or constraint counts.

Emits constraints: no.

Authority boundary: the scalar terminal equality checks a cubic image, not
the underlying projection. Because both zero and one are roots of
`(z+1)z(z-1)`, `YZcolBoundToSources` is a separate necessary obligation. The
fixtures derive the semantic side from `SplitNc.Sources.Data`; no circuit or
implementation module is imported.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc.terminal.necessity.scalar` | replacing the assignment-two projection by zero flips terminal equality | retained check | `scalarTerminalCheck_is_necessary` |
| `nifs.pi_ccs.nc.terminal.necessity.binding` | zero and one share the same cubic image but only zero is source-bound | retained check | `scalarTerminalCheck_is_insufficient` |
| `nifs.pi_ccs.nc.terminal.necessity.fixture` | one source, one 54-coordinate carrier, and `64 x 64` padded domains satisfy coverage | computed | `witnessCovers` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Necessity.YZcolTerminal

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc

/-! ## Smallest useful typed fixture -/

/-- One running source and one logical coordinate. Phi81 carrier completion
makes the authoritative assignment width exactly 54. Matrices are absent
because this counterexample isolates the NC output boundary. -/
def witnessShape : SemanticShape where
  rowVariables := 0
  logicalWidth := 1
  freshCount := 0
  runningCount := 1
  matrixCount := 0

/-- Six Boolean coordinates cover both the 54-coordinate completed carrier
and its 54 active Phi81 lanes. -/
def witnessDomain : FlatNcDomain where
  columnVariables := 6
  laneVariables := 6

/-- The chosen padded domains cover every authoritative coordinate. -/
theorem witnessCovers : witnessDomain.Covers witnessShape := by
  simp [FlatNcDomain.Covers, FlatNcDomain.columnCount,
    FlatNcDomain.laneCount, SemanticShape.carrierWidth, witnessDomain,
    witnessShape, Phi81CarrierLayout.carrierWidth,
    Phi81ColumnLayout.blockCount, ringDegree]

/-- Empty CCS syntax; the NC counterexamples do not inspect a matrix. -/
def emptyConstraintPolynomial :
    CCSResidualTable.ConstraintPolynomial F witnessShape.matrixCount where
  degreeBound := 0
  terms := []
  termsBelowDegree := by simp

/-- The unique row point in dimension zero. -/
def emptyPriorPoint : CubePoint K witnessShape.rowVariables where
  coordinates := []
  dimension := rfl

/-- Running assignment with value two at the first carrier coordinate and
zero everywhere else. -/
def assignmentTwo : Assignment F witnessShape.carrierWidth :=
  fun column => if column.val = 0 then 2 else 0

/-- All-zero running assignment over the same complete carrier. -/
def assignmentZero : Assignment F witnessShape.carrierWidth := fun _ => 0

private def noMatrices : Fin witnessShape.matrixCount ->
    BooleanMatrix F witnessShape.rowVariables witnessShape.logicalWidth :=
  fun matrix => Fin.elim0 matrix

private def noFreshAssignments :
    Fin witnessShape.freshCount -> Assignment F witnessShape.logicalWidth :=
  fun source => Fin.elim0 source

private def noClaimedCoefficient :
    CarriedCoordinate witnessShape.paperShape -> K :=
  fun coordinate => Fin.elim0 coordinate.matrix

/-- Independent semantic source whose sole assignment has first coordinate
two. The projection named honest below is canonical for this source even
though the strict-norm statement is intentionally false. -/
def dataTwo : Data witnessShape where
  matrices := noMatrices
  constraintPolynomial := emptyConstraintPolynomial
  freshAssignments := noFreshAssignments
  runningAssignments := fun _ => assignmentTwo
  priorPoint := emptyPriorPoint
  claimedCoefficient := noClaimedCoefficient

/-- Independent semantic source whose sole assignment is zero. -/
def dataZero : Data witnessShape where
  matrices := noMatrices
  constraintPolynomial := emptyConstraintPolynomial
  freshAssignments := noFreshAssignments
  runningAssignments := fun _ => assignmentZero
  priorPoint := emptyPriorPoint
  claimedCoefficient := noClaimedCoefficient

/-- First authoritative carrier coordinate. -/
def carrierZero : Fin witnessShape.carrierWidth := ⟨0, by decide⟩

/-- Unique source coordinate. -/
def uniqueSource : Fin witnessShape.sourceCount := ⟨0, by decide⟩

/-- First active Phi81 lane. -/
def activeLaneZero : Fin ringDegree := ⟨0, by decide⟩

/-- First carrier coordinate embedded in the padded column cube. -/
def columnZero : Fin witnessDomain.columnCount :=
  witnessDomain.carrierColumn witnessCovers carrierZero

/-- First active Phi81 lane embedded in the padded lane cube. -/
def laneZero : Fin witnessDomain.laneCount :=
  witnessDomain.phi81Lane witnessCovers activeLaneZero

/-- Terminal point selecting carrier coordinate zero and lane zero. -/
def terminalPoint : Point witnessDomain := booleanPoint columnZero laneZero

/-- The terminal equality selectors target the same Boolean point and the
single-source gamma weight is one. -/
def coins : Mixing.Coins witnessDomain where
  betaM := terminalPoint.column
  betaA := terminalPoint.lane
  gamma := K.one

def verifierPoints (data : Data witnessShape) :
    VerifierPoints witnessShape witnessDomain where
  rPrime := data.priorPoint
  sPrime := terminalPoint.column

/-- At the Boolean zero point, the canonical active output is exactly the
authoritative diagonal entry. This uses the generic source-projection bridge
and avoids evaluating a handwritten 54-term fold in the counterexamples. -/
theorem canonicalYZcol_at_zero (data : Data witnessShape) :
    canonicalYZcol witnessCovers data (verifierPoints data)
        uniqueSource activeLaneZero =
      K.embed (Semantics.Nc.diagonal
        (data.assignment uniqueSource) carrierZero activeLaneZero) := by
  change canonicalYZcol witnessCovers data
      ({ rPrime := data.priorPoint, sPrime := terminalPoint.column } :
        VerifierPoints witnessShape witnessDomain)
      uniqueSource activeLaneZero = _
  rw [Terminal.canonicalYZcol_eq_columnValueAt]
  have live := SourceProjection.sourceValueAt_live witnessCovers data
    uniqueSource carrierZero activeLaneZero
  unfold SourceProjection.sourceValueAt SourceProjection.laneTableAtColumn at live
  simp only [booleanPoint] at live
  rw [SumCheckTruthPath.evaluate_toCubePoint_eq_valueAt
    ConcreteCarrier.extensionOps ConcreteCarrier.extensionLaws] at live
  rw [BooleanTable.valueAt_tabulate] at live
  simpa [verifierPoints, terminalPoint, columnZero, laneZero, booleanPoint]
    using live

/-! ## Scalar terminal equality is necessary -/

/-- Canonical source-derived output for the assignment-two fixture. -/
def honestTwo : OutputMessage witnessShape :=
  canonicalClaims witnessCovers dataTwo (verifierPoints dataTwo)

/-- The canonical assignment-two projection exposes two at the selected
source/lane coordinate. -/
theorem honestTwo_yZcol_zero :
    honestTwo.yZcol uniqueSource activeLaneZero = K.embed (2 : F) := by
  change canonicalYZcol witnessCovers dataTwo (verifierPoints dataTwo)
      uniqueSource activeLaneZero = K.embed (2 : F)
  rw [canonicalYZcol_at_zero]
  decide

/-- Forgery changing only the `yZcol` branch of `honestTwo` to zero. -/
def forgedZero : OutputMessage witnessShape :=
  { honestTwo with yZcol := fun _ _ => K.zero }

/-- Exact scalar terminal equation checked after the NC SumCheck. -/
def ScalarTerminalCheck
    (data : Data witnessShape)
    (message : OutputMessage witnessShape) : Prop :=
  Terminal.terminalFromMessage .paperNc message coins terminalPoint =
    Mixing.qAtPoint .paperNc witnessCovers data coins terminalPoint

/-- At the selected Boolean point both equality gates are one, the unique
paper-relative source has exponent zero, and the scalar terminal is exactly
that source's cubic. -/
theorem terminalFromMessage_eq_rangeAt
    (message : OutputMessage witnessShape) :
    Terminal.terminalFromMessage .paperNc message coins terminalPoint =
      Terminal.rangeAt (domain := witnessDomain) message uniqueSource
        terminalPoint.lane := by
  have columnSelector :
      SumCheckTruthPath.pointEquality ConcreteCarrier.extensionOps
          terminalPoint.column coins.betaM = K.one := by
    rw [show coins.betaM = terminalPoint.column from rfl]
    change SumCheckTruthPath.pointEquality ConcreteCarrier.extensionOps
        (BooleanVertex.toCubePoint ConcreteCarrier.extensionOps
          (columnVertex columnZero))
        (BooleanVertex.toCubePoint ConcreteCarrier.extensionOps
          (columnVertex columnZero)) = K.one
    rw [SumCheckTruthPath.pointEquality_toCubePoint_eq_equalityWeight
      ConcreteCarrier.extensionOps ConcreteCarrier.extensionLaws]
    rw [BooleanReproduction.equalityWeight_toCubePoint
      ConcreteCarrier.extensionOps ConcreteCarrier.extensionLaws]
    simp [ConcreteCarrier.extensionOps]
  have laneSelector :
      SumCheckTruthPath.pointEquality ConcreteCarrier.extensionOps
          terminalPoint.lane coins.betaA = K.one := by
    rw [show coins.betaA = terminalPoint.lane from rfl]
    change SumCheckTruthPath.pointEquality ConcreteCarrier.extensionOps
        (BooleanVertex.toCubePoint ConcreteCarrier.extensionOps
          (laneVertex laneZero))
        (BooleanVertex.toCubePoint ConcreteCarrier.extensionOps
          (laneVertex laneZero)) = K.one
    rw [SumCheckTruthPath.pointEquality_toCubePoint_eq_equalityWeight
      ConcreteCarrier.extensionOps ConcreteCarrier.extensionLaws]
    rw [BooleanReproduction.equalityWeight_toCubePoint
      ConcreteCarrier.extensionOps ConcreteCarrier.extensionLaws]
    simp [ConcreteCarrier.extensionOps]
  unfold Terminal.terminalFromMessage Terminal.mixedRangeAt
  rw [columnSelector, laneSelector]
  simp [witnessShape, SemanticShape.sourceCount, canonicalFinIndices,
    FiniteSumAlgebra.sumMap, BooleanTable.finiteSum,
    SignedJointIdentity.gammaTerm, Mixing.sourceExponent, coins,
    TargetPolynomial.power, ConcreteCarrier.extensionOps,
    K.mul, K.add, K.one, K.zero, uniqueSource,
    Fin.one_mul, Fin.zero_mul, Fin.mul_zero, Fin.add_zero]

/-- The message lane MLE at the selected Boolean point reproduces the active
lane-zero claim exactly. -/
theorem valueAt_eq_yZcol_zero
    (message : OutputMessage witnessShape) :
    Terminal.valueAt (domain := witnessDomain) message uniqueSource
        terminalPoint.lane =
      message.yZcol uniqueSource activeLaneZero := by
  unfold Terminal.valueAt Terminal.laneTable
  change (BooleanTable.tabulate fun lane =>
      Terminal.paddedYZcol (domain := witnessDomain) message uniqueSource
        (laneIndex lane)).evaluate ConcreteCarrier.extensionOps
      (BooleanVertex.toCubePoint ConcreteCarrier.extensionOps
        (laneVertex laneZero)) = _
  rw [SumCheckTruthPath.evaluate_toCubePoint_eq_valueAt
    ConcreteCarrier.extensionOps ConcreteCarrier.extensionLaws]
  rw [BooleanTable.valueAt_tabulate, laneIndex_laneVertex]
  exact Terminal.paddedYZcol_live witnessCovers message uniqueSource activeLaneZero

/-- The forgery preserves the whole `yRing` branch definitionally. -/
theorem forgedZero_yRing_eq : forgedZero.yRing = honestTwo.yRing := rfl

/-- The honest assignment-two projection is nonzero at source/lane zero, so
the forged message genuinely changes `yZcol`. -/
theorem forgedZero_yZcol_ne : forgedZero.yZcol ≠ honestTwo.yZcol := by
  intro equal
  have entry := congrFun (congrFun equal uniqueSource) activeLaneZero
  have zeroEq : K.zero = honestTwo.yZcol uniqueSource activeLaneZero := by
    simpa [forgedZero] using entry
  rw [honestTwo_yZcol_zero] at zeroEq
  have different : K.zero ≠ K.embed (2 : F) := by decide
  exact different zeroEq

/-- The canonical assignment-two output satisfies the scalar terminal
equation through the independent source-binding theorem. -/
theorem honestTwo_scalarTerminalCheck : ScalarTerminalCheck dataTwo honestTwo := by
  exact Terminal.terminal_eq_qAtPoint_of_yZcolBoundToSources
    .paperNc witnessCovers dataTwo coins terminalPoint honestTwo
    (canonicalClaims_yZcolBoundToSources witnessCovers dataTwo
      (verifierPoints dataTwo))

theorem forgedZero_rangeAt_eq_zero :
    Terminal.rangeAt (domain := witnessDomain) forgedZero uniqueSource
        terminalPoint.lane = K.zero := by
  unfold Terminal.rangeAt
  rw [valueAt_eq_yZcol_zero]
  decide

theorem honestTwo_rangeAt_ne_zero :
    Terminal.rangeAt (domain := witnessDomain) honestTwo uniqueSource
        terminalPoint.lane ≠ K.zero := by
  unfold Terminal.rangeAt
  rw [valueAt_eq_yZcol_zero, honestTwo_yZcol_zero]
  decide

theorem forgedZero_terminal_ne_honestTwo :
    Terminal.terminalFromMessage .paperNc forgedZero coins terminalPoint ≠
      Terminal.terminalFromMessage .paperNc honestTwo coins terminalPoint := by
  rw [terminalFromMessage_eq_rangeAt, terminalFromMessage_eq_rangeAt,
    forgedZero_rangeAt_eq_zero]
  exact fun equal => honestTwo_rangeAt_ne_zero equal.symm

/-- Replacing the same projection by zero fails the scalar terminal equation. -/
theorem forgedZero_not_scalarTerminalCheck :
    ¬ ScalarTerminalCheck dataTwo forgedZero := by
  intro forgedCheck
  apply forgedZero_terminal_ne_honestTwo
  exact forgedCheck.trans honestTwo_scalarTerminalCheck.symm

/-- Inclusion-necessity witness for the scalar terminal check: two messages
share every non-`yZcol` field, but removing the check admits the forged one. -/
theorem scalarTerminalCheck_is_necessary :
    ∃ honest forged : OutputMessage witnessShape,
      honest.yRing = forged.yRing ∧
      honest.yZcol ≠ forged.yZcol ∧
      ScalarTerminalCheck dataTwo honest ∧
      ¬ ScalarTerminalCheck dataTwo forged := by
  exact ⟨honestTwo, forgedZero, forgedZero_yRing_eq.symm,
    forgedZero_yZcol_ne.symm, honestTwo_scalarTerminalCheck,
    forgedZero_not_scalarTerminalCheck⟩

/-! ## Scalar terminal equality is not source authority -/

/-- Canonical source-derived output for the all-zero assignment. -/
def honestZero : OutputMessage witnessShape :=
  canonicalClaims witnessCovers dataZero (verifierPoints dataZero)

/-- The canonical zero assignment projects to zero at source/lane zero. -/
theorem honestZero_yZcol_zero :
    honestZero.yZcol uniqueSource activeLaneZero = K.zero := by
  change canonicalYZcol witnessCovers dataZero (verifierPoints dataZero)
      uniqueSource activeLaneZero = K.zero
  rw [canonicalYZcol_at_zero]
  decide

/-- Forge only source zero, lane zero to one. All other `yZcol` coordinates
remain zero and the entire `yRing` branch remains canonical. -/
def forgedOne : OutputMessage witnessShape :=
  { honestZero with
    yZcol := fun source lane =>
      if source.val = 0 ∧ lane.val = 0 then K.one else K.zero }

/-- The forged unit is not the canonical projection of the zero assignment. -/
theorem forgedOne_not_yZcolBoundToSources :
    ¬ YZcolBoundToSources witnessCovers dataZero
      (verifierPoints dataZero) forgedOne := by
  intro bound
  have entry := bound uniqueSource activeLaneZero
  have oneEq : K.one = honestZero.yZcol uniqueSource activeLaneZero := by
    simpa [forgedOne] using entry
  rw [honestZero_yZcol_zero] at oneEq
  have different : K.one ≠ K.zero := by decide
  exact different oneEq

/-- The canonical zero projection satisfies the scalar terminal equation. -/
theorem honestZero_scalarTerminalCheck :
    ScalarTerminalCheck dataZero honestZero := by
  exact Terminal.terminal_eq_qAtPoint_of_yZcolBoundToSources
    .paperNc witnessCovers dataZero coins terminalPoint honestZero
    (canonicalClaims_yZcolBoundToSources witnessCovers dataZero
      (verifierPoints dataZero))

/-- At the selected Boolean lane, the forged unit and canonical zero have the
same cubic image. This equality concerns only the scalar terminal, not the
underlying output messages. -/
theorem forgedOne_terminal_eq_honestZero :
    Terminal.terminalFromMessage .paperNc forgedOne coins terminalPoint =
      Terminal.terminalFromMessage .paperNc honestZero coins terminalPoint := by
  rw [terminalFromMessage_eq_rangeAt, terminalFromMessage_eq_rangeAt]
  unfold Terminal.rangeAt
  rw [valueAt_eq_yZcol_zero, valueAt_eq_yZcol_zero,
    honestZero_yZcol_zero]
  decide

/-- Both the semantic zero and forged unit are roots of the strict norm
cubic, so the forged message still satisfies the scalar terminal equation. -/
theorem forgedOne_scalarTerminalCheck :
    ScalarTerminalCheck dataZero forgedOne := by
  unfold ScalarTerminalCheck
  rw [forgedOne_terminal_eq_honestZero]
  exact honestZero_scalarTerminalCheck

/-- The scalar terminal equation cannot replace source binding: this explicit
message passes the former and violates the latter. -/
theorem scalarTerminalCheck_is_insufficient :
    ∃ message : OutputMessage witnessShape,
      ScalarTerminalCheck dataZero message ∧
      ¬ YZcolBoundToSources witnessCovers dataZero
        (verifierPoints dataZero) message := by
  exact ⟨forgedOne, forgedOne_scalarTerminalCheck,
    forgedOne_not_yZcolBoundToSources⟩

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Necessity.YZcolTerminal
