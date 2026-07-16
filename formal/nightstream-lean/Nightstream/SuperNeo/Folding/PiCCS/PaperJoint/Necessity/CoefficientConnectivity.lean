import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier

/-!
A concrete necessity witness for carried-coefficient connectivity in paper
joint `Pi_CCS`.

Protocol: SuperNeo `Pi_CCS` (Section 7.3 / Appendix D.4).
Phase: common matrix-image ownership before CCS and carried evaluation.
Constraint family: connection between the CCS structure matrices and the
coefficient-expanded matrices consumed by prior-CE checks.

Owns: a smallest finite countermodel showing that the current
`UnifiedInputs.coefficientMatrices` freedom is semantically relevant. Two
inputs have identical layout, CCS structure, assignments, prior point, and
claimed coefficients, but changing only the disconnected coefficient-matrix
family flips `UnifiedInputs.SemanticTruth`.

Does not own: the correct coefficient-packing source type, a proof that the
paper row domain and production block/lane domains coincide, concrete ring
matrix semantics, Rust, R1CS, row removal, or constraint counts.

Emits constraints: no.

Authority boundary: this is a kernel-checked countermodel, not evidence from
the existing circuit. It proves that the coefficient-expanded matrices cannot
be omitted, guessed from the current `Shape`, or left as unauthenticated
prover data. A later repair must introduce one mathematical matrix owner whose
CCS-row and CE-coefficient views are both derived and then prove the required
packing/layout theorem.

| Protocol | Phase | Family | Mathematical obligation |
|---|---|---|---|
| `Pi_CCS` | source ownership | CCS structure | one fixed zero structure matrix |
| `Pi_CCS` | source ownership | running assignment | one fixed assignment coordinate equal to one |
| `Pi_CCS` | carried evaluation | coefficient matrix | compare a zero view with a unit view |
| `Pi_CCS` | public claim | prior CE coefficient | the fixed claimed value is zero |
| assurance | necessity | cross-view connectivity | changing only the coefficient view flips semantic truth |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.CoefficientConnectivity

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.UnifiedSources

/-- Smallest shape with one running source, one matrix, one coefficient, and
one Boolean row. There are no fresh CCS sources, so the witness isolates the
carried-coefficient connection rather than CCS satisfiability. -/
def counterexampleShape : Shape where
  cubeVariables := 0
  freshCount := 0
  runningCount := 1
  matrixCount := 1
  coefficientCount := 1

/-- The unique Boolean vertex is in exact correspondence with the unique
assignment column. -/
def singletonLayout : ColumnLayout 0 1 where
  toColumn := fun _ => 0
  toVertex := fun _ => .nil
  toColumn_toVertex := by
    intro column
    apply Fin.eq_of_val_eq
    omega
  toVertex_toColumn := by
    intro vertex
    cases vertex
    rfl

/-- A fixed CCS structure. Its matrix is zero and its sparse polynomial has
no terms. Both counterexample inputs share this exact object. -/
def fixedSystem :
    CCSResidualTable.Structure F counterexampleShape 1 where
  matrices := fun _ _ _ => 0
  constraintPolynomial :=
    { degreeBound := 1
      terms := []
      termsBelowDegree := by simp }

/-- The sole authoritative assignment coordinate is one, which satisfies the
strict centered `b = 2` norm. -/
def fixedAssignments :
    Fin counterexampleShape.sourceCount -> Assignment F 1 :=
  fun _ _ => 1

/-- The unique prior point has dimension zero. -/
def fixedPriorPoint : CubePoint K counterexampleShape.cubeVariables where
  coordinates := []
  dimension := rfl

/-- The public carried claim is fixed to zero. -/
def fixedClaim : CarriedCoordinate counterexampleShape -> K :=
  fun _ => K.zero

/-- A coefficient-expanded view whose sole entry is zero. -/
def zeroCoefficientMatrices :
    Fin counterexampleShape.matrixCount ->
      Fin counterexampleShape.coefficientCount ->
        BooleanMatrix F counterexampleShape.cubeVariables 1 :=
  fun _ _ _ _ => 0

/-- A disconnected coefficient-expanded view whose sole entry is one. -/
def unitCoefficientMatrices :
    Fin counterexampleShape.matrixCount ->
      Fin counterexampleShape.coefficientCount ->
        BooleanMatrix F counterexampleShape.cubeVariables 1 :=
  fun _ _ _ _ => 1

/-- The semantically valid input uses the zero coefficient view, matching the
fixed zero public claim. -/
def validInputs : UnifiedInputs K counterexampleShape 1 where
  layout := singletonLayout
  system := fixedSystem
  assignments := fixedAssignments
  coefficientMatrices := zeroCoefficientMatrices
  priorPoint := fixedPriorPoint
  claimedCoefficient := fixedClaim

/-- The disconnected input changes only the coefficient-expanded matrix
family. Every other field is definitionally shared with `validInputs`. -/
def disconnectedInputs : UnifiedInputs K counterexampleShape 1 :=
  { validInputs with coefficientMatrices := unitCoefficientMatrices }

/-- The unique running-source index. Explicit bounds avoid asking numeral
elaboration to unfold the named shape. -/
def uniqueRunning : Fin counterexampleShape.runningCount :=
  ⟨0, by simp [counterexampleShape]⟩

/-- The unique matrix index. -/
def uniqueMatrix : Fin counterexampleShape.matrixCount :=
  ⟨0, by simp [counterexampleShape]⟩

/-- The unique ring-coefficient index in this isolated model. -/
def uniqueCoefficient : Fin counterexampleShape.coefficientCount :=
  ⟨0, by simp [counterexampleShape]⟩

/-- The sole carried coordinate. -/
def uniqueCoordinate : CarriedCoordinate counterexampleShape where
  running := uniqueRunning
  matrix := uniqueMatrix
  coefficient := uniqueCoefficient

/-- Exact statement that every current input surface other than the
coefficient-expanded matrix family is unchanged. -/
def SameNonCoefficientInputs
    (left right : UnifiedInputs K counterexampleShape 1) : Prop :=
  left.layout = right.layout ∧
    left.system = right.system ∧
    left.assignments = right.assignments ∧
    left.priorPoint = right.priorPoint ∧
    left.claimedCoefficient = right.claimedCoefficient

/-- The two inputs are indistinguishable through every current source except
the disconnected coefficient-expanded matrix family. -/
theorem sameNonCoefficientInputs :
    SameNonCoefficientInputs validInputs disconnectedInputs := by
  exact ⟨rfl, rfl, rfl, rfl, rfl⟩

/-- The coefficient-expanded matrix families genuinely differ. -/
theorem coefficientMatrices_ne :
    validInputs.coefficientMatrices ≠
      disconnectedInputs.coefficientMatrices := by
  intro equal
  have entryEqual := congrFun
    (congrFun
      (congrFun
        (congrFun equal uniqueMatrix)
        uniqueCoefficient)
      (.nil : BooleanVertex counterexampleShape.cubeVariables))
    (0 : Fin 1)
  have zero_ne_one : (0 : F) ≠ 1 := by decide
  exact zero_ne_one (by
    simpa [validInputs, disconnectedInputs, zeroCoefficientMatrices,
      unitCoefficientMatrices] using entryEqual)

/-- With the zero coefficient view, every carried coordinate computes zero. -/
theorem validInputs_computedCoefficient_eq_zero
    (coordinate : CarriedCoordinate counterexampleShape) :
    CarriedEvaluationResidual.computedCoefficient ConcreteCarrier.baseOps
        ConcreteCarrier.extensionOps K.embed validInputs.carriedData
        coordinate =
      K.zero := by
  rw [← CarriedEvaluationResidual.imageTable_evaluate_eq_computedCoefficient
    ConcreteCarrier.baseOps ConcreteCarrier.extensionOps
    ConcreteCarrier.extensionLaws K.embed validInputs.carriedData coordinate]
  rfl

/-- With the unit coefficient view and unit assignment, the sole carried
coordinate computes one. -/
theorem disconnectedInputs_computedCoefficient_eq_one :
    CarriedEvaluationResidual.computedCoefficient ConcreteCarrier.baseOps
        ConcreteCarrier.extensionOps K.embed disconnectedInputs.carriedData
        uniqueCoordinate =
      K.one := by
  rw [← CarriedEvaluationResidual.imageTable_evaluate_eq_computedCoefficient
    ConcreteCarrier.baseOps ConcreteCarrier.extensionOps
    ConcreteCarrier.extensionLaws K.embed disconnectedInputs.carriedData
    uniqueCoordinate]
  rfl

/-- The zero coefficient view computes the claimed zero prior evaluation. -/
theorem validInputs_semanticTruth :
    validInputs.SemanticTruth ConcreteCarrier.baseOps
      ConcreteCarrier.extensionOps K.embed := by
  constructor
  · intro source
    exact Fin.elim0 source
  constructor
  · intro source column
    change centeredMagnitude (1 : F) < 2
    decide
  · intro coordinate
    unfold CarriedEvaluationResidual.EvaluationClaimHolds
    calc
      validInputs.carriedData.claimedCoefficient coordinate = K.zero := rfl
      _ = CarriedEvaluationResidual.computedCoefficient
          ConcreteCarrier.baseOps ConcreteCarrier.extensionOps K.embed
          validInputs.carriedData coordinate :=
        (validInputs_computedCoefficient_eq_zero coordinate).symm

/-- Changing only the disconnected coefficient view makes the same zero claim
false: the derived prior evaluation is now one. -/
theorem disconnectedInputs_not_semanticTruth :
    ¬ disconnectedInputs.SemanticTruth ConcreteCarrier.baseOps
      ConcreteCarrier.extensionOps K.embed := by
  intro semanticTruth
  have carried := semanticTruth.2.2 uniqueCoordinate
  have claimZero :
      disconnectedInputs.carriedData.claimedCoefficient uniqueCoordinate =
        K.zero := by
    rfl
  have zero_eq_one : K.zero = K.one := by
    exact claimZero.symm.trans <|
      carried.trans disconnectedInputs_computedCoefficient_eq_one
  exact (by decide : K.zero ≠ K.one) zero_eq_one

/-- Inclusion-necessity witness: the existing non-coefficient sources do not
determine carried-evaluation truth. Any sound verifier model must bind a
single authoritative matrix representation to both the CCS and coefficient
views before this obligation can be removed or compressed. -/
theorem omitting_coefficient_connectivity_changes_semantic_truth :
    ∃ left right : UnifiedInputs K counterexampleShape 1,
      SameNonCoefficientInputs left right ∧
        left.coefficientMatrices ≠ right.coefficientMatrices ∧
        left.SemanticTruth ConcreteCarrier.baseOps
          ConcreteCarrier.extensionOps K.embed ∧
        ¬ right.SemanticTruth ConcreteCarrier.baseOps
          ConcreteCarrier.extensionOps K.embed := by
  exact ⟨validInputs, disconnectedInputs, sameNonCoefficientInputs,
    coefficientMatrices_ne, validInputs_semanticTruth,
    disconnectedInputs_not_semanticTruth⟩

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.CoefficientConnectivity
