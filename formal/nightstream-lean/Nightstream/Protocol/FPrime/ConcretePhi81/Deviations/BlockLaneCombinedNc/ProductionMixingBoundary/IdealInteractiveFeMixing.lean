import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary.IdealInteractiveCarrier
import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary.IdealInteractiveRootCounting
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.MixingSoundness

/-!
Finite mixing-root bounds for production FE.

Assurance tier: model-level registered-deviation refinement.

Owns: the exact fresh-row and carried-lane Boolean controllers, a dense
constant-first coefficient encoding of the production FE gamma expression,
exact evaluation transport to the repository's named `Fe.MixingRoot`, and
finite-root bounds over the selected ideal-interactive carrier.

Does not own: NC mixing, SumCheck round collisions, Fiat--Shamir, Poseidon2,
closed concrete field certificates, Rust/R1CS, artifacts, costs, or rows.

Emits constraints: no.

The dense coefficient encoding retains every production exponent. It appends
only an explicit final block of zero coefficients, so its stated degree is a
conservative upper bound rather than a claim of canonical trimming.

| Boundary | Owned equation |
| --- | --- |
| FE selectors | Boolean controller evaluation equals the exact row/lane selector event |
| FE gamma | Dense coefficient evaluation equals the named production gamma expression |
| Probability | Each nonzero controller/polynomial is bounded by degree over support cardinality |
-/

set_option autoImplicit false
set_option maxHeartbeats 1000000
set_option maxRecDepth 2000

namespace Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary.IdealInteractiveFeMixing

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial
open Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteWords
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
open Nightstream.SuperNeo.SumCheck.Finite
open IdealInteractiveCarrier
open IdealInteractiveRootCounting

private abbrev ops := ConcreteCarrier.extensionOps
private abbrev laws := ConcreteCarrier.extensionLaws

private noncomputable def propositionEvent (proposition : Prop) : Bool :=
  @ite Bool proposition (Classical.propDecidable proposition) true false

private theorem zero_mul (value : K) :
    K.mul K.zero value = K.zero := by
  change ops.mul ops.zero value = ops.zero
  rw [laws.mul_comm, laws.mul_zero]

private theorem mul_zero (value : K) :
    K.mul value K.zero = K.zero := by
  change ops.mul value ops.zero = ops.zero
  exact laws.mul_zero value

private theorem zero_add (value : K) :
    K.add K.zero value = value := by
  change ops.add ops.zero value = value
  exact laws.zero_add value

private theorem add_zero (value : K) :
    K.add value K.zero = value := by
  change ops.add value ops.zero = value
  exact laws.add_zero value

private theorem evaluate_zero : forall (gamma : K) (values : List K),
    CoefficientRootCounting.AllZero ops values ->
      Message.evaluateCoefficients ops.toOps gamma values = K.zero
  | _, [], _ => rfl
  | gamma, value :: values, allZero => by
      have valueZero : value = K.zero := allZero value (by simp)
      have tailZero : CoefficientRootCounting.AllZero ops values := by
        intro prior member
        exact allZero prior (by simp [member])
      change ops.add value
          (ops.mul gamma
            (Message.evaluateCoefficients ops.toOps gamma values)) =
        ops.zero
      rw [valueZero, evaluate_zero gamma values tailZero]
      calc
        ops.add ops.zero (ops.mul gamma ops.zero) =
            ops.add ops.zero ops.zero :=
          congrArg (ops.add ops.zero) (laws.mul_zero gamma)
        _ = ops.zero := laws.zero_add ops.zero

private theorem evaluate_replicate_zero
    (gamma : K) (count : Nat) :
    Message.evaluateCoefficients ops.toOps gamma
        (List.replicate count K.zero) = K.zero := by
  apply evaluate_zero
  intro value member
  simpa using List.eq_of_mem_replicate member

/-! ## Exact FE coins and selector tables -/

/-- Exact production FE coins from the selected engine coordinates.
`betaA` remains present even though the complete initial residual compression
eliminates it by lane partition of unity. -/
def coins
    {shape : SemanticShape}
    (alpha : AlphaWord)
    (betaA : BetaAWord)
    (betaR : BetaRWord shape)
    (gamma : K) :
    Fe.Coins shape PiCcsDomains.production.fe where
  alpha := cubePoint alpha
  betaA := cubePoint betaA
  betaR := cubePoint betaR
  gamma := gamma

/-- One embedded fresh CCS residual table before the sampled row selector. -/
def freshTable
    {shape : SemanticShape}
    (data : Data shape)
    (fresh : Fin shape.freshCount) :
    BooleanTable K shape.rowVariables :=
  BooleanTable.tabulate fun row =>
    K.embed <| CCSResidualTable.residualAt ConcreteCarrier.baseOps
      data.freshBatch.system (data.freshBatch.assignments fresh) row

/-- The fresh table's MLE at `betaR` is exactly the corresponding production
FE gamma coefficient before its sign. -/
theorem freshTable_evaluate
    {shape : SemanticShape}
    (data : Data shape)
    (fresh : Fin shape.freshCount)
    (betaR : BetaRWord shape) :
    (freshTable data fresh).evaluate ops (cubePoint betaR) =
      FiniteSumAlgebra.sumMap ops
        (BooleanVertex.all shape.rowVariables) fun row =>
          K.mul (row.equalityWeight ops (cubePoint betaR)) <|
            K.embed <| CCSResidualTable.residualAt ConcreteCarrier.baseOps
              data.freshBatch.system
              (data.freshBatch.assignments fresh) row := by
  exact
    (BooleanReproduction.equalityWeighted_tabulate_eq_evaluate
      ops laws (cubePoint betaR) (fun row =>
        K.embed <| CCSResidualTable.residualAt ConcreteCarrier.baseOps
          data.freshBatch.system
          (data.freshBatch.assignments fresh) row)).symm

/-- A nonzero base-field fresh residual table remains nonzero after the
authoritative base-to-extension embedding. -/
theorem freshTable_nonzero
    {shape : SemanticShape}
    (data : Data shape)
    (fresh : Fin shape.freshCount)
    (nonzero :
      ¬ (data.freshBatch.residualTables ConcreteCarrier.baseOps fresh
        ).AllEntriesZero ConcreteCarrier.baseOps) :
    ¬ (freshTable data fresh).AllEntriesZero ops := by
  classical
  intro liftedZero
  apply nonzero
  unfold CCSResidualTable.FreshBatch.residualTables
  unfold CCSResidualTable.residualTable
  rw [BooleanTable.tabulate_allEntriesZero_iff]
  intro row
  have embeddedZero :=
    (BooleanTable.tabulate_allEntriesZero_iff ops _).mp liftedZero row
  exact (ConcreteCarrier.zeroReflectingLift.zero_iff _).mp embeddedZero

/-- One complete carried residual lane table before the sampled `alpha`.
Only the 54 active Phi81 lanes enter `paddedLaneEvaluation`; padded lanes are
derived zeros and never certificate input. -/
def carriedTable
    {shape : SemanticShape}
    (profile : Fe.SupportedProfile shape PiCcsDomains.production.fe)
    (data : Data shape)
    (running : Fin shape.runningCount)
    (matrix : Fin shape.matrixCount) :
    BooleanTable K PiCcsDomains.production.fe.laneVariables :=
  BooleanTable.tabulate fun sampled =>
    Fe.paddedLaneEvaluation profile.laneCovers
      (fun lane => CarriedEvaluationResidual.residual
        ConcreteCarrier.baseOps ops K.embed data.carriedData {
          running := running
          matrix := matrix
          coefficient := lane
        })
      (sampled.toCubePoint ops)

/-- The carried table's MLE at `alpha` is exactly the production padded-lane
specialization. -/
theorem carriedTable_evaluate
    {shape : SemanticShape}
    (profile : Fe.SupportedProfile shape PiCcsDomains.production.fe)
    (data : Data shape)
    (running : Fin shape.runningCount)
    (matrix : Fin shape.matrixCount)
    (alpha : AlphaWord) :
    (carriedTable profile data running matrix).evaluate ops
        (cubePoint alpha) =
      Fe.paddedLaneEvaluation profile.laneCovers
        (fun lane => CarriedEvaluationResidual.residual
          ConcreteCarrier.baseOps ops K.embed data.carriedData {
            running := running
            matrix := matrix
            coefficient := lane
          })
        (cubePoint alpha) := by
  exact
    (BooleanReproduction.equalityWeighted_tabulate_eq_evaluate
      ops laws (cubePoint alpha) (fun sampled =>
        Fe.paddedLaneEvaluation profile.laneCovers
          (fun lane => CarriedEvaluationResidual.residual
            ConcreteCarrier.baseOps ops K.embed data.carriedData {
              running := running
              matrix := matrix
              coefficient := lane
            })
          (sampled.toCubePoint ops))).symm.trans
      (Fe.InitialSum.paddedLaneEvaluation_selectorSum
        profile.laneCovers _ (cubePoint alpha))

/-- Evaluating the zero-extended active-lane MLE at one live Boolean lane
recovers that exact authoritative lane value. -/
theorem paddedLaneEvaluation_at_liveLane
    {shape : SemanticShape}
    (profile : Fe.SupportedProfile shape PiCcsDomains.production.fe)
    (values : Fin ringDegree -> K)
    (selected : Fin ringDegree) :
    Fe.paddedLaneEvaluation profile.laneCovers values
        ((NumericBooleanDomain.vertex
          PiCcsDomains.production.fe.laneVariables
          (Fe.liveLane profile.laneCovers selected)).toCubePoint ops) =
      values selected := by
  let selectedVertex :=
    NumericBooleanDomain.vertex
      PiCcsDomains.production.fe.laneVariables
      (Fe.liveLane profile.laneCovers selected)
  have vertexInjective :
      Function.Injective
        (NumericBooleanDomain.vertex
          PiCcsDomains.production.fe.laneVariables) := by
    intro left right equal
    apply Fin.ext
    calc
      left.val =
          NumericBooleanDomain.index
            (NumericBooleanDomain.vertex
              PiCcsDomains.production.fe.laneVariables left) :=
        (NumericBooleanDomain.index_vertex _ left).symm
      _ = NumericBooleanDomain.index
            (NumericBooleanDomain.vertex
              PiCcsDomains.production.fe.laneVariables right) := by
          rw [equal]
      _ = right.val :=
        NumericBooleanDomain.index_vertex _ right
  have liveVertex_eq_iff (lane : Fin ringDegree) :
      NumericBooleanDomain.vertex
          PiCcsDomains.production.fe.laneVariables
          (Fe.liveLane profile.laneCovers lane) =
        selectedVertex ↔ lane = selected := by
    constructor
    · intro equal
      have liveEqual := vertexInjective equal
      apply Fin.ext
      simpa using congrArg Fin.val liveEqual
    · intro equal
      subst lane
      rfl
  unfold Fe.paddedLaneEvaluation
  calc
    FiniteSumAlgebra.sumMap ops (canonicalFinIndices ringDegree)
        (fun lane =>
          K.mul
            (NumericBooleanDomain.tensorWeight ops
              (Fe.liveLane profile.laneCovers lane)
              (selectedVertex.toCubePoint ops))
            (values lane)) =
      FiniteSumAlgebra.sumMap ops (canonicalFinIndices ringDegree)
        (fun lane =>
          if lane = selected then values lane else K.zero) := by
            apply FiniteSumAlgebra.sumMap_congr
            intro lane _member
            rw [NumericBooleanDomain.tensorWeight_eq_equalityWeight,
              BooleanReproduction.equalityWeight_toCubePoint ops laws]
            by_cases equal : lane = selected
            · subst lane
              rw [if_pos rfl, if_pos rfl]
              change ops.mul ops.one (values selected) = values selected
              exact laws.one_mul _
            · have vertexNe :
                  NumericBooleanDomain.vertex
                      PiCcsDomains.production.fe.laneVariables
                      (Fe.liveLane profile.laneCovers lane) ≠
                    selectedVertex := by
                intro vertexEqual
                exact equal ((liveVertex_eq_iff lane).mp vertexEqual)
              rw [if_neg vertexNe, if_neg equal]
              exact zero_mul (values lane)
    _ = values selected :=
      BooleanReproduction.sumMap_ite_eq_of_mem_nodup ops laws
        (canonicalFinIndices ringDegree) selected values
        (by simp [canonicalFinIndices])
        (canonicalFinIndices_nodup ringDegree)

/-- One nonzero carried coordinate supplies a nonzero lane controller. -/
theorem carriedTable_nonzero
    {shape : SemanticShape}
    (profile : Fe.SupportedProfile shape PiCcsDomains.production.fe)
    (data : Data shape)
    (coordinate : CarriedCoordinate shape.paperShape)
    (nonzero :
      CarriedEvaluationResidual.residual
        ConcreteCarrier.baseOps ops K.embed data.carriedData coordinate ≠
          K.zero) :
    ¬ (carriedTable profile data coordinate.running coordinate.matrix
      ).AllEntriesZero ops := by
  intro allZero
  have selectedZero :=
    (BooleanTable.tabulate_allEntriesZero_iff ops _).mp allZero
      (NumericBooleanDomain.vertex
        PiCcsDomains.production.fe.laneVariables
        (Fe.liveLane profile.laneCovers coordinate.coefficient))
  apply nonzero
  simpa [carriedTable,
    paddedLaneEvaluation_at_liveLane profile
      (fun lane => CarriedEvaluationResidual.residual
        ConcreteCarrier.baseOps ops K.embed data.carriedData {
          running := coordinate.running
          matrix := coordinate.matrix
          coefficient := lane
        })
      coordinate.coefficient] using selectedZero

/-! ## Exact dense gamma coefficient encoding -/

/-- Fresh row-specialized values before the sign forced by
`initial - sum Q`. -/
def freshValues
    {shape : SemanticShape}
    (data : Data shape)
    (betaR : BetaRWord shape) : List K :=
  (canonicalFinIndices shape.freshCount).map fun fresh =>
    (freshTable data fresh).evaluate ops (cubePoint betaR)

/-- Signed fresh FE coefficient block. -/
def freshCoefficients
    {shape : SemanticShape}
    (data : Data shape)
    (betaR : BetaRWord shape) : List K :=
  (freshValues data betaR).map ops.neg

theorem freshCoefficients_length
    {shape : SemanticShape}
    (data : Data shape)
    (betaR : BetaRWord shape) :
    (freshCoefficients data betaR).length = shape.freshCount := by
  simp [freshCoefficients, freshValues, canonicalFinIndices_length]

/-- Exact evaluation of the signed fresh block. -/
theorem freshCoefficients_evaluate
    {shape : SemanticShape}
    (data : Data shape)
    (betaR : BetaRWord shape)
    (gamma : K) :
    Message.evaluateCoefficients ops.toOps gamma
        (freshCoefficients data betaR) =
      ops.neg
        (Fe.InitialSum.freshResidualMix data
          (coins (fun _ => K.zero) (fun _ => K.zero) betaR gamma)) := by
  rw [freshCoefficients,
    SignedCoefficientPolynomial.evaluate_map_neg ops laws]
  unfold freshValues
  rw [SignedCoefficientPolynomial.evaluate_canonicalFinMap_eq_gammaSum
    ops laws]
  unfold Fe.InitialSum.freshResidualMix
  apply congrArg ops.neg
  apply FiniteSumAlgebra.sumMap_congr
  intro fresh _member
  unfold SignedJointIdentity.gammaTerm
  apply congrArg
  exact freshTable_evaluate data fresh betaR

/-- One matrix's running carried coefficients before its production stride
padding. -/
def runningCoefficients
    {shape : SemanticShape}
    (profile : Fe.SupportedProfile shape PiCcsDomains.production.fe)
    (data : Data shape)
    (alpha : AlphaWord)
    (matrix : Fin shape.matrixCount) : List K :=
  (canonicalFinIndices shape.runningCount).map fun running =>
    (carriedTable profile data running matrix).evaluate ops
      (cubePoint alpha)

theorem runningCoefficients_length
    {shape : SemanticShape}
    (profile : Fe.SupportedProfile shape PiCcsDomains.production.fe)
    (data : Data shape)
    (alpha : AlphaWord)
    (matrix : Fin shape.matrixCount) :
    (runningCoefficients profile data alpha matrix).length =
      shape.runningCount := by
  simp [runningCoefficients, canonicalFinIndices_length]

/-- Every matrix occupies one full `sourceCount` stride: the active running
coefficients followed by `freshCount` explicit zeros. -/
def matrixBlock
    {shape : SemanticShape}
    (profile : Fe.SupportedProfile shape PiCcsDomains.production.fe)
    (data : Data shape)
    (alpha : AlphaWord)
    (matrix : Fin shape.matrixCount) : List K :=
  runningCoefficients profile data alpha matrix ++
    List.replicate shape.freshCount K.zero

theorem matrixBlock_length
    {shape : SemanticShape}
    (profile : Fe.SupportedProfile shape PiCcsDomains.production.fe)
    (data : Data shape)
    (alpha : AlphaWord)
    (matrix : Fin shape.matrixCount) :
    (matrixBlock profile data alpha matrix).length =
      shape.sourceCount := by
  simp [matrixBlock, runningCoefficients_length,
    SemanticShape.sourceCount, Nat.add_comm]

/-- Canonical matrix order with the running source changing fastest. -/
def matrixBlocks
    {shape : SemanticShape}
    (profile : Fe.SupportedProfile shape PiCcsDomains.production.fe)
    (data : Data shape)
    (alpha : AlphaWord) : List K :=
  (canonicalFinIndices shape.matrixCount).flatMap fun matrix =>
    matrixBlock profile data alpha matrix

private theorem evaluate_flatMap_fixed
    {Index : Type}
    (gamma : K)
    (blockLength : Nat)
    (indices : List Index)
    (block : Index -> List K)
    (eachLength :
      ∀ index, index ∈ indices -> (block index).length = blockLength) :
    Message.evaluateCoefficients ops.toOps gamma
        (indices.flatMap block) =
      Message.evaluateCoefficients ops.toOps
        (TargetPolynomial.power ops.toOps gamma blockLength)
        (indices.map fun index =>
          Message.evaluateCoefficients ops.toOps gamma (block index)) := by
  induction indices with
  | nil =>
      change ops.zero = ops.zero
      rfl
  | cons index indices inductionHypothesis =>
      rw [List.flatMap_cons, List.map_cons]
      rw [SignedCoefficientPolynomial.evaluate_append ops laws]
      simp only [Message.evaluateCoefficients]
      rw [eachLength index (by simp)]
      have tailLengths :
          ∀ prior, prior ∈ indices ->
            (block prior).length = blockLength := by
        intro prior member
        exact eachLength prior (by simp [member])
      rw [inductionHypothesis tailLengths]

private def shiftLaws : TargetPolynomial.ShiftLaws ops.toOps where
  one_mul := laws.one_mul
  mul_assoc := laws.mul_assoc
  mul_zero := laws.mul_zero
  mul_add := laws.left_distrib

private theorem power_power
    (gamma : K)
    (blockLength : Nat) :
    ∀ exponent,
      TargetPolynomial.power ops.toOps
          (TargetPolynomial.power ops.toOps gamma blockLength) exponent =
        TargetPolynomial.power ops.toOps gamma (exponent * blockLength)
  | 0 => by simp [TargetPolynomial.power]
  | exponent + 1 => by
      simp only [TargetPolynomial.power, Nat.succ_mul]
      rw [power_power gamma blockLength exponent]
      rw [TargetPolynomial.power_add ops.toOps shiftLaws]
      exact laws.mul_comm _ _

private theorem gammaTerm_stride
    (gamma value : K)
    (blockLength blockIndex innerIndex : Nat) :
    SignedJointIdentity.gammaTerm ops
        (TargetPolynomial.power ops.toOps gamma blockLength)
        blockIndex
        (SignedJointIdentity.gammaTerm ops gamma innerIndex value) =
      SignedJointIdentity.gammaTerm ops gamma
        (innerIndex + blockIndex * blockLength) value := by
  unfold SignedJointIdentity.gammaTerm
  rw [power_power]
  rw [TargetPolynomial.power_add ops.toOps shiftLaws]
  calc
    ops.mul
        (TargetPolynomial.power ops.toOps gamma
          (blockIndex * blockLength))
        (ops.mul
          (TargetPolynomial.power ops.toOps gamma innerIndex) value) =
      ops.mul
        (ops.mul
          (TargetPolynomial.power ops.toOps gamma
            (blockIndex * blockLength))
          (TargetPolynomial.power ops.toOps gamma innerIndex))
        value := (laws.mul_assoc _ _ _).symm
    _ = ops.mul
        (ops.mul
          (TargetPolynomial.power ops.toOps gamma innerIndex)
          (TargetPolynomial.power ops.toOps gamma
            (blockIndex * blockLength)))
        value := by
          apply congrArg (fun factor => ops.mul factor value)
          exact laws.mul_comm _ _
    _ = _ := rfl

private theorem gammaTerm_shift
    (gamma value : K)
    (shift exponent : Nat) :
    SignedJointIdentity.gammaTerm ops gamma (shift + exponent) value =
      ops.mul
        (TargetPolynomial.power ops.toOps gamma shift)
        (SignedJointIdentity.gammaTerm ops gamma exponent value) := by
  unfold SignedJointIdentity.gammaTerm
  rw [TargetPolynomial.power_add ops.toOps shiftLaws]
  exact laws.mul_assoc _ _ _

/-- Running specialization within one matrix is the exact local gamma
polynomial for that matrix. -/
theorem runningCoefficients_evaluate
    {shape : SemanticShape}
    (profile : Fe.SupportedProfile shape PiCcsDomains.production.fe)
    (data : Data shape)
    (alpha : AlphaWord)
    (matrix : Fin shape.matrixCount)
    (gamma : K) :
    Message.evaluateCoefficients ops.toOps gamma
        (runningCoefficients profile data alpha matrix) =
      FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices shape.runningCount) fun running =>
          SignedJointIdentity.gammaTerm ops gamma running.val <|
            Fe.paddedLaneEvaluation profile.laneCovers
              (fun lane => CarriedEvaluationResidual.residual
                ConcreteCarrier.baseOps ops K.embed data.carriedData {
                  running := running
                  matrix := matrix
                  coefficient := lane
                })
              (cubePoint alpha) := by
  unfold runningCoefficients
  rw [SignedCoefficientPolynomial.evaluate_canonicalFinMap_eq_gammaSum
    ops laws]
  apply FiniteSumAlgebra.sumMap_congr
  intro running _member
  unfold SignedJointIdentity.gammaTerm
  apply congrArg
  exact carriedTable_evaluate profile data running matrix alpha

/-- Explicit stride padding does not change one matrix's local evaluation. -/
theorem matrixBlock_evaluate
    {shape : SemanticShape}
    (profile : Fe.SupportedProfile shape PiCcsDomains.production.fe)
    (data : Data shape)
    (alpha : AlphaWord)
    (matrix : Fin shape.matrixCount)
    (gamma : K) :
    Message.evaluateCoefficients ops.toOps gamma
        (matrixBlock profile data alpha matrix) =
      Message.evaluateCoefficients ops.toOps gamma
        (runningCoefficients profile data alpha matrix) := by
  unfold matrixBlock
  rw [SignedCoefficientPolynomial.evaluate_append ops laws]
  rw [evaluate_replicate_zero]
  calc
    ops.add
        (Message.evaluateCoefficients ops.toOps gamma
          (runningCoefficients profile data alpha matrix))
        (ops.mul
          (TargetPolynomial.power ops.toOps gamma
            (runningCoefficients profile data alpha matrix).length)
          ops.zero) =
      ops.add
        (Message.evaluateCoefficients ops.toOps gamma
          (runningCoefficients profile data alpha matrix))
        ops.zero := by
          apply congrArg (ops.add
            (Message.evaluateCoefficients ops.toOps gamma
              (runningCoefficients profile data alpha matrix)))
          exact laws.mul_zero _
    _ = _ := laws.add_zero _

/-- Compacted matrix-level view used only to state the exact evaluation of
the fixed-width dense serialization. -/
def matrixResidualMix
    {shape : SemanticShape}
    (profile : Fe.SupportedProfile shape PiCcsDomains.production.fe)
    (data : Data shape)
    (alpha : AlphaWord)
    (gamma : K) : K :=
  FiniteSumAlgebra.sumMap ops
    (canonicalFinIndices shape.matrixCount) fun matrix =>
      FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices shape.runningCount) fun running =>
          SignedJointIdentity.gammaTerm ops gamma
            (running.val + matrix.val * shape.sourceCount) <|
            Fe.paddedLaneEvaluation profile.laneCovers
              (fun lane => CarriedEvaluationResidual.residual
                ConcreteCarrier.baseOps ops K.embed data.carriedData {
                  running := running
                  matrix := matrix
                  coefficient := lane
                })
              (cubePoint alpha)

/-- The concatenated fixed-width matrix blocks evaluate to the exact
matrix/running production exponent schedule. -/
theorem matrixBlocks_evaluate
    {shape : SemanticShape}
    (profile : Fe.SupportedProfile shape PiCcsDomains.production.fe)
    (data : Data shape)
    (alpha : AlphaWord)
    (gamma : K) :
    Message.evaluateCoefficients ops.toOps gamma
        (matrixBlocks profile data alpha) =
      matrixResidualMix profile data alpha gamma := by
  unfold matrixBlocks
  rw [evaluate_flatMap_fixed gamma shape.sourceCount
    (canonicalFinIndices shape.matrixCount)
    (matrixBlock profile data alpha)]
  · rw [SignedCoefficientPolynomial.evaluate_canonicalFinMap_eq_gammaSum
      ops laws]
    unfold matrixResidualMix
    apply FiniteSumAlgebra.sumMap_congr
    intro matrix _matrixMember
    rw [matrixBlock_evaluate, runningCoefficients_evaluate]
    change
      ops.mul
          (TargetPolynomial.power ops.toOps
            (TargetPolynomial.power ops.toOps gamma shape.sourceCount)
            matrix.val)
          (FiniteSumAlgebra.sumMap ops
            (canonicalFinIndices shape.runningCount) fun running =>
              SignedJointIdentity.gammaTerm ops gamma running.val <|
                Fe.paddedLaneEvaluation profile.laneCovers
                  (fun lane => CarriedEvaluationResidual.residual
                    ConcreteCarrier.baseOps ops K.embed data.carriedData {
                      running := running
                      matrix := matrix
                      coefficient := lane
                    })
                  (cubePoint alpha)) =
        FiniteSumAlgebra.sumMap ops
          (canonicalFinIndices shape.runningCount) fun running =>
            SignedJointIdentity.gammaTerm ops gamma
              (running.val + matrix.val * shape.sourceCount) <|
              Fe.paddedLaneEvaluation profile.laneCovers
                (fun lane => CarriedEvaluationResidual.residual
                  ConcreteCarrier.baseOps ops K.embed data.carriedData {
                    running := running
                    matrix := matrix
                    coefficient := lane
                  })
                (cubePoint alpha)
    rw [← FiniteSumAlgebra.sumMap_mul_left ops laws]
    apply FiniteSumAlgebra.sumMap_congr
    intro running _runningMember
    exact gammaTerm_stride gamma
      (Fe.paddedLaneEvaluation profile.laneCovers
        (fun lane => CarriedEvaluationResidual.residual
          ConcreteCarrier.baseOps ops K.embed data.carriedData {
            running := running
            matrix := matrix
            coefficient := lane
          })
        (cubePoint alpha))
      shape.sourceCount matrix.val running.val
  · intro matrix _member
    exact matrixBlock_length profile data alpha matrix

/-- The explicit `freshCount` shift of the compacted matrix mix is exactly
the production carried residual polynomial. -/
theorem shiftedMatrixResidualMix_eq_carriedResidualMix
    {shape : SemanticShape}
    (profile : Fe.SupportedProfile shape PiCcsDomains.production.fe)
    (data : Data shape)
    (alpha : AlphaWord)
    (gamma : K) :
    ops.mul
        (TargetPolynomial.power ops.toOps gamma shape.freshCount)
        (matrixResidualMix profile data alpha gamma) =
      Fe.InitialSum.carriedResidualMix profile data
        (coins alpha (fun _ => K.zero) (fun _ => K.zero) gamma) := by
  unfold matrixResidualMix Fe.InitialSum.carriedResidualMix
  rw [← FiniteSumAlgebra.sumMap_mul_left ops laws]
  apply FiniteSumAlgebra.sumMap_congr
  intro matrix _matrixMember
  rw [← FiniteSumAlgebra.sumMap_mul_left ops laws]
  apply FiniteSumAlgebra.sumMap_congr
  intro running _runningMember
  rw [show
    Fe.carriedGammaExponent shape running matrix =
      shape.freshCount +
        (running.val + matrix.val * shape.sourceCount) by
      unfold Fe.carriedGammaExponent
      omega]
  exact (gammaTerm_shift gamma
    (Fe.paddedLaneEvaluation profile.laneCovers
      (fun lane => CarriedEvaluationResidual.residual
        ConcreteCarrier.baseOps ops K.embed data.carriedData {
          running := running
          matrix := matrix
          coefficient := lane
        })
      (cubePoint alpha))
    shape.freshCount
    (running.val + matrix.val * shape.sourceCount)).symm

/-- Local carried block, including the production's leading fresh-count
offset before matrix zero. -/
def carriedLocalCoefficients
    {shape : SemanticShape}
    (profile : Fe.SupportedProfile shape PiCcsDomains.production.fe)
    (data : Data shape)
    (alpha : AlphaWord) : List K :=
  List.replicate shape.freshCount K.zero ++
    matrixBlocks profile data alpha

/-- The dense local carried block has the exact production evaluation. -/
theorem carriedLocalCoefficients_evaluate
    {shape : SemanticShape}
    (profile : Fe.SupportedProfile shape PiCcsDomains.production.fe)
    (data : Data shape)
    (alpha : AlphaWord)
    (gamma : K) :
    Message.evaluateCoefficients ops.toOps gamma
        (carriedLocalCoefficients profile data alpha) =
      Fe.InitialSum.carriedResidualMix profile data
        (coins alpha (fun _ => K.zero) (fun _ => K.zero) gamma) := by
  unfold carriedLocalCoefficients
  rw [SignedCoefficientPolynomial.evaluate_append ops laws]
  rw [evaluate_replicate_zero, List.length_replicate]
  have discardLeadingZero :
      ops.add ops.zero
          (ops.mul
            (TargetPolynomial.power ops.toOps gamma shape.freshCount)
            (Message.evaluateCoefficients ops.toOps gamma
              (matrixBlocks profile data alpha))) =
        ops.mul
          (TargetPolynomial.power ops.toOps gamma shape.freshCount)
          (Message.evaluateCoefficients ops.toOps gamma
            (matrixBlocks profile data alpha)) :=
    laws.zero_add _
  change
    ops.add ops.zero
        (ops.mul
          (TargetPolynomial.power ops.toOps gamma shape.freshCount)
          (Message.evaluateCoefficients ops.toOps gamma
            (matrixBlocks profile data alpha))) =
      Fe.InitialSum.carriedResidualMix profile data
        (coins alpha (fun _ => K.zero) (fun _ => K.zero) gamma)
  rw [discardLeadingZero]
  rw [matrixBlocks_evaluate]
  exact shiftedMatrixResidualMix_eq_carriedResidualMix
    profile data alpha gamma

/-- Complete constant-first production FE coefficients. The prefix is padded
from `freshCount` to `sourceCount`; the local carried block then owns its own
`freshCount` shift and one full `sourceCount` stride per matrix. -/
def gammaCoefficients
    {shape : SemanticShape}
    (profile : Fe.SupportedProfile shape PiCcsDomains.production.fe)
    (data : Data shape)
    (alpha : AlphaWord)
    (betaR : BetaRWord shape) : List K :=
  (freshCoefficients data betaR ++
      List.replicate shape.runningCount K.zero) ++
    carriedLocalCoefficients profile data alpha

/-- Exact evaluation transport from the dense coefficient list to the named
production FE mixed residual. -/
theorem gammaCoefficients_evaluate
    {shape : SemanticShape}
    (profile : Fe.SupportedProfile shape PiCcsDomains.production.fe)
    (data : Data shape)
    (alpha : AlphaWord)
    (betaA : BetaAWord)
    (betaR : BetaRWord shape)
    (gamma : K) :
    Message.evaluateCoefficients ops.toOps gamma
        (gammaCoefficients profile data alpha betaR) =
      Fe.InitialSum.mixedResidual profile data
        (coins alpha betaA betaR gamma) := by
  unfold gammaCoefficients
  rw [SignedCoefficientPolynomial.evaluate_append ops laws]
  rw [SignedCoefficientPolynomial.evaluate_append ops laws]
  rw [freshCoefficients_evaluate, evaluate_replicate_zero]
  have discardFreshPadding :
      ops.add
          (ops.neg (Fe.InitialSum.freshResidualMix data
            (coins (fun _ => K.zero) (fun _ => K.zero) betaR gamma)))
          (ops.mul
            (TargetPolynomial.power ops.toOps gamma
              (freshCoefficients data betaR).length)
            ops.zero) =
        ops.neg (Fe.InitialSum.freshResidualMix data
          (coins (fun _ => K.zero) (fun _ => K.zero) betaR gamma)) := by
    calc
      _ = ops.add
          (ops.neg (Fe.InitialSum.freshResidualMix data
            (coins (fun _ => K.zero) (fun _ => K.zero) betaR gamma)))
          ops.zero := by
            apply congrArg (ops.add
              (ops.neg (Fe.InitialSum.freshResidualMix data
                (coins (fun _ => K.zero) (fun _ => K.zero) betaR gamma))))
            exact laws.mul_zero _
      _ = _ := laws.add_zero _
  change
    ops.add
        (ops.add
          (ops.neg (Fe.InitialSum.freshResidualMix data
            (coins (fun _ => K.zero) (fun _ => K.zero) betaR gamma)))
          (ops.mul
            (TargetPolynomial.power ops.toOps gamma
              (freshCoefficients data betaR).length)
            ops.zero))
        (ops.mul
          (TargetPolynomial.power ops.toOps gamma
            (freshCoefficients data betaR ++
              List.replicate shape.runningCount K.zero).length)
          (Message.evaluateCoefficients ops.toOps gamma
            (carriedLocalCoefficients profile data alpha))) =
      Fe.InitialSum.mixedResidual profile data
        (coins alpha betaA betaR gamma)
  rw [discardFreshPadding]
  rw [List.length_append, freshCoefficients_length,
    List.length_replicate]
  rw [carriedLocalCoefficients_evaluate]
  change
    K.add
        (ops.neg (Fe.InitialSum.freshResidualMix data
          (coins (fun _ => K.zero) (fun _ => K.zero) betaR gamma)))
        (K.mul
          (TargetPolynomial.power ops.toOps gamma
            (shape.freshCount + shape.runningCount))
          (Fe.InitialSum.carriedResidualMix profile data
            (coins alpha (fun _ => K.zero) (fun _ => K.zero) gamma))) =
      Fe.InitialSum.mixedResidual profile data
        (coins alpha betaA betaR gamma)
  rfl

private theorem flatMap_length_fixed
    {Index : Type}
    (indices : List Index)
    (block : Index -> List K)
    (blockLength : Nat)
    (eachLength :
      ∀ index, index ∈ indices -> (block index).length = blockLength) :
    (indices.flatMap block).length = indices.length * blockLength := by
  induction indices with
  | nil =>
      simp only [List.flatMap_nil, List.length_nil, Nat.zero_mul]
  | cons index indices inductionHypothesis =>
      rw [List.flatMap_cons, List.length_append,
        eachLength index (by simp)]
      have tailLengths :
          ∀ prior, prior ∈ indices ->
            (block prior).length = blockLength := by
        intro prior member
        exact eachLength prior (by simp [member])
      rw [inductionHypothesis tailLengths]
      rw [List.length_cons, Nat.succ_mul]
      exact Nat.add_comm _ _

theorem matrixBlocks_length
    {shape : SemanticShape}
    (profile : Fe.SupportedProfile shape PiCcsDomains.production.fe)
    (data : Data shape)
    (alpha : AlphaWord) :
    (matrixBlocks profile data alpha).length =
      shape.matrixCount * shape.sourceCount := by
  unfold matrixBlocks
  rw [flatMap_length_fixed
    (canonicalFinIndices shape.matrixCount)
    (matrixBlock profile data alpha) shape.sourceCount]
  · rw [canonicalFinIndices_length]
  · intro matrix _member
    exact matrixBlock_length profile data alpha matrix

/-- Number of represented coefficients in the conservative dense production
FE encoding. -/
def gammaCoefficientCount (shape : SemanticShape) : Nat :=
  shape.freshCount + (shape.matrixCount + 1) * shape.sourceCount

/-- Root-counting degree derived from the represented coefficient count. -/
def gammaDegree (shape : SemanticShape) : Nat :=
  gammaCoefficientCount shape - 1

theorem gammaCoefficients_length
    {shape : SemanticShape}
    (profile : Fe.SupportedProfile shape PiCcsDomains.production.fe)
    (data : Data shape)
    (alpha : AlphaWord)
    (betaR : BetaRWord shape) :
    (gammaCoefficients profile data alpha betaR).length =
      gammaCoefficientCount shape := by
  unfold gammaCoefficients carriedLocalCoefficients gammaCoefficientCount
  rw [List.length_append, List.length_append, List.length_append,
    freshCoefficients_length, List.length_replicate,
    List.length_replicate, matrixBlocks_length]
  unfold SemanticShape.sourceCount
  simp only [Nat.add_mul, Nat.one_mul]
  ac_rfl

theorem gammaCoefficients_count_eq_degree_add_one
    {shape : SemanticShape}
    (profile : Fe.SupportedProfile shape PiCcsDomains.production.fe)
    (data : Data shape)
    (alpha : AlphaWord)
    (betaR : BetaRWord shape) :
    (gammaCoefficients profile data alpha betaR).length =
      gammaDegree shape + 1 := by
  rw [gammaCoefficients_length]
  unfold gammaDegree
  have positive : 0 < gammaCoefficientCount shape := by
    unfold gammaCoefficientCount
    exact Nat.lt_of_lt_of_le profile.fresh_nonempty
      (Nat.le_add_right shape.freshCount
        ((shape.matrixCount + 1) * shape.sourceCount))
  omega

private theorem neg_eq_zero_implies_zero
    {value : K}
    (negZero : ops.neg value = ops.zero) :
    value = ops.zero := by
  calc
    value = ops.add value ops.zero := (laws.add_zero value).symm
    _ = ops.add value (ops.neg value) := by rw [negZero]
    _ = ops.zero := laws.add_neg value

/-! ## Selector controller extracted from semantic nontruth -/

/-- One verifier-independent residual controls the sampled selector stage.
Fresh residuals are controlled by `betaR`; carried coefficient residuals are
controlled by `alpha`. -/
inductive Controller
    {shape : SemanticShape}
    (profile : Fe.SupportedProfile shape PiCcsDomains.production.fe)
    (data : Data shape) : Type where
  | fresh
      (source : Fin shape.freshCount)
      (nonzero : ¬ (freshTable data source).AllEntriesZero ops)
  | carried
      (coordinate : CarriedCoordinate shape.paperShape)
      (nonzero :
        ¬ (carriedTable profile data coordinate.running coordinate.matrix
          ).AllEntriesZero ops)

private theorem controllerExists
    {shape : SemanticShape}
    (profile : Fe.SupportedProfile shape PiCcsDomains.production.fe)
    (data : Data shape)
    (residualsNonzero : ¬ Semantics.Fe.ResidualsZero data) :
    Nonempty (Controller profile data) := by
  classical
  by_cases freshZero :
      ∀ source,
        (data.freshBatch.residualTables ConcreteCarrier.baseOps source
          ).AllEntriesZero ConcreteCarrier.baseOps
  · have carriedNonzero :
        ¬ ∀ coordinate,
          CarriedEvaluationResidual.residual
              ConcreteCarrier.baseOps ops K.embed data.carriedData
              coordinate = K.zero := by
      intro carriedZero
      exact residualsNonzero ⟨freshZero, carriedZero⟩
    obtain ⟨coordinate, coordinateNonzero⟩ :=
      Classical.not_forall.mp carriedNonzero
    exact ⟨.carried coordinate
      (carriedTable_nonzero profile data coordinate coordinateNonzero)⟩
  · obtain ⟨source, sourceNonzero⟩ :=
      Classical.not_forall.mp freshZero
    exact ⟨.fresh source (freshTable_nonzero data source sourceNonzero)⟩

/-- A nonzero independent FE residual family supplies one exact controller.
No controller is prover input. -/
noncomputable def controllerOfResidualsNonzero
    {shape : SemanticShape}
    (profile : Fe.SupportedProfile shape PiCcsDomains.production.fe)
    (data : Data shape)
    (residualsNonzero : ¬ Semantics.Fe.ResidualsZero data) :
    Controller profile data :=
  Classical.choice (controllerExists profile data residualsNonzero)

/-- Fresh selector-root event. It is identically false when the extracted
controller belongs to the carried family. -/
noncomputable def freshSelectorBad
    {shape : SemanticShape}
    {profile : Fe.SupportedProfile shape PiCcsDomains.production.fe}
    {data : Data shape}
    (controller : Controller profile data)
    (engineHead : (AlphaWord × BetaAWord) × BetaRWord shape) : Bool :=
  match controller with
  | .fresh source _ =>
      decide
        ((freshTable data source).evaluateCoordinates ops
          (List.ofFn engineHead.2) = ops.zero)
  | .carried _ _ => false

/-- Carried selector-root event. It is identically false when the extracted
controller belongs to the fresh family. -/
noncomputable def carriedSelectorBad
    {shape : SemanticShape}
    {profile : Fe.SupportedProfile shape PiCcsDomains.production.fe}
    {data : Data shape}
    (controller : Controller profile data)
    (engineHead : (AlphaWord × BetaAWord) × BetaRWord shape) : Bool :=
  match controller with
  | .fresh _ _ => false
  | .carried coordinate _ =>
      decide
        ((carriedTable profile data coordinate.running coordinate.matrix
          ).evaluateCoordinates ops (List.ofFn engineHead.1.1) = ops.zero)

private theorem freshCoefficient_mem
    {shape : SemanticShape}
    (profile : Fe.SupportedProfile shape PiCcsDomains.production.fe)
    (data : Data shape)
    (source : Fin shape.freshCount)
    (alpha : AlphaWord)
    (betaR : BetaRWord shape) :
    ops.neg ((freshTable data source).evaluate ops (cubePoint betaR)) ∈
      gammaCoefficients profile data alpha betaR := by
  apply List.mem_append_left
  apply List.mem_append_left
  unfold freshCoefficients
  apply List.mem_map.mpr
  refine ⟨(freshTable data source).evaluate ops (cubePoint betaR), ?_, rfl⟩
  unfold freshValues
  exact List.mem_map.mpr ⟨source, by simp [canonicalFinIndices], rfl⟩

private theorem carriedCoefficient_mem
    {shape : SemanticShape}
    (profile : Fe.SupportedProfile shape PiCcsDomains.production.fe)
    (data : Data shape)
    (coordinate : CarriedCoordinate shape.paperShape)
    (alpha : AlphaWord)
    (betaR : BetaRWord shape) :
    (carriedTable profile data coordinate.running coordinate.matrix
      ).evaluate ops (cubePoint alpha) ∈
      gammaCoefficients profile data alpha betaR := by
  apply List.mem_append_right
  unfold carriedLocalCoefficients
  apply List.mem_append_right
  unfold matrixBlocks
  apply List.mem_flatMap.mpr
  refine ⟨coordinate.matrix, by simp [canonicalFinIndices], ?_⟩
  unfold matrixBlock
  apply List.mem_append_left
  unfold runningCoefficients
  exact List.mem_map.mpr
    ⟨coordinate.running, by simp [canonicalFinIndices], rfl⟩

/-- Outside both applicable selector-root events, the exact dense gamma
coefficient list is nonzero. -/
theorem Controller.controls
    {shape : SemanticShape}
    {profile : Fe.SupportedProfile shape PiCcsDomains.production.fe}
    {data : Data shape}
    (controller : Controller profile data)
    (engineHead : (AlphaWord × BetaAWord) × BetaRWord shape)
    (freshGood : freshSelectorBad controller engineHead = false)
    (carriedGood : carriedSelectorBad controller engineHead = false) :
    ¬ CoefficientRootCounting.AllZero ops
      (gammaCoefficients profile data engineHead.1.1 engineHead.2) := by
  cases controller with
  | fresh source tableNonzero =>
      have evaluationNonzero :
          (freshTable data source).evaluate ops
              (cubePoint engineHead.2) ≠ ops.zero := by
        simpa [freshSelectorBad, BooleanTable.evaluate, cubePoint] using
          freshGood
      intro allZero
      have negativeZero :=
        allZero
          (ops.neg ((freshTable data source).evaluate ops
            (cubePoint engineHead.2)))
          (freshCoefficient_mem profile data source engineHead.1.1
            engineHead.2)
      exact evaluationNonzero (neg_eq_zero_implies_zero negativeZero)
  | carried coordinate tableNonzero =>
      have evaluationNonzero :
          (carriedTable profile data coordinate.running coordinate.matrix
            ).evaluate ops (cubePoint engineHead.1.1) ≠ ops.zero := by
        simpa [carriedSelectorBad, BooleanTable.evaluate, cubePoint] using
          carriedGood
      intro allZero
      exact evaluationNonzero
        (allZero
          ((carriedTable profile data coordinate.running coordinate.matrix
            ).evaluate ops (cubePoint engineHead.1.1))
          (carriedCoefficient_mem profile data coordinate engineHead.1.1
            engineHead.2))

end Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary.IdealInteractiveFeMixing
