import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentityConcreteAlgebra
import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PaperForkModules
import Nightstream.SuperNeo.Concrete.Phi81StrongSet
import Nightstream.SuperNeo.Folding.PiRLC.PaperForkExtraction

/-!
Concrete coordinate-fork extraction algebra for `PaddedRowIdentity`.

Owns: exact module structures for the selected complete assignment, Ajtai
commitment, public prefix, and full Phi81 evaluation family; preservation of
those structures by the semantic maps; the total evaluation-array combination
contract; corrected ambient-norm coverage; and the production strong-set
bridge.

Does not own: the analytic low-norm invertibility theorem, Ajtai/MSIS binding,
Poseidon2 transcript security, Rust, R1CS, or generated matrix bytes.

`LowNormInvertibility` remains the sole explicit mathematical premise of the
strong-set construction. No algebraic law is accepted from a caller.
-/

set_option autoImplicit false
set_option maxHeartbeats 800000
set_option maxRecDepth 4000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityConcreteExtraction

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiRLC

namespace ConcreteAlgebra
export PaddedRowIdentityConcreteAlgebra
  (Structure Assignment PublicInput Point Evaluation Commitment AjtaiKey
    relationShape verifierRows canonicalStructure evaluationFamily semantics
    semantics_evaluations_size evaluationZero combineEvaluationFamily
    combineEvaluations piRlcAlgebra)
end ConcreteAlgebra

namespace Modules
export PaperForkModules
  (ringFOps ringFLaws ringFModule ringFModuleLaws ringFNeg assignmentModule
    assignmentModuleLaws pointwiseModule pointwiseModuleLaws ringKModule
    ringKModuleLaws)
end Modules

abbrev Assignment := ConcreteAlgebra.Assignment
abbrev PublicInput := ConcreteAlgebra.PublicInput
abbrev Evaluation := ConcreteAlgebra.Evaluation
abbrev Commitment := ConcreteAlgebra.Commitment
abbrev AjtaiKey := ConcreteAlgebra.AjtaiKey

/-! ## Exact module carriers -/

def assignmentModule : PaperForkAlgebra.ModuleOps RingF Assignment :=
  Modules.assignmentModule ConcreteAlgebra.relationShape.logicalWidth

theorem assignmentModuleLaws :
    PaperForkAlgebra.ModuleLaws Modules.ringFOps assignmentModule :=
  Modules.assignmentModuleLaws ConcreteAlgebra.relationShape.logicalWidth

def commitmentModule : PaperForkAlgebra.ModuleOps RingF Commitment :=
  Modules.pointwiseModule (Fin ConcreteAlgebra.verifierRows) Modules.ringFModule

theorem commitmentModuleLaws :
    PaperForkAlgebra.ModuleLaws Modules.ringFOps commitmentModule :=
  Modules.pointwiseModuleLaws
    (Fin ConcreteAlgebra.verifierRows) Modules.ringFModule Modules.ringFModuleLaws

/-- Pointwise additive inverse on the exact public prefix. -/
def publicNeg (input : PublicInput) : PublicInput :=
  fun column => -input column

def publicInputModule : PaperForkAlgebra.ModuleOps RingF PublicInput where
  zero := PiRLCAlgebra.PublicInput.publicZero
  add := PiRLCAlgebra.PublicInput.publicAdd
  neg := publicNeg
  smul := PiRLCAlgebra.PublicInput.publicAct

private theorem publicBlock_injective :
    Function.Injective
      (fun input : PublicInput =>
        PiRLCAlgebra.PublicInput.publicBlock input) := by
  intro left right equal
  funext column
  have atCoordinate := congrFun
    (congrFun equal
      (PiRLCAlgebra.PublicInput.publicBlockIndex
        ConcreteAlgebra.relationShape column))
    (PiRLCAlgebra.PublicInput.publicLaneIndex column)
  have indexValue :
      column.val / ringDegree * ringDegree + column.val % ringDegree =
        column.val :=
    Nat.div_add_mod' column.val ringDegree
  simpa only [PiRLCAlgebra.PublicInput.publicBlock,
    PiRLCAlgebra.PublicInput.publicBlockIndex,
    PiRLCAlgebra.PublicInput.publicLaneIndex,
    indexValue] using atCoordinate

private theorem publicBlock_act
    (scalar : RingF) (input : PublicInput)
    (block : Fin ConcreteAlgebra.relationShape.publicRingColumns) :
    PiRLCAlgebra.PublicInput.publicBlock
        (PiRLCAlgebra.PublicInput.publicAct scalar input) block =
      ringFMul scalar (PiRLCAlgebra.PublicInput.publicBlock input block) := by
  funext lane
  let column : Fin ConcreteAlgebra.relationShape.publicWidth :=
    ⟨block.val * ringDegree + lane.val, by
      have blockLt := block.isLt
      have laneLt := lane.isLt
      change block.val * 54 + lane.val < 54 *
        ConcreteAlgebra.relationShape.publicRingColumns
      change block.val < ConcreteAlgebra.relationShape.publicRingColumns at blockLt
      change lane.val < 54 at laneLt
      omega⟩
  change ringFMul scalar
      (PiRLCAlgebra.PublicInput.publicBlock input
        (PiRLCAlgebra.PublicInput.publicBlockIndex
          ConcreteAlgebra.relationShape column))
      (PiRLCAlgebra.PublicInput.publicLaneIndex column) =
    ringFMul scalar
      (PiRLCAlgebra.PublicInput.publicBlock input block) lane
  have blockEq :
      PiRLCAlgebra.PublicInput.publicBlockIndex
          ConcreteAlgebra.relationShape column = block := by
    apply Fin.ext
    change (block.val * 54 + lane.val) / 54 = block.val
    have laneLt := lane.isLt
    change lane.val < 54 at laneLt
    rw [Nat.mul_comm block.val 54]
    rw [Nat.mul_add_div (by decide)]
    simp [Nat.div_eq_of_lt laneLt]
  have laneEq :
      PiRLCAlgebra.PublicInput.publicLaneIndex column = lane := by
    apply Fin.ext
    change (block.val * 54 + lane.val) % 54 = lane.val
    have laneLt := lane.isLt
    change lane.val < 54 at laneLt
    rw [Nat.mul_add_mod_self_right]
    exact Nat.mod_eq_of_lt laneLt
  rw [blockEq, laneEq]

private theorem publicBlock_add (left right : PublicInput)
    (block : Fin ConcreteAlgebra.relationShape.publicRingColumns) :
    PiRLCAlgebra.PublicInput.publicBlock
        (PiRLCAlgebra.PublicInput.publicAdd left right) block =
      ringFAdd
        (PiRLCAlgebra.PublicInput.publicBlock left block)
        (PiRLCAlgebra.PublicInput.publicBlock right block) := by
  rfl

private theorem publicBlock_neg (value : PublicInput)
    (block : Fin ConcreteAlgebra.relationShape.publicRingColumns) :
    PiRLCAlgebra.PublicInput.publicBlock (publicNeg value) block =
      Modules.ringFNeg
        (PiRLCAlgebra.PublicInput.publicBlock value block) := by
  rfl

private theorem publicBlock_zero
    (block : Fin ConcreteAlgebra.relationShape.publicRingColumns) :
    PiRLCAlgebra.PublicInput.publicBlock
        (PiRLCAlgebra.PublicInput.publicZero : PublicInput) block = ringFZero := by
  rfl

theorem publicInputModuleLaws :
    PaperForkAlgebra.ModuleLaws Modules.ringFOps publicInputModule where
  add_assoc := by
    intro left middle right
    apply publicBlock_injective
    funext block
    simp only [publicInputModule, publicBlock_add]
    exact Modules.ringFLaws.add_assoc _ _ _
  add_comm := by
    intro left right
    apply publicBlock_injective
    funext block
    simp only [publicInputModule, publicBlock_add]
    exact Modules.ringFLaws.add_comm _ _
  zero_add := by
    intro value
    apply publicBlock_injective
    funext block
    simp only [publicInputModule, publicBlock_add, publicBlock_zero]
    exact Modules.ringFLaws.zero_add _
  add_zero := by
    intro value
    apply publicBlock_injective
    funext block
    simp only [publicInputModule, publicBlock_add, publicBlock_zero]
    exact Modules.ringFLaws.add_zero _
  add_neg := by
    intro value
    apply publicBlock_injective
    funext block
    simp only [publicInputModule, publicBlock_add, publicBlock_neg,
      publicBlock_zero]
    exact Modules.ringFLaws.add_neg _
  zero_smul := by
    intro value
    apply publicBlock_injective
    funext block
    simp only [publicInputModule, publicBlock_act, publicBlock_zero]
    exact Modules.ringFModuleLaws.zero_smul _
  add_smul := by
    intro left right value
    apply publicBlock_injective
    funext block
    simp only [publicInputModule, publicBlock_act, publicBlock_add]
    exact Modules.ringFModuleLaws.add_smul _ _ _
  one_smul := by
    intro value
    apply publicBlock_injective
    funext block
    simp only [publicInputModule, publicBlock_act]
    exact Modules.ringFModuleLaws.one_smul _
  mul_smul := by
    intro left right value
    apply publicBlock_injective
    funext block
    simp only [publicInputModule, publicBlock_act]
    exact Modules.ringFModuleLaws.mul_smul _ _ _
  smul_zero := by
    intro scalar
    apply publicBlock_injective
    funext block
    simp only [publicInputModule, publicBlock_act, publicBlock_zero]
    exact Modules.ringFModuleLaws.smul_zero _
  smul_add := by
    intro scalar left right
    apply publicBlock_injective
    funext block
    simp only [publicInputModule, publicBlock_act, publicBlock_add]
    exact Modules.ringFModuleLaws.smul_add _ _ _

def evaluationModule : PaperForkAlgebra.ModuleOps RingF Evaluation :=
  Modules.pointwiseModule (Fin PaddedRowIdentity.shape.matrixCount)
    Modules.ringKModule

theorem evaluationModuleLaws :
    PaperForkAlgebra.ModuleLaws Modules.ringFOps evaluationModule :=
  Modules.pointwiseModuleLaws
    (Fin PaddedRowIdentity.shape.matrixCount)
    Modules.ringKModule Modules.ringKModuleLaws

/-! ## Semantic linear maps -/

private theorem inverse_eq_of_add_eq_zero
    {Value : Type}
    (module : PaperForkAlgebra.ModuleOps RingF Value)
    (laws : PaperForkAlgebra.ModuleLaws Modules.ringFOps module)
    (left right : Value)
    (sumZero : module.add left right = module.zero) :
    module.neg left = right := by
  calc
    module.neg left = module.add (module.neg left) module.zero :=
      (laws.add_zero _).symm
    _ = module.add (module.neg left) (module.add left right) := by
      rw [sumZero]
    _ = module.add (module.add (module.neg left) left) right :=
      (laws.add_assoc _ _ _).symm
    _ = module.add module.zero right := by
      rw [laws.add_comm (module.neg left) left, laws.add_neg]
    _ = right := laws.zero_add right

private theorem map_neg_of_add_zero_smul
    {Source Target : Type}
    (source : PaperForkAlgebra.ModuleOps RingF Source)
    (target : PaperForkAlgebra.ModuleOps RingF Target)
    (sourceLaws : PaperForkAlgebra.ModuleLaws Modules.ringFOps source)
    (targetLaws : PaperForkAlgebra.ModuleLaws Modules.ringFOps target)
    (map : Source -> Target)
    (mapZero : map source.zero = target.zero)
    (mapAdd : forall left right,
      map (source.add left right) = target.add (map left) (map right))
    (value : Source) :
    map (source.neg value) = target.neg (map value) := by
  symm
  apply inverse_eq_of_add_eq_zero target targetLaws
  calc
    target.add (map value) (map (source.neg value)) =
        map (source.add value (source.neg value)) :=
      (mapAdd value (source.neg value)).symm
    _ = map source.zero := by rw [sourceLaws.add_neg]
    _ = target.zero := mapZero

private def linearMapLawsOf
    {Source Target : Type}
    (source : PaperForkAlgebra.ModuleOps RingF Source)
    (target : PaperForkAlgebra.ModuleOps RingF Target)
    (sourceLaws : PaperForkAlgebra.ModuleLaws Modules.ringFOps source)
    (targetLaws : PaperForkAlgebra.ModuleLaws Modules.ringFOps target)
    (map : Source -> Target)
    (mapZero : map source.zero = target.zero)
    (mapAdd : forall left right,
      map (source.add left right) = target.add (map left) (map right))
    (mapSmul : forall scalar value,
      map (source.smul scalar value) = target.smul scalar (map value)) :
    PaperForkExtraction.LinearMapLaws source target map where
  map_sub := by
    intro left right
    unfold PaperForkAlgebra.ModuleOps.sub
    rw [mapAdd]
    rw [map_neg_of_add_zero_smul source target sourceLaws targetLaws
      map mapZero mapAdd]
  map_smul := mapSmul

def commitmentMapLaws (key : AjtaiKey) :
    PaperForkExtraction.LinearMapLaws assignmentModule commitmentModule
      (PiRLCAlgebra.Commitment.commit key) :=
  linearMapLawsOf assignmentModule commitmentModule
    assignmentModuleLaws commitmentModuleLaws
    (PiRLCAlgebra.Commitment.commit key)
    (PiRLCAlgebra.Commitment.commit_zero key)
    (PiRLCAlgebra.Commitment.commit_add key)
    (PiRLCAlgebra.Commitment.commit_act key)

def publicInputMapLaws :
    PaperForkExtraction.LinearMapLaws assignmentModule publicInputModule
      (Phi81Relation.projectPublicInput
        (shape := ConcreteAlgebra.relationShape)) :=
  linearMapLawsOf assignmentModule publicInputModule
    assignmentModuleLaws publicInputModuleLaws
    (Phi81Relation.projectPublicInput
      (shape := ConcreteAlgebra.relationShape))
    PiRLCAlgebra.PublicInput.projectPublicInput_zero
    PiRLCAlgebra.PublicInput.projectPublicInput_add
    PiRLCAlgebra.PublicInput.projectPublicInput_act

private theorem evaluationFamily_zero
    (source : ConcreteAlgebra.Structure) (point : ConcreteAlgebra.Point) :
    ConcreteAlgebra.evaluationFamily source assignmentModule.zero point =
      evaluationModule.zero := by
  funext matrix
  exact BaseLinear.matrixEvaluation_zero
    (ConcreteAlgebra.canonicalStructure source) point matrix

private theorem evaluationFamily_add
    (source : ConcreteAlgebra.Structure)
    (left right : Assignment) (point : ConcreteAlgebra.Point) :
    ConcreteAlgebra.evaluationFamily source
        (assignmentModule.add left right) point =
      evaluationModule.add
        (ConcreteAlgebra.evaluationFamily source left point)
        (ConcreteAlgebra.evaluationFamily source right point) := by
  funext matrix
  exact BaseLinear.matrixEvaluation_add
    (ConcreteAlgebra.canonicalStructure source) left right point matrix

private theorem evaluationFamily_smul
    (source : ConcreteAlgebra.Structure) (scalar : RingF)
    (assignment : Assignment) (point : ConcreteAlgebra.Point) :
    ConcreteAlgebra.evaluationFamily source
        (assignmentModule.smul scalar assignment) point =
      evaluationModule.smul scalar
        (ConcreteAlgebra.evaluationFamily source assignment point) := by
  funext matrix
  exact PiRLC.matrixEvaluation_act
    (ConcreteAlgebra.canonicalStructure source) scalar
    (PiRLC.productOrderLaw scalar) assignment point matrix

def evaluationFamilyMapLaws
    (source : ConcreteAlgebra.Structure) (point : ConcreteAlgebra.Point) :
    PaperForkExtraction.LinearMapLaws assignmentModule evaluationModule
      (fun assignment => ConcreteAlgebra.evaluationFamily source assignment point) :=
  linearMapLawsOf assignmentModule evaluationModule
    assignmentModuleLaws evaluationModuleLaws
    (fun assignment => ConcreteAlgebra.evaluationFamily source assignment point)
    (evaluationFamily_zero source point)
    (fun left right => evaluationFamily_add source left right point)
    (fun scalar assignment =>
      evaluationFamily_smul source scalar assignment point)

private theorem semantic_getD_zero
    (key : AjtaiKey) (source : ConcreteAlgebra.Structure)
    (assignment : Assignment) (point : ConcreteAlgebra.Point) :
    ((ConcreteAlgebra.semantics key).evaluations source assignment point).getD
        0 ConcreteAlgebra.evaluationZero =
      ConcreteAlgebra.evaluationFamily source assignment point := by
  rfl

private theorem semantic_getD_of_ne_zero
    (key : AjtaiKey) (source : ConcreteAlgebra.Structure)
    (assignment : Assignment) (point : ConcreteAlgebra.Point)
    (index : Nat) (indexNe : index ≠ 0) :
    ((ConcreteAlgebra.semantics key).evaluations source assignment point).getD
        index ConcreteAlgebra.evaluationZero =
      ConcreteAlgebra.evaluationZero := by
  change (#[ConcreteAlgebra.evaluationFamily source assignment point]).getD
      index ConcreteAlgebra.evaluationZero = ConcreteAlgebra.evaluationZero
  rw [Array.getD_eq_getD_getElem?]
  have outside :
      (#[ConcreteAlgebra.evaluationFamily source assignment point]).size <=
        index := by
    change 1 <= index
    omega
  rw [Array.getElem?_eq_none outside]
  rfl

def evaluationMapLaws
    (key : AjtaiKey) (source : ConcreteAlgebra.Structure)
    (point : ConcreteAlgebra.Point) (index : Nat) :
    PaperForkExtraction.LinearMapLaws assignmentModule evaluationModule
      (fun assignment =>
        ((ConcreteAlgebra.semantics key).evaluations source assignment point).getD
          index ConcreteAlgebra.evaluationZero) := by
  by_cases indexZero : index = 0
  · subst index
    simpa only [semantic_getD_zero] using
      evaluationFamilyMapLaws source point
  · refine {
      map_sub := ?_
      map_smul := ?_
    }
    · intro left right
      rw [semantic_getD_of_ne_zero key source _ point index indexZero]
      rw [semantic_getD_of_ne_zero key source left point index indexZero]
      rw [semantic_getD_of_ne_zero key source right point index indexZero]
      exact (evaluationModuleLaws.add_neg evaluationModule.zero)
    · intro scalar value
      rw [semantic_getD_of_ne_zero key source _ point index indexZero]
      rw [semantic_getD_of_ne_zero key source value point index indexZero]
      exact (evaluationModuleLaws.smul_zero scalar).symm

/-! ## Verifier combination refinement -/

theorem combineCommitment_eq
    (key : AjtaiKey) {count : Nat}
    (coefficients : Fin count -> RingF)
    (values : Fin count -> Commitment) :
    (ConcreteAlgebra.piRlcAlgebra key).combineCommitment coefficients values =
      PaperForkAlgebra.linearCombination Modules.ringFOps commitmentModule
        coefficients values := by
  change PiRLCAlgebra.Commitment.combineCommitments coefficients values = _
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [PaperForkAlgebra.linearCombination]
      change PiRLCAlgebra.Commitment.commitmentAdd
          (PiRLCAlgebra.Commitment.commitmentAct
            (coefficients 0) (values 0))
          (PiRLCAlgebra.Commitment.combineCommitments
            (fun index => coefficients index.succ)
            (fun index => values index.succ)) = _
      rw [inductionHypothesis
        (fun index => coefficients index.succ)
        (fun index => values index.succ)]
      rfl

theorem combinePublicInput_eq
    {count : Nat}
    (coefficients : Fin count -> RingF)
    (values : Fin count -> PublicInput) :
    PiRLCAlgebra.PublicInput.combinePublicInputs coefficients values =
      PaperForkAlgebra.linearCombination Modules.ringFOps publicInputModule
        coefficients values := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [PaperForkAlgebra.linearCombination]
      change PiRLCAlgebra.PublicInput.publicAdd
          (PiRLCAlgebra.PublicInput.publicAct
            (coefficients 0) (values 0))
          (PiRLCAlgebra.PublicInput.combinePublicInputs
            (fun index => coefficients index.succ)
            (fun index => values index.succ)) = _
      rw [inductionHypothesis
        (fun index => coefficients index.succ)
        (fun index => values index.succ)]
      rfl

private theorem combineEvaluationFamily_eq
    {count : Nat}
    (coefficients : Fin count -> RingF)
    (values : Fin count -> Evaluation) :
    ConcreteAlgebra.combineEvaluationFamily coefficients values =
      PaperForkAlgebra.linearCombination Modules.ringFOps evaluationModule
        coefficients values := by
  funext matrix
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [ConcreteAlgebra.combineEvaluationFamily,
        PiRLCFinite.combineEvaluation,
        PaperForkAlgebra.linearCombination]
      change ringKAdd
          (ringKMul (RingKAction.embedChallenge (coefficients 0))
            (values 0 matrix))
          (ConcreteAlgebra.combineEvaluationFamily
            (fun index => coefficients index.succ)
            (fun index => values index.succ) matrix) =
        ringKAdd
          (ringKMul (RingKAction.embedChallenge (coefficients 0))
            (values 0 matrix))
          ((PaperForkAlgebra.linearCombination Modules.ringFOps
            evaluationModule
            (fun index => coefficients index.succ)
            (fun index => values index.succ)) matrix)
      rw [inductionHypothesis
        (fun index => coefficients index.succ)
        (fun index => values index.succ)]

private theorem linearCombination_zero
    {count : Nat} (coefficients : Fin count -> RingF) :
    PaperForkAlgebra.linearCombination Modules.ringFOps evaluationModule
        coefficients (fun _ => evaluationModule.zero) =
      evaluationModule.zero := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [PaperForkAlgebra.linearCombination]
      rw [evaluationModuleLaws.smul_zero]
      rw [inductionHypothesis (fun index => coefficients index.succ)]
      exact evaluationModuleLaws.zero_add evaluationModule.zero

theorem combineEvaluations_size
    {count : Nat}
    (coefficients : Fin count -> RingF)
    (values : Fin count -> Array Evaluation)
    (expectedSize : Nat)
    (positive : 0 < count)
    (sizes : forall index, (values index).size = expectedSize) :
    (ConcreteAlgebra.combineEvaluations coefficients values).size =
      expectedSize := by
  cases count with
  | zero => omega
  | succ count =>
      change (Array.ofFn fun index : Fin (values 0).size =>
        ConcreteAlgebra.combineEvaluationFamily coefficients fun source =>
          (values source).getD index.val ConcreteAlgebra.evaluationZero).size =
        expectedSize
      rw [Array.size_ofFn, sizes 0]

theorem combineEvaluations_getD
    {count : Nat}
    (coefficients : Fin count -> RingF)
    (values : Fin count -> Array Evaluation)
    (expectedSize index : Nat)
    (positive : 0 < count)
    (sizes : forall source, (values source).size = expectedSize) :
    (ConcreteAlgebra.combineEvaluations coefficients values).getD
        index evaluationModule.zero =
      PaperForkAlgebra.linearCombination Modules.ringFOps evaluationModule
        coefficients
        (fun source => (values source).getD index evaluationModule.zero) := by
  cases count with
  | zero => omega
  | succ count =>
      by_cases active : index < expectedSize
      · have outputLt :
            index < (ConcreteAlgebra.combineEvaluations coefficients values).size := by
          rw [combineEvaluations_size coefficients values expectedSize
            (by omega) sizes]
          exact active
        rw [Array.getD_eq_getD_getElem?,
          Array.getElem?_eq_getElem outputLt]
        rw [Option.getD_some]
        simp only [ConcreteAlgebra.combineEvaluations, Array.getElem_ofFn]
        change ConcreteAlgebra.combineEvaluationFamily coefficients
            (fun source =>
              (values source).getD index ConcreteAlgebra.evaluationZero) = _
        exact combineEvaluationFamily_eq coefficients
          (fun source =>
            (values source).getD index ConcreteAlgebra.evaluationZero)
      · have outputOutside :
            (ConcreteAlgebra.combineEvaluations coefficients values).size <=
              index := by
          rw [combineEvaluations_size coefficients values expectedSize
            (by omega) sizes]
          omega
        rw [Array.getD_eq_getD_getElem?,
          Array.getElem?_eq_none outputOutside]
        change evaluationModule.zero = _
        calc
          evaluationModule.zero =
              PaperForkAlgebra.linearCombination Modules.ringFOps
                evaluationModule coefficients
                (fun _ => evaluationModule.zero) :=
            (linearCombination_zero coefficients).symm
          _ = PaperForkAlgebra.linearCombination Modules.ringFOps
                evaluationModule coefficients
                (fun source =>
                  (values source).getD index evaluationModule.zero) := by
            apply congrArg
              (PaperForkAlgebra.linearCombination Modules.ringFOps
                evaluationModule coefficients)
            funext source
            rw [Array.getD_eq_getD_getElem?]
            have sourceOutside : (values source).size <= index := by
              rw [sizes source]
              omega
            rw [Array.getElem?_eq_none sourceOutside]
            rfl

/-! ## Complete extraction package -/

def extractionAlgebra (key : AjtaiKey) :
    PaperForkExtraction.ExtractionAlgebra
      (ConcreteAlgebra.semantics key) productionGlobalParams
      (ConcreteAlgebra.piRlcAlgebra key) where
  ring := Modules.ringFOps
  ringLaws := Modules.ringFLaws
  assignmentModule := assignmentModule
  assignmentLaws := assignmentModuleLaws
  commitmentModule := commitmentModule
  commitmentLaws := commitmentModuleLaws
  publicInputModule := publicInputModule
  publicInputLaws := publicInputModuleLaws
  evaluationModule := evaluationModule
  evaluationLaws := evaluationModuleLaws
  combineCommitment_eq := combineCommitment_eq key
  combinePublicInput_eq := combinePublicInput_eq
  semanticEvaluations_size_eq := by
    intro system point left right
    rw [ConcreteAlgebra.semantics_evaluations_size,
      ConcreteAlgebra.semantics_evaluations_size]
  combineEvaluations_size := by
    intro count coefficients values expectedSize positive sizes
    exact combineEvaluations_size coefficients values expectedSize positive sizes
  combineEvaluations_getD := by
    intro count coefficients values expectedSize index positive sizes
    exact combineEvaluations_getD coefficients values expectedSize index
      positive sizes
  commitMap := commitmentMapLaws key
  publicInputMap := publicInputMapLaws
  evaluationsMap := evaluationMapLaws key
  correctedNormCoverage := by
    intro assignment column
    rw [PaperCorrections.production_correctedAmbientBoundFor_eq]
    exact PaperCorrections.all_centeredMagnitude_lt_correctedAmbientBound
      (assignment column)

theorem ring_sub_eq_strong_set_sub (left right : RingF) :
    Modules.ringFOps.sub left right =
      Phi81StrongSet.ringFSub left right := by
  funext lane
  simp only [PaperForkAlgebra.CommutativeRingOps.sub, Modules.ringFOps,
    Modules.ringFNeg, Phi81StrongSet.ringFSub, ringFAdd,
    Fin.sub_eq_add_neg]

/-- Production challenge differences are units, conditional only on the
explicit low-norm invertibility theorem boundary. -/
noncomputable def strongSetUnits
    (theorem8 : Phi81StrongSet.LowNormInvertibility) :
    PaperForkExtraction.StrongSetUnits Modules.ringFOps
      PiRLCAlgebra.Challenge.challengeValid where
  differenceUnit := by
    intro left right leftValid rightValid different
    let secure :=
      PiRLCAlgebra.Challenge.pairwiseSecure_of_lowNormInvertibility theorem8
        leftValid rightValid different
    let inverse := Classical.choose secure
    have inverseProperties := Classical.choose_spec secure
    refine {
      inverse := inverse
      inverse_mul := ?_
      mul_inverse := ?_
    }
    · rw [ring_sub_eq_strong_set_sub]
      exact inverseProperties.2
    · rw [ring_sub_eq_strong_set_sub]
      exact inverseProperties.1

/-- The same strong-set witness, typed through the selected extraction
record. This bridge prevents clients from unfolding the full extraction
record only to recover its ring field. -/
noncomputable def extractionStrongSetUnits
    (key : AjtaiKey)
    (theorem8 : Phi81StrongSet.LowNormInvertibility) :
    PaperForkExtraction.StrongSetUnits (extractionAlgebra key).ring
      (ConcreteAlgebra.piRlcAlgebra key).challengeValid := by
  simpa [extractionAlgebra] using strongSetUnits theorem8

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityConcreteExtraction
