import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal

/-!
Owns the proof-only transport rules shared by the PiCCS v1_1 assembler.
It adds no protocol predicate, circuit row, or alternate verifier path.
-/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.CompletenessSupport

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.SumCheck
open NightstreamFPrime.Gadgets.Multilinear
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

private theorem cubePoint_eq_of_coordinates
    {Field : Type} {variableCount : Nat}
    (left right : CubePoint Field variableCount)
    (coordinates : left.coordinates = right.coordinates) : left = right := by
  cases left
  cases right
  simp_all

private theorem proof_ext
    {degree : Nat}
    (left right : Proof degree)
    (rounds : left.piCcsRounds = right.piCcsRounds)
    (output : left.piCcsOutput = right.piCcsOutput)
    (commitments : left.piDecCommitments = right.piDecCommitments)
    (evaluations : left.piDecEvaluations = right.piDecEvaluations) :
    left = right := by
  cases left
  cases right
  simp_all

theorem evalRunning_eq_of_agree_below
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) (left right : Env)
    (below : ExternalInputsBelow interface offset)
    (agrees : ∀ index, index < offset → left index = right index) :
    evalRunning interface offset left = evalRunning interface offset right := by
  unfold evalRunning StatementAbsorption.evalRunning
  congr 1
  · apply cubePoint_eq_of_coordinates
    change List.ofFn (fun coordinate =>
        ((interface.running offset).point coordinate).eval left) =
      List.ofFn (fun coordinate =>
        ((interface.running offset).point coordinate).eval right)
    apply congrArg List.ofFn
    funext coordinate
    exact ((interface.running offset).point coordinate
      ).eval_eq_of_agree_below offset left right
        (below.runningPoint coordinate) agrees
  · funext source row coefficient
    exact ((interface.running offset).commitment source row coefficient
      ).eval_eq_of_agree_below offset left right
        (below.runningCommitment source row coefficient) agrees
  · funext source column
    exact ((interface.running offset).publicInput source column
      ).eval_eq_of_agree_below offset left right
        (below.runningPublicInput source column) agrees
  · funext source
    unfold StatementAbsorption.evalEvaluation
    congr 1
    · funext coefficient
      exact (((interface.running offset).evaluation source).eval_K coefficient
        ).eval_eq_of_agree_below offset left right
          (below.runningEval_K source coefficient) agrees
    · funext matrix coefficient
      exact (((interface.running offset).evaluation source).eval_A matrix
        coefficient).eval_eq_of_agree_below offset left right
          (below.runningEval_A source matrix coefficient) agrees

theorem evalFresh_eq_of_agree_below
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) (left right : Env)
    (below : ExternalInputsBelow interface offset)
    (agrees : ∀ index, index < offset → left index = right index) :
    evalFresh interface offset left = evalFresh interface offset right := by
  unfold evalFresh StatementAbsorption.evalFresh
  congr 1
  · funext source row coefficient
    exact ((interface.fresh offset).commitment source row coefficient
      ).eval_eq_of_agree_below offset left right
        (below.freshCommitment source row coefficient) agrees
  · funext source column
    exact ((interface.fresh offset).publicInput source column
      ).eval_eq_of_agree_below offset left right
        (below.freshPublicInput source column) agrees

theorem evalOutput_eq_of_agree_below
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) (left right : Env)
    (below : ExternalInputsBelow interface offset)
    (agrees : ∀ index, index < offset → left index = right index) :
    evalOutput interface offset left = evalOutput interface offset right := by
  unfold evalOutput
  congr 1
  · funext source coefficient
    exact ((interface.output offset).padCoordinate source coefficient
      ).eval_eq_of_agree_below offset left right
        (below.outputEval_K source coefficient) agrees
  · funext source matrix coefficient
    exact ((interface.output offset).matrixCoordinate source matrix coefficient
      ).eval_eq_of_agree_below offset left right
        (below.outputEval_A source matrix coefficient) agrees

theorem evalProof_eq_of_agree_below
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth
      (ProductionKey.degreeBound relation) publicFits)
    (offset : Nat) (left right : Env)
    (template : Proof (ProductionKey.degreeBound relation))
    (below : ExternalInputsBelow interface offset)
    (agrees : ∀ index, index < offset → left index = right index) :
    evalProof relation interface offset left template =
      evalProof relation interface offset right template := by
  have roundsEq :
      (fun roundIndex =>
        (interface.round offset roundIndex).semanticPolynomial left) =
      fun roundIndex =>
        (interface.round offset roundIndex).semanticPolynomial right := by
    funext roundIndex
    unfold RoundTranscript.Message.semanticPolynomial
    apply FixedChain.Round.semanticPolynomial_eq_of_agree_below
    · exact ⟨fun coefficient => below.roundCoefficient roundIndex coefficient,
        ⟨trivial, trivial⟩⟩
    · exact agrees
  have outputEq :=
    evalOutput_eq_of_agree_below interface offset left right below agrees
  apply proof_ext
  · exact roundsEq
  · exact outputEq
  · rfl
  · rfl

theorem accepted_of_agree_below
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Interface logicalWidth
      (ProductionKey.degreeBound relation) publicFits)
    (offset : Nat) (left right : Env)
    (template : Proof (ProductionKey.degreeBound relation))
    (below : ExternalInputsBelow interface offset)
    (agrees : ∀ index, index < offset → left index = right index)
    (accepted : NightstreamFPrime.Spec.Folding.PiCCS.Accepted
      (ProductionKey.key relation ajtai)
      (evalRunning interface offset left)
      (evalFresh interface offset left)
      (evalProof relation interface offset left template)) :
    NightstreamFPrime.Spec.Folding.PiCCS.Accepted
      (ProductionKey.key relation ajtai)
      (evalRunning interface offset right)
      (evalFresh interface offset right)
      (evalProof relation interface offset right template) := by
  rw [← evalRunning_eq_of_agree_below interface offset left right below agrees,
    ← evalFresh_eq_of_agree_below interface offset left right below agrees,
    ← evalProof_eq_of_agree_below relation interface offset left right template
      below agrees]
  exact accepted

def pointWeightedAssumptionsAt
    {variableCount : Nat}
    {interface : PointWeightedHorner.Interface variableCount}
    {offset : Nat} {env : Env}
    (assumptions : PointWeightedHorner.Assumptions interface offset env)
    (current : Env) :
    PointWeightedHorner.Assumptions interface offset current :=
  ⟨assumptions.point, assumptions.hornerExternal, assumptions.expectedBelow⟩

def ownedPointWeightedAssumptionsAt
    {variableCount : Nat}
    {interface : PointWeightedHorner.Owned.Interface variableCount}
    {offset : Nat} {env : Env}
    (assumptions : PointWeightedHorner.Owned.Assumptions
      interface offset env)
    (current : Env) :
    PointWeightedHorner.Owned.Assumptions interface offset current :=
  ⟨assumptions.point, assumptions.hornerExternal⟩

def finalIdentityAssumptionsAt
    {interface : FinalIdentity.Interface}
    {offset : Nat} {env : Env}
    (assumptions : FinalIdentity.Assumptions interface offset env)
    (current : Env) : FinalIdentity.Assumptions interface offset current :=
  ⟨assumptions.point, assumptions.gammaBelow, assumptions.eval_KBelow,
    assumptions.eval_ABelow, assumptions.ccsBelow, assumptions.normBelow,
    assumptions.terminalBelow⟩

def stateBindingAssumptionsAt
    {interface : StateBinding.Interface}
    {offset : Nat} {env : Env}
    (assumptions : StateBinding.Assumptions interface offset env)
    (current : Env) :
    StateBinding.Assumptions interface offset current :=
  ⟨assumptions.priorFixed, assumptions.outputFixed,
    assumptions.priorContext, assumptions.outputContext,
    assumptions.expectedContext⟩

/-- Child assumptions contain only syntactic range facts and do not depend
on environment values. -/
def assumptionsAt
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {interface : Interface logicalWidth degreeBound publicFits}
    {offset : Nat} {env : Env}
    (assumptions : Assumptions relation interface offset env)
    (current : Env) : Assumptions relation interface offset current := by
  rcases assumptions with
    ⟨external, statementBinding, statementAbsorption, challenge,
      roundTranscript, initialClaim, sumcheck, eval_K, eval_A, ccs, norm,
      finalIdentity, outputBinding⟩
  exact ⟨external, stateBindingAssumptionsAt statementBinding current,
    statementAbsorption, challenge,
    roundTranscript, initialClaim, sumcheck,
    ownedPointWeightedAssumptionsAt eval_K current,
    ownedPointWeightedAssumptionsAt eval_A current, ccs, norm,
    finalIdentityAssumptionsAt finalIdentity current, outputBinding⟩

theorem childOp_localLength (name : String) (child : FormalCircuit)
    (offset : Nat) :
    (childOp name child offset).localLength = childLength child offset := by
  change (child.asSubcircuit name offset).localLength =
    localLength (Circuit.ops child.main offset)
  exact FormalCircuit.asSubcircuit_localLength child name offset

/-- Append one child after transporting its contract from the named start to
the definitionally equal end of the current prefix. -/
theorem appendAt
    {initial : Env} {base : Nat}
    (before : Sequence.Prefix initial base)
    (name : String) (child : FormalCircuit) (namedStart : Nat)
    (startEq : base + localLength before.operations = namedStart)
    (childScope : ∀ expression ∈ flatConstraints
      (Circuit.ops child.main namedStart),
      expression.VarsBelow
        (namedStart + localLength (Circuit.ops child.main namedStart)))
    (assumptions : child.assumptions namedStart before.current)
    (specification : child.spec namedStart before.current) :
    ∃ after : Sequence.Prefix initial base,
      after.operations = before.operations ++ [childOp name child namedStart] ∧
      base + localLength after.operations = nextOffset child namedStart ∧
      Sequence.PreservesPrefix before after ∧
      holdsFlat after.current (Circuit.ops child.main namedStart) := by
  have assumptionsAtEnd : child.assumptions
      (base + localLength before.operations) before.current := by
    simpa only [startEq] using assumptions
  have specificationAtEnd : child.spec
      (base + localLength before.operations) before.current := by
    simpa only [startEq] using specification
  have childScopeAtEnd : ∀ expression ∈ flatConstraints
      (Circuit.ops child.main
        (base + localLength before.operations)),
      expression.VarsBelow
        (base + localLength before.operations +
          localLength (Circuit.ops child.main
            (base + localLength before.operations))) := by
    simpa only [startEq] using childScope
  rcases Sequence.append before child (childOp name child namedStart)
      (by
        rw [childOp_localLength]
        unfold childLength
        rw [startEq])
      (by
        change flatConstraints (Circuit.ops child.main namedStart) =
          flatConstraints (Circuit.ops child.main
            (base + localLength before.operations))
        rw [startEq])
      childScopeAtEnd assumptionsAtEnd specificationAtEnd with
    ⟨after, operationsEq, preserves, childRows⟩
  refine ⟨after, operationsEq, ?_, preserves, ?_⟩
  · rw [operationsEq, Sequence.localLength_append,
      Sequence.localLength_singleton, childOp_localLength]
    rw [← Nat.add_assoc, startEq]
    rfl
  · simpa only [startEq] using childRows

/-- Append a child whose canonical logical builder already produced the
satisfying assignment. This keeps the same opaque `childOp` assembly path. -/
theorem appendBuiltAt
    {initial : Env} {base : Nat}
    (before : Sequence.Prefix initial base)
    (name : String) (child : FormalCircuit) (namedStart : Nat)
    (startEq : base + localLength before.operations = namedStart)
    (childScope : ∀ expression ∈ flatConstraints
      (Circuit.ops child.main namedStart),
      expression.VarsBelow
        (namedStart + localLength (Circuit.ops child.main namedStart)))
    (after : Env)
    (childAgrees : AgreesOutside before.current after namedStart
      (localLength (Circuit.ops child.main namedStart)))
    (childRows : holdsFlat after (Circuit.ops child.main namedStart)) :
    ∃ completed : Sequence.Prefix initial base,
      completed.operations = before.operations ++ [childOp name child namedStart] ∧
      base + localLength completed.operations = nextOffset child namedStart ∧
      Sequence.PreservesPrefix before completed ∧
      holdsFlat completed.current (Circuit.ops child.main namedStart) := by
  have childAgreesAtEnd : AgreesOutside before.current after
      (base + localLength before.operations)
      (localLength (Circuit.ops child.main
        (base + localLength before.operations))) := by
    simpa only [startEq] using childAgrees
  have childRowsAtEnd : holdsFlat after
      (Circuit.ops child.main
        (base + localLength before.operations)) := by
    simpa only [startEq] using childRows
  have childScopeAtEnd : ∀ expression ∈ flatConstraints
      (Circuit.ops child.main
        (base + localLength before.operations)),
      expression.VarsBelow
        (base + localLength before.operations +
          localLength (Circuit.ops child.main
            (base + localLength before.operations))) := by
    simpa only [startEq] using childScope
  rcases Sequence.appendBuilt before child (childOp name child namedStart)
      (by
        rw [childOp_localLength]
        unfold childLength
        rw [startEq])
      (by
        change flatConstraints (Circuit.ops child.main namedStart) =
          flatConstraints (Circuit.ops child.main
            (base + localLength before.operations))
        rw [startEq])
      childScopeAtEnd after childAgreesAtEnd childRowsAtEnd with
    ⟨completed, operationsEq, preserves, rows⟩
  refine ⟨completed, operationsEq, ?_, preserves, ?_⟩
  · rw [operationsEq, Sequence.localLength_append,
      Sequence.localLength_singleton, childOp_localLength]
    rw [← Nat.add_assoc, startEq]
    rfl
  · simpa only [startEq] using rows

end NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.CompletenessSupport
