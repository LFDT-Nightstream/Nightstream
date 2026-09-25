import NightstreamFPrime.Export.Stage1.PiRLCInputCheck
import NightstreamFPrime.Export.Stage1.PiDECParity

/-!
Check actual PiDEC messages after the selected C/R execution. The parent
comes from its typed R trace. The paper verifier owns the public split,
point, stage and recomposition checks. The supplied running-claim encoding
must also match every verifier-computed public input and the shared point.
The full matrix relation and Ajtai key are the selected application values.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECInputCheck

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra

abbrev Input := PiCCSInputCheck.Input
abbrev Messages := PiCCSInputCheck.RunningInput
abbrev ParentValues := PiRLCInputCheck.ParentValues
abbrev logicalWidth :=
  PerApplicationFixedPoint.logicalWidth Poseidon2HashChainV1Package.application
abbrev publicFits :=
  PerApplicationFixedPoint.publicFits Poseidon2HashChainV1Package.application
abbrev Claim := CE.Instance (PaperAlgebra.Structure logicalWidth)
  PiCCSInputCheck.PublicInput PaperAlgebra.Point PaperAlgebra.Evaluation
  PaperAlgebra.Commitment

@[noinline] private def selectedMatrix (matrix : Fin productionProfile.ccsMatrices) :
    PaperLinearAlgebra.BooleanMatrix F Lifecycle.cubeVariables logicalWidth :=
  (PerApplicationFixedPoint.relation Poseidon2HashChainV1Package.application
    Poseidon2HashChainV1Package.fits).matrices matrix

/-- Keep matrix construction behind entry access; public PiDEC checks do not
evaluate matrix entries. The relation is still exactly the selected plan. -/
def relation : ProductionKey.LogicalRelation logicalWidth publicFits where
  matrices := selectedMatrix
  cubeFits := Poseidon2HashChainV1Package.fits.carrier

theorem relation_eq_selected : relation = PerApplicationFixedPoint.relation
    Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits := by
  have ext (left right : ProductionKey.LogicalRelation logicalWidth publicFits)
      (matrices : left.matrices = right.matrices) : left = right := by
    cases left
    cases right
    cases matrices
    rfl
  apply ext
  funext matrix
  simp only [relation, selectedMatrix]

def evaluation (values : ParentValues) : PaperAlgebra.Evaluation where
  pad := values.evalK.toRing
  matrix := fun matrix => (values.evalA.get matrix).toRing

def parent (values : ParentValues) : Claim where
  constraintSystem := Lifecycle.PiRLC.v1_1.InputBinding.relationSource relation
  commitment := values.commitment.toCommitment
  publicInput := fun coordinate : Fin 270 => values.publicInput.get coordinate
  point := values.point
  evaluations := #[evaluation values]
  stage := .combined

def attempt (values : ParentValues) (messages : Messages) :
    PiDEC.PaperVerifier.Attempt (PaperAlgebra.Structure logicalWidth)
      PiCCSInputCheck.PublicInput PaperAlgebra.Point PaperAlgebra.Evaluation
      PaperAlgebra.Commitment productionGlobalParams where
  parent := parent values
  messages := fun child => {
    commitment := (PiCCSInputCheck.runningFromInput messages).commitments child
    evaluations := #[(PiCCSInputCheck.runningFromInput messages).evaluations child] }

def children (values : ParentValues) (messages : Messages) : Fin 16 → Claim :=
  PiDEC.PaperVerifier.children
    (publicInputSplit Poseidon2HashChainV1Setup.productionAjtaiKey)
    (attempt values messages)

def parentBounded (values : ParentValues) : Bool :=
  let split := publicInputSplit Poseidon2HashChainV1Setup.productionAjtaiKey
  letI := split.parentBounded_decidable (parent values).publicInput
  decide (split.parentBounded (parent values).publicInput)

def paperAccepted (values : ParentValues) (messages : Messages) : Bool :=
  letI := piDecDecision Poseidon2HashChainV1Setup.productionAjtaiKey (attempt values messages)
  decide (PiDEC.PaperVerifier.Accepted
    (piDecAlgebra Poseidon2HashChainV1Setup.productionAjtaiKey)
    (publicInputSplit Poseidon2HashChainV1Setup.productionAjtaiKey)
    (evaluationArity Poseidon2HashChainV1Setup.productionAjtaiKey)
    (attempt values messages))

def outputMatches (values : ParentValues) (messages : Messages) : Bool :=
  decide (messages.point.toList = values.point.coordinates) &&
    (List.finRange 16).all fun child =>
      decide ((messages.publicInputs.get child).toList =
        List.ofFn (fun coordinate : Fin 270 =>
          (children values messages child).publicInput coordinate))

def accepted (values : ParentValues) (messages : Messages) : Bool :=
  if parentBounded values then paperAccepted values messages && outputMatches values messages
  else false

/-- The concrete check implies the existing paper predicate at the selected
key and exact parent. No opening or canonical-assignment premise is added. -/
theorem accepted_implies_paper (values : ParentValues) (messages : Messages)
    (checked : accepted values messages = true) :
    PiDEC.PaperVerifier.Accepted
      (piDecAlgebra Poseidon2HashChainV1Setup.productionAjtaiKey)
      (publicInputSplit Poseidon2HashChainV1Setup.productionAjtaiKey)
      (evaluationArity Poseidon2HashChainV1Setup.productionAjtaiKey)
      (attempt values messages) := by
  unfold accepted at checked
  split at checked
  · simp only [Bool.and_eq_true] at checked
    letI := piDecDecision Poseidon2HashChainV1Setup.productionAjtaiKey (attempt values messages)
    exact of_decide_eq_true checked.1
  · cases checked

theorem accepted_implies_outputMatches (values : ParentValues) (messages : Messages)
    (checked : accepted values messages = true) : outputMatches values messages = true := by
  unfold accepted at checked
  split at checked
  · simp only [Bool.and_eq_true] at checked
    exact checked.2
  · cases checked

theorem rejected_on_unbounded (values : ParentValues) (messages : Messages)
    (unbounded : ¬ PublicInput.parentBounded (parent values).publicInput) :
    accepted values messages = false := by
  cases result : accepted values messages with
  | false => rfl
  | true => exact False.elim (unbounded (accepted_implies_paper values messages result).parentBounded)

theorem child_point (values : ParentValues) (messages : Messages) (child : Fin 16) :
    (children values messages child).point = values.point := rfl

theorem child_structure (values : ParentValues) (messages : Messages) (child : Fin 16) :
    (children values messages child).constraintSystem =
      Lifecycle.PiRLC.v1_1.InputBinding.relationSource
        (PerApplicationFixedPoint.relation Poseidon2HashChainV1Package.application
          Poseidon2HashChainV1Package.fits) := by
  rw [← relation_eq_selected]
  rfl

theorem child_publicInput (values : ParentValues) (messages : Messages) (child : Fin 16) :
    (children values messages child).publicInput =
      PublicInput.splitPublicInput (parent values).publicInput child := rfl

/-- The checked execution consumes the existing knowledge reduction. Valid
openings of its actual children reconstruct an opening of its actual parent. -/
theorem accepted_reduces_knowledge (values : ParentValues) (messages : Messages)
    (assignments : Fin 16 → PaperAlgebra.Assignment
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (checked : accepted values messages = true)
    (childrenValid : ∀ child,
      CE.Holds (semantics Poseidon2HashChainV1Setup.productionAjtaiKey)
        productionGlobalParams (children values messages child) (assignments child)) :
    CE.Holds (semantics Poseidon2HashChainV1Setup.productionAjtaiKey)
      productionGlobalParams (parent values) (Radix.recomposeAssignment assignments) :=
  PiDEC.PaperVerifier.reduce_knowledge
    (semantics Poseidon2HashChainV1Setup.productionAjtaiKey) productionGlobalParams
    (piDecAlgebra Poseidon2HashChainV1Setup.productionAjtaiKey)
    (publicInputSplit Poseidon2HashChainV1Setup.productionAjtaiKey)
    (evaluationArity Poseidon2HashChainV1Setup.productionAjtaiKey)
    (attempt values messages) assignments (by decide)
    (accepted_implies_paper values messages checked) childrenValid

def publicValue (value : Fin 270 → F) : Value :=
  PiCCSParity.fieldWordsValue (List.ofFn value)

def claimValue (claim : Claim) : Value :=
  let value := claim.evaluations.getD 0 evaluationZero
  .array [PiDECParity.commitmentValue claim.commitment, publicValue claim.publicInput,
    PiDECParity.pointValue claim.point, PiDECParity.evalKValue value,
    PiDECParity.evalAValue value, .atom (if claim.stage = .combined then 1 else 0)]

def runningValue (values : ParentValues) (messages : Messages) : Value :=
  .array [PiDECParity.pointValue values.point,
    .array ((List.finRange 16).map fun child =>
      PiDECParity.commitmentValue (children values messages child).commitment),
    .array ((List.finRange 16).map fun child =>
      publicValue (children values messages child).publicInput),
    .array ((List.finRange 16).map fun child =>
      PiDECParity.evalKValue ((children values messages child).evaluations.getD 0 evaluationZero)),
    .array ((List.finRange 16).map fun child =>
      PiDECParity.evalAValue ((children values messages child).evaluations.getD 0 evaluationZero))]

def inputValue (values : ParentValues) (messages : Messages) : Value :=
  let running := PiCCSInputCheck.runningFromInput messages
  .array [claimValue (parent values),
    .array ((List.finRange 16).map fun child => PiDECParity.commitmentValue (running.commitments child)),
    .array ((List.finRange 16).map fun child => PiDECParity.evalKValue (running.evaluations child)),
    .array ((List.finRange 16).map fun child => PiDECParity.evalAValue (running.evaluations child)),
    .array ((List.finRange 16).map fun child => publicValue (running.publicInputs child)),
    PiDECParity.pointValue running.point,
    PiCCSParity.stateValue values.outgoing]

def unbounded (values : ParentValues) : ParentValues :=
  { values with publicInput := PiRLCPartialTrace.FixedArray.ofFn fun coordinate =>
      if coordinate.val = 0 then Radix.fieldOfNat Radix.combinedBound
      else values.publicInput.get coordinate }

def resultValue (values : ParentValues) (messages : Messages) : Value :=
  if !parentBounded values then
    .array [.atom 0, .atom 0, PiCCSParity.stateValue values.outgoing]
  else
    let running := PiCCSInputCheck.runningFromInput messages
    let combinedC := Commitment.recomposeCommitment running.commitments
    let combinedX := PublicInput.recomposePublicInput fun child => (children values messages child).publicInput
    let combinedY := recomposeEvaluationFamily running.evaluations
    let checks := accepted values messages
    .array [PiCCSParity.boolValue checks, PiCCSParity.boolValue (parentBounded values),
      .array ((List.finRange 16).map fun child => publicValue (children values messages child).publicInput),
      .array ((List.finRange 270).map fun coordinate =>
        PiCCSParity.boolValue (decide (centeredMagnitude ((parent values).publicInput coordinate) < Radix.combinedBound))),
      .array ((List.finRange 16).map fun child => .array ((List.finRange 270).map fun coordinate =>
        PiCCSParity.boolValue (decide (centeredMagnitude ((children values messages child).publicInput coordinate) < 2)))),
      PiDECParity.commitmentValue combinedC,
      PiCCSParity.boolValue (decide (serializeCommitment combinedC = serializeCommitment (parent values).commitment)),
      publicValue combinedX,
      PiCCSParity.boolValue (decide (List.ofFn combinedX = List.ofFn (parent values).publicInput)),
      PiDECParity.evalKValue combinedY,
      PiCCSParity.boolValue (decide (List.ofFn combinedY.pad = List.ofFn (evaluation values).pad)),
      PiDECParity.evalAValue combinedY,
      PiCCSParity.boolValue (decide (List.ofFn (fun matrix => List.ofFn (combinedY.matrix matrix)) =
        List.ofFn (fun matrix => List.ofFn ((evaluation values).matrix matrix)))),
      .array ((List.finRange 16).map fun child => claimValue (children values messages child)),
      PiCCSParity.stateValue values.outgoing,
      PiCCSParity.boolValue (!accepted (unbounded values) messages),
      if checks then .array [.atom 1, runningValue values messages] else .array [.atom 0]]

def checkValueIO (input : Input) (messages : Messages)
    (packageIdentity : VerifierContext.Digest4) : IO Value := do
  let previous ← PiRLCInputCheck.checkIO input packageIdentity
  IO.eprintln "PiCCS/PiRLC execution finished; checking PiDEC"
  match previous.parent with
  | none => return .array (previous.fields ++ [.array [], .array [.atom 0]])
  | some values => return .array (previous.fields ++
      [inputValue values messages, resultValue values messages])

end NightstreamFPrime.Export.Stage1.PiDECInputCheck
