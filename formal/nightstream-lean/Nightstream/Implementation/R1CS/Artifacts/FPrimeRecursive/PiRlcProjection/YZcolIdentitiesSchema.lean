import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcProjection.RhoEvaluationsSchema
import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.YZcolProjectionSchema

/-!
Typed schema for the two complete active PiRLC `y_zcol` identities.

Owns: identity-local physical coordinates, reconstruction of input evaluators,
rho/input products, quotient evaluation, quotient/Phi product, final assertion
rows, and composition with separately owned ladder/rho/output traces.

Does not own: generated values, shared beta/rho rows, parent-output evaluator
rows, transcript or semantic column authority, source-row satisfaction,
encoded lowering, bad-root probability, padding, or row removal.

Emits constraints: no.

| Branch | Mathematical obligation | Physical owner |
|---|---|---|
| `pairs[i].inputEvaluation` | evaluate input `c_i(beta)` | 108 source rows |
| `pairs[i].product` | multiply `rho_i(beta) * c_i(beta)` | 5 source rows |
| `outputEvaluation` | evaluate parent output | separate existing owner |
| `quotientEvaluation` | evaluate `q(beta)` | 106 source rows |
| `quotientPhiProduct` | multiply `q(beta) * Phi81(beta)` | 5 source rows |
| `checks` | equate the two extension limbs | 2 source rows |
-/

namespace Nightstream.Implementation.R1CS

structure PiRlcYZcolIdentityPairOwner where
  pairIndex : Nat
  inputEvaluationTraceIndex : Nat
  productTraceIndex : Nat
  inputEvaluationRowStart : Nat
  inputEvaluationRowEnd : Nat
  inputEvaluationAllocatedStart : Nat
  inputEvaluationAllocatedEnd : Nat
  inputColumns : List Nat
  inputEvaluationOutput : ProjectionProgram.KColumns
  productRowStart : Nat
  productRowEnd : Nat
  productAllocatedStart : Nat
  productAllocatedEnd : Nat
  productOutput : ProjectionProgram.KColumns
deriving DecidableEq, Repr, Inhabited

namespace PiRlcYZcolIdentityPairOwner

def inputEvaluationRowCount
    (owner : PiRlcYZcolIdentityPairOwner) : Nat :=
  owner.inputEvaluationRowEnd - owner.inputEvaluationRowStart

def inputEvaluationAllocatedCount
    (owner : PiRlcYZcolIdentityPairOwner) : Nat :=
  owner.inputEvaluationAllocatedEnd - owner.inputEvaluationAllocatedStart

def productRowCount (owner : PiRlcYZcolIdentityPairOwner) : Nat :=
  owner.productRowEnd - owner.productRowStart

def productAllocatedCount (owner : PiRlcYZcolIdentityPairOwner) : Nat :=
  owner.productAllocatedEnd - owner.productAllocatedStart

def inputEvaluation (owner : PiRlcYZcolIdentityPairOwner)
    (powers : List ProjectionProgram.KColumns) : ProjectionProgram.EvalTrace :=
  ProjectionProgram.EvalTrace.ofColumns owner.inputColumns powers
    owner.inputEvaluationOutput

def product (owner : PiRlcYZcolIdentityPairOwner)
    (rhoOutput : ProjectionProgram.KColumns) : ProjectionProgram.KMulTrace :=
  ProjectionProgram.KMulTrace.ofColumns rhoOutput
    owner.inputEvaluationOutput owner.productOutput

def inputRowDefinitions (owner : PiRlcYZcolIdentityPairOwner)
    (powers : List ProjectionProgram.KColumns) :
    List (Nat × Program.Definition) :=
  List.zip
    (List.range' owner.inputEvaluationRowStart
      owner.inputEvaluationRowCount)
    (owner.inputEvaluation powers).definitions

def productRowDefinitions (owner : PiRlcYZcolIdentityPairOwner)
    (rhoOutput : ProjectionProgram.KColumns) :
    List (Nat × Program.Definition) :=
  List.zip (List.range' owner.productRowStart owner.productRowCount)
    (owner.product rhoOutput).definitions

def Valid (owner : PiRlcYZcolIdentityPairOwner)
    (coefficientCount : Nat) : Prop :=
  0 < coefficientCount ∧
  owner.inputEvaluationRowStart < owner.inputEvaluationRowEnd ∧
  owner.inputEvaluationAllocatedStart < owner.inputEvaluationAllocatedEnd ∧
  owner.inputColumns.length = coefficientCount ∧
  owner.inputEvaluationRowCount = 2 * (coefficientCount - 1) + 2 ∧
  owner.inputEvaluationAllocatedCount = owner.inputEvaluationRowCount ∧
  owner.inputEvaluationOutput.c0 = owner.inputEvaluationAllocatedEnd - 2 ∧
  owner.inputEvaluationOutput.c1 = owner.inputEvaluationAllocatedEnd - 1 ∧
  (∀ column ∈ owner.inputColumns,
    column < owner.inputEvaluationAllocatedStart) ∧
  owner.productRowStart = owner.inputEvaluationRowEnd ∧
  owner.productAllocatedStart = owner.inputEvaluationAllocatedEnd ∧
  owner.productRowCount = 5 ∧
  owner.productAllocatedCount = 5 ∧
  owner.productOutput.c0 = owner.productAllocatedEnd - 2 ∧
  owner.productOutput.c1 = owner.productAllocatedEnd - 1 ∧
  (owner.inputEvaluation
    (List.replicate coefficientCount default)).LayoutValid

instance (owner : PiRlcYZcolIdentityPairOwner) (coefficientCount : Nat) :
    Decidable (owner.Valid coefficientCount) := by
  unfold Valid
  infer_instance

def Adjacent (left right : PiRlcYZcolIdentityPairOwner) : Prop :=
  left.pairIndex + 1 = right.pairIndex ∧
  left.inputEvaluationTraceIndex + 1 = right.inputEvaluationTraceIndex ∧
  left.productTraceIndex + 1 = right.productTraceIndex ∧
  left.productRowEnd = right.inputEvaluationRowStart ∧
  left.productAllocatedEnd = right.inputEvaluationAllocatedStart

instance (left right : PiRlcYZcolIdentityPairOwner) :
    Decidable (left.Adjacent right) := by
  unfold Adjacent
  infer_instance

def OrderedContiguous : List PiRlcYZcolIdentityPairOwner → Prop
  | [] | [_] => True
  | left :: right :: rest =>
      left.Adjacent right ∧ OrderedContiguous (right :: rest)

private def orderedContiguousDecidable :
    (owners : List PiRlcYZcolIdentityPairOwner) →
      Decidable (OrderedContiguous owners)
  | [] | [_] => isTrue trivial
  | left :: right :: rest => by
      letI := orderedContiguousDecidable (right :: rest)
      exact inferInstanceAs (Decidable
        (left.Adjacent right ∧ OrderedContiguous (right :: rest)))

instance (owners : List PiRlcYZcolIdentityPairOwner) :
    Decidable (OrderedContiguous owners) :=
  orderedContiguousDecidable owners

end PiRlcYZcolIdentityPairOwner

structure PiRlcYZcolIdentityOwner where
  identityIndex : Nat
  limb : Nat
  identityRowStart : Nat
  identityRowEnd : Nat
  identityAllocatedStart : Nat
  identityAllocatedEnd : Nat
  inputStagePath : String
  productStagePath : String
  outputStagePath : String
  quotientStagePath : String
  quotientPhiStagePath : String
  finalChecksStagePath : String
  pairs : List PiRlcYZcolIdentityPairOwner
  outputColumns : List Nat
  outputEvaluationTraceIndex : Nat
  outputEvaluationRowStart : Nat
  outputEvaluationRowEnd : Nat
  outputEvaluationAllocatedStart : Nat
  outputEvaluationAllocatedEnd : Nat
  outputEvaluationOutput : ProjectionProgram.KColumns
  quotientColumns : List Nat
  quotientEvaluationTraceIndex : Nat
  quotientEvaluationRowStart : Nat
  quotientEvaluationRowEnd : Nat
  quotientEvaluationAllocatedStart : Nat
  quotientEvaluationAllocatedEnd : Nat
  quotientEvaluationOutput : ProjectionProgram.KColumns
  quotientPhiProductTraceIndex : Nat
  quotientPhiRowStart : Nat
  quotientPhiRowEnd : Nat
  quotientPhiAllocatedStart : Nat
  quotientPhiAllocatedEnd : Nat
  quotientPhiOutput : ProjectionProgram.KColumns
  finalCheckRowStart : Nat
  finalCheckRowEnd : Nat
deriving DecidableEq, Repr, Inhabited

namespace PiRlcYZcolIdentityOwner

def identityRowCount (owner : PiRlcYZcolIdentityOwner) : Nat :=
  owner.identityRowEnd - owner.identityRowStart

def identityAllocatedCount (owner : PiRlcYZcolIdentityOwner) : Nat :=
  owner.identityAllocatedEnd - owner.identityAllocatedStart

def quotientEvaluationRowCount (owner : PiRlcYZcolIdentityOwner) : Nat :=
  owner.quotientEvaluationRowEnd - owner.quotientEvaluationRowStart

def quotientEvaluationAllocatedCount
    (owner : PiRlcYZcolIdentityOwner) : Nat :=
  owner.quotientEvaluationAllocatedEnd -
    owner.quotientEvaluationAllocatedStart

def quotientPhiRowCount (owner : PiRlcYZcolIdentityOwner) : Nat :=
  owner.quotientPhiRowEnd - owner.quotientPhiRowStart

def quotientPhiAllocatedCount (owner : PiRlcYZcolIdentityOwner) : Nat :=
  owner.quotientPhiAllocatedEnd - owner.quotientPhiAllocatedStart

def quotientEvaluation (owner : PiRlcYZcolIdentityOwner)
    (powers : List ProjectionProgram.KColumns) : ProjectionProgram.EvalTrace :=
  ProjectionProgram.EvalTrace.ofColumns owner.quotientColumns powers
    owner.quotientEvaluationOutput

def quotientPhiProduct (owner : PiRlcYZcolIdentityOwner)
    (powers : List ProjectionProgram.KColumns) : ProjectionProgram.KMulTrace :=
  ProjectionProgram.KMulTrace.quotientPhi owner.quotientEvaluationOutput
    powers owner.quotientPhiOutput

def pairTrace (owner : PiRlcYZcolIdentityOwner)
    (powers : List ProjectionProgram.KColumns)
    (rhoOwners : List PiRlcRhoEvaluationOwner)
    (index : Nat) : ProjectionProgram.PairTrace :=
  let pair := owner.pairs.getD index default
  let rho := rhoOwners.getD index default
  ProjectionProgram.PairTrace.ofColumns powers rho.coefficientColumns
    pair.inputColumns rho.outputColumns pair.inputEvaluationOutput
    pair.productOutput

def pairTraces (owner : PiRlcYZcolIdentityOwner)
    (powers : List ProjectionProgram.KColumns)
    (rhoOwners : List PiRlcRhoEvaluationOwner) :
    List ProjectionProgram.PairTrace :=
  (List.range owner.pairs.length).map
    (owner.pairTrace powers rhoOwners)

def trace (owner : PiRlcYZcolIdentityOwner)
    (ladder : ProjectionProgram.LadderTrace)
    (rhoOwners : List PiRlcRhoEvaluationOwner)
    (outputOwner : YZcolOutputEvaluationOwner) :
    ProjectionProgram.ProjectionTrace where
  ladder := ladder
  pairs := owner.pairTraces ladder.powers rhoOwners
  outputColumns := owner.outputColumns
  quotientColumns := owner.quotientColumns
  outputEvaluation := outputOwner.evalTrace
  quotientEvaluation := owner.quotientEvaluation ladder.powers
  quotientPhiProduct := owner.quotientPhiProduct ladder.powers
  maxDegree := 106

def inputRowDefinitions (owner : PiRlcYZcolIdentityOwner)
    (powers : List ProjectionProgram.KColumns) :
    List (Nat × Program.Definition) :=
  owner.pairs.flatMap fun pair => pair.inputRowDefinitions powers

def productRowDefinitions (owner : PiRlcYZcolIdentityOwner)
    (rhoOwners : List PiRlcRhoEvaluationOwner) :
    List (Nat × Program.Definition) :=
  (List.zip owner.pairs rhoOwners).flatMap fun entry =>
    entry.1.productRowDefinitions entry.2.outputColumns

def quotientRowDefinitions (owner : PiRlcYZcolIdentityOwner)
    (powers : List ProjectionProgram.KColumns) :
    List (Nat × Program.Definition) :=
  List.zip
    (List.range' owner.quotientEvaluationRowStart
      owner.quotientEvaluationRowCount)
    (owner.quotientEvaluation powers).definitions

def quotientPhiRowDefinitions (owner : PiRlcYZcolIdentityOwner)
    (powers : List ProjectionProgram.KColumns) :
    List (Nat × Program.Definition) :=
  List.zip
    (List.range' owner.quotientPhiRowStart owner.quotientPhiRowCount)
    (owner.quotientPhiProduct powers).definitions

def tailDefinitionRowDefinitions (owner : PiRlcYZcolIdentityOwner)
    (powers : List ProjectionProgram.KColumns)
    (rhoOwners : List PiRlcRhoEvaluationOwner) :
    List (Nat × Program.Definition) :=
  owner.productRowDefinitions rhoOwners ++
    owner.quotientRowDefinitions powers ++
    owner.quotientPhiRowDefinitions powers

def checkRows (owner : PiRlcYZcolIdentityOwner)
    (trace : ProjectionProgram.ProjectionTrace) : List (Nat × Row) :=
  List.zip
    (List.range' owner.finalCheckRowStart
      (owner.finalCheckRowEnd - owner.finalCheckRowStart))
    trace.checks

/-- Local physical schedule only. Shared-column and output-owner linkage are
checked by the stable artifact facade. -/
def Valid (owner : PiRlcYZcolIdentityOwner)
    (pairCount coefficientCount : Nat) : Prop :=
  owner.limb < 2 ∧
  owner.identityRowStart < owner.identityRowEnd ∧
  owner.identityAllocatedStart < owner.identityAllocatedEnd ∧
  owner.inputStagePath ≠ "" ∧ owner.productStagePath ≠ "" ∧
  owner.outputStagePath ≠ "" ∧ owner.quotientStagePath ≠ "" ∧
  owner.quotientPhiStagePath ≠ "" ∧
  owner.finalChecksStagePath ≠ "" ∧
  owner.pairs.length = pairCount ∧
  owner.pairs.map (fun pair => pair.pairIndex) = List.range pairCount ∧
  (∀ pair ∈ owner.pairs, pair.Valid coefficientCount) ∧
  PiRlcYZcolIdentityPairOwner.OrderedContiguous owner.pairs ∧
  (owner.pairs.getD 0 default).inputEvaluationRowStart =
    owner.identityRowStart ∧
  (owner.pairs.getD 0 default).inputEvaluationAllocatedStart =
    owner.identityAllocatedStart ∧
  (owner.pairs.getD (pairCount - 1) default).productRowEnd =
    owner.outputEvaluationRowStart ∧
  (owner.pairs.getD (pairCount - 1) default).productAllocatedEnd =
    owner.outputEvaluationAllocatedStart ∧
  owner.outputEvaluationRowEnd = owner.quotientEvaluationRowStart ∧
  owner.outputEvaluationAllocatedEnd =
    owner.quotientEvaluationAllocatedStart ∧
  owner.outputColumns.length = coefficientCount ∧
  owner.outputEvaluationRowEnd - owner.outputEvaluationRowStart =
    2 * (coefficientCount - 1) + 2 ∧
  owner.outputEvaluationAllocatedEnd -
      owner.outputEvaluationAllocatedStart =
    owner.outputEvaluationRowEnd - owner.outputEvaluationRowStart ∧
  owner.outputEvaluationOutput.c0 = owner.outputEvaluationAllocatedEnd - 2 ∧
  owner.outputEvaluationOutput.c1 = owner.outputEvaluationAllocatedEnd - 1 ∧
  owner.quotientColumns.length = coefficientCount - 1 ∧
  owner.quotientEvaluationRowCount =
    2 * (owner.quotientColumns.length - 1) + 2 ∧
  owner.quotientEvaluationAllocatedCount = owner.quotientEvaluationRowCount ∧
  owner.quotientEvaluationOutput.c0 =
    owner.quotientEvaluationAllocatedEnd - 2 ∧
  owner.quotientEvaluationOutput.c1 =
    owner.quotientEvaluationAllocatedEnd - 1 ∧
  owner.quotientPhiRowStart = owner.quotientEvaluationRowEnd ∧
  owner.quotientPhiAllocatedStart = owner.quotientEvaluationAllocatedEnd ∧
  owner.quotientPhiRowCount = 5 ∧ owner.quotientPhiAllocatedCount = 5 ∧
  owner.quotientPhiOutput.c0 = owner.quotientPhiAllocatedEnd - 2 ∧
  owner.quotientPhiOutput.c1 = owner.quotientPhiAllocatedEnd - 1 ∧
  owner.finalCheckRowStart = owner.quotientPhiRowEnd ∧
  owner.finalCheckRowEnd = owner.finalCheckRowStart + 2 ∧
  owner.finalCheckRowEnd = owner.identityRowEnd ∧
  owner.quotientPhiAllocatedEnd = owner.identityAllocatedEnd ∧
  owner.identityRowCount = 1916 ∧
  owner.identityAllocatedCount = 1914

instance (owner : PiRlcYZcolIdentityOwner)
    (pairCount coefficientCount : Nat) :
    Decidable (owner.Valid pairCount coefficientCount) := by
  unfold Valid
  infer_instance

end PiRlcYZcolIdentityOwner

end Nightstream.Implementation.R1CS
