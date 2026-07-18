import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcProjection.IndexedRows

/-!
Typed schema for the active shared PiRLC rho-evaluation block.

Owns: physical coordinates for one `enforce_eval_at_beta` call, independent
reconstruction of its `ProjectionProgram.EvalTrace`, and exact adjacency of
the repeated evaluations.

Does not own: generated values, transcript derivation or semantic authority of
rho, beta transcript authority, source-row satisfaction, projection identity
soundness, encoded lowering, or row removal.

Emits constraints: no.

| Stage path | Mathematical obligation | Source-R1CS leaf |
|---|---|---|
| `nifs.pi_rlc.verify.projection_shared.rho_evaluations` | `rho_i(beta) = sum_j rho_i[j] beta^j` | one 54-coefficient `EvalTrace` per challenge |
-/

namespace Nightstream.Implementation.R1CS

structure PiRlcRhoEvaluationOwner where
  stagePath : String
  pairIndex : Nat
  traceIndex : Nat
  rowStart : Nat
  rowEnd : Nat
  allocatedStart : Nat
  allocatedEnd : Nat
  coefficientColumns : List Nat
  powerColumns : List ProjectionProgram.KColumns
  outputColumns : ProjectionProgram.KColumns
deriving DecidableEq, Repr, Inhabited

namespace PiRlcRhoEvaluationOwner

def rowCount (owner : PiRlcRhoEvaluationOwner) : Nat :=
  owner.rowEnd - owner.rowStart

def allocatedCount (owner : PiRlcRhoEvaluationOwner) : Nat :=
  owner.allocatedEnd - owner.allocatedStart

def evalTrace (owner : PiRlcRhoEvaluationOwner) :
    ProjectionProgram.EvalTrace :=
  ProjectionProgram.EvalTrace.ofColumns owner.coefficientColumns
    owner.powerColumns owner.outputColumns

def rowDefinitions (owner : PiRlcRhoEvaluationOwner) :
    List (Nat × Program.Definition) :=
  List.zip (List.range' owner.rowStart owner.rowCount)
    owner.evalTrace.definitions

/-- Local physical layout only. Coefficient and transcript authority remain
outside this predicate. -/
def Valid (owner : PiRlcRhoEvaluationOwner) (coefficientCount : Nat) : Prop :=
  owner.stagePath ≠ "" ∧
  0 < coefficientCount ∧
  owner.rowStart < owner.rowEnd ∧
  owner.allocatedStart < owner.allocatedEnd ∧
  owner.coefficientColumns.length = coefficientCount ∧
  owner.powerColumns.length = coefficientCount ∧
  owner.rowCount = 2 * (coefficientCount - 1) + 2 ∧
  owner.allocatedCount = owner.rowCount ∧
  owner.outputColumns.c0 = owner.allocatedEnd - 2 ∧
  owner.outputColumns.c1 = owner.allocatedEnd - 1 ∧
  (∀ column ∈ owner.coefficientColumns,
    column < owner.allocatedStart) ∧
  (∀ power ∈ owner.powerColumns,
    power.c0 < owner.allocatedStart ∧
    power.c1 < owner.allocatedStart) ∧
  owner.evalTrace.LayoutValid ∧
  owner.evalTrace.definitions.length = owner.rowCount ∧
  owner.rowDefinitions.length = owner.rowCount

instance (owner : PiRlcRhoEvaluationOwner) (coefficientCount : Nat) :
    Decidable (owner.Valid coefficientCount) := by
  unfold Valid
  infer_instance

theorem Valid.layout {owner : PiRlcRhoEvaluationOwner}
    {coefficientCount : Nat} (valid : owner.Valid coefficientCount) :
    owner.evalTrace.LayoutValid := by
  rcases valid with
    ⟨_, _, _, _, _, _, _, _, _, _, _, _, layout, _, _⟩
  exact layout

theorem Valid.coefficient_length {owner : PiRlcRhoEvaluationOwner}
    {coefficientCount : Nat} (valid : owner.Valid coefficientCount) :
    owner.coefficientColumns.length = coefficientCount := by
  exact valid.2.2.2.2.1

def Adjacent (left right : PiRlcRhoEvaluationOwner) : Prop :=
  left.pairIndex + 1 = right.pairIndex ∧
  left.traceIndex + 1 = right.traceIndex ∧
  left.rowEnd = right.rowStart ∧
  left.allocatedEnd = right.allocatedStart

instance (left right : PiRlcRhoEvaluationOwner) :
    Decidable (left.Adjacent right) := by
  unfold Adjacent
  infer_instance

def OrderedContiguous : List PiRlcRhoEvaluationOwner → Prop
  | [] | [_] => True
  | left :: right :: rest => left.Adjacent right ∧
      OrderedContiguous (right :: rest)

private def orderedContiguousDecidable :
    (owners : List PiRlcRhoEvaluationOwner) →
      Decidable (OrderedContiguous owners)
  | [] | [_] => isTrue trivial
  | left :: right :: rest => by
      letI := orderedContiguousDecidable (right :: rest)
      exact inferInstanceAs (Decidable
        (left.Adjacent right ∧ OrderedContiguous (right :: rest)))

instance (owners : List PiRlcRhoEvaluationOwner) :
    Decidable (OrderedContiguous owners) :=
  orderedContiguousDecidable owners

end PiRlcRhoEvaluationOwner

end Nightstream.Implementation.R1CS
