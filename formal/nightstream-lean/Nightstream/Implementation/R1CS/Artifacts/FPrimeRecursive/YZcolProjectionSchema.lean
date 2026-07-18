import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcProjection.IndexedRows

/-!
Typed schema for the two fixed-profile parent `y_zcol` evaluation leaves.

Owns: row and column coordinates for one `enforce_eval_at_beta` call and its
reconstructed `ProjectionProgram.EvalTrace`.

Does not own: generated values, parent-opening authority, transcript timing,
the shared beta ladder rows, the ten padded-lane zero/canonicalization checks,
generic indexed-row matching, semantic projection soundness, Rust conformance,
cost estimates, or permission to remove rows.

Emits constraints: no.

Authority boundary: every record field is untrusted artifact data. `Valid`
checks only local layout and ownership. A correspondence theorem must still
interpret the reconstructed equations and bind their inputs to independent
protocol semantics.

| Stage path | Mathematical obligation | Source-R1CS leaf |
|---|---|---|
| `nifs.pi_rlc.verify.identities.y_zcol.evaluations.output.limb0` | `E0 = sum_i parent_y_zcol[i].c0 * beta^i` | one 54-coefficient `EvalTrace` |
| `nifs.pi_rlc.verify.identities.y_zcol.evaluations.output.limb1` | `E1 = sum_i parent_y_zcol[i].c1 * beta^i` | one 54-coefficient `EvalTrace` |
-/

namespace Nightstream.Implementation.R1CS

/-- Artifact coordinates for one active parent-`y_zcol` limb evaluation. -/
structure YZcolOutputEvaluationOwner where
  stagePath : String
  identityIndex : Nat
  limb : Nat
  identityRowStart : Nat
  identityRowEnd : Nat
  evaluationRowStart : Nat
  evaluationRowEnd : Nat
  evaluationAllocatedStart : Nat
  evaluationAllocatedEnd : Nat
  parentCoefficientColumns : List Nat
  powerColumns : List ProjectionProgram.KColumns
  evaluationOutputColumns : ProjectionProgram.KColumns
deriving DecidableEq, Repr, Inhabited

namespace YZcolOutputEvaluationOwner

def evaluationRowCount (owner : YZcolOutputEvaluationOwner) : Nat :=
  owner.evaluationRowEnd - owner.evaluationRowStart

def evaluationAllocatedCount (owner : YZcolOutputEvaluationOwner) : Nat :=
  owner.evaluationAllocatedEnd - owner.evaluationAllocatedStart

/-- Reconstructed production evaluator. Its definitions are the actual leaf
equations; the generated artifact supplies only their column coordinates. -/
def evalTrace (owner : YZcolOutputEvaluationOwner) : ProjectionProgram.EvalTrace :=
  ProjectionProgram.EvalTrace.ofColumns owner.parentCoefficientColumns
    owner.powerColumns owner.evaluationOutputColumns

/-- Exact source-row index paired with each reconstructed R1CS definition. -/
def rowDefinitions (owner : YZcolOutputEvaluationOwner) :
    List (Nat × Program.Definition) :=
  List.zip
    (List.range' owner.evaluationRowStart owner.evaluationRowCount)
    owner.evalTrace.definitions

/-- Local artifact validity. This establishes a complete, contiguous
`2 * (n - 1) + 2` evaluator leaf inside its enclosing projection identity.
It does not establish the meaning or authority of any input column. -/
def Valid (owner : YZcolOutputEvaluationOwner) (activeLaneCount : Nat) : Prop :=
  owner.stagePath ≠ "" ∧
  0 < activeLaneCount ∧
  owner.limb < 2 ∧
  owner.identityRowStart ≤ owner.evaluationRowStart ∧
  owner.evaluationRowEnd ≤ owner.identityRowEnd ∧
  owner.evaluationRowStart < owner.evaluationRowEnd ∧
  owner.evaluationAllocatedStart < owner.evaluationAllocatedEnd ∧
  owner.parentCoefficientColumns.length = activeLaneCount ∧
  owner.powerColumns.length = activeLaneCount ∧
  owner.evaluationRowCount = 2 * (activeLaneCount - 1) + 2 ∧
  owner.evaluationAllocatedCount = owner.evaluationRowCount ∧
  owner.evaluationOutputColumns.c0 = owner.evaluationAllocatedEnd - 2 ∧
  owner.evaluationOutputColumns.c1 = owner.evaluationAllocatedEnd - 1 ∧
  (∀ column ∈ owner.parentCoefficientColumns,
    column < owner.evaluationAllocatedStart) ∧
  (∀ power ∈ owner.powerColumns,
    power.c0 < owner.evaluationAllocatedStart ∧
    power.c1 < owner.evaluationAllocatedStart) ∧
  owner.evalTrace.LayoutValid ∧
  owner.evalTrace.definitions.length = owner.evaluationRowCount ∧
  owner.rowDefinitions.length = owner.evaluationRowCount

instance (owner : YZcolOutputEvaluationOwner) (activeLaneCount : Nat) :
    Decidable (owner.Valid activeLaneCount) := by
  unfold Valid
  infer_instance

theorem Valid.layout {owner : YZcolOutputEvaluationOwner}
    {activeLaneCount : Nat} (valid : owner.Valid activeLaneCount) :
    owner.evalTrace.LayoutValid := by
  rcases valid with
    ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, layout, _, _⟩
  exact layout

end YZcolOutputEvaluationOwner

end Nightstream.Implementation.R1CS
