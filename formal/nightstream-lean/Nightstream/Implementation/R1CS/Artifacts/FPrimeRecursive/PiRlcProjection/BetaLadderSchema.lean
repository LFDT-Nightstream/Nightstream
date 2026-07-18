import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcProjection.IndexedRows

/-!
Typed schema for the active shared PiRLC beta-power ladder.

Owns: physical row/column coordinates and reconstruction of the exact
`ProjectionProgram.LadderTrace.ofColumns` definition schedule.

Does not own: generated values, beta transcript derivation, row satisfaction,
the rho evaluations, projection identities, semantic soundness, or row
removal.

Emits constraints: no.

| Stage path | Mathematical obligation | Source-R1CS shape |
|---|---|---|
| `nifs.pi_rlc.verify.projection_shared.beta_ladder` | `p[0] = 1`; `p[i+1] = p[i] * beta` | two base rows plus one five-row K-mul per successor power |
-/

namespace Nightstream.Implementation.R1CS

structure PiRlcProjectionBetaLadderOwner where
  stagePath : String
  rowStart : Nat
  rowEnd : Nat
  allocatedStart : Nat
  allocatedEnd : Nat
  betaColumns : ProjectionProgram.KColumns
  powerColumns : List ProjectionProgram.KColumns
deriving DecidableEq, Repr, Inhabited

namespace PiRlcProjectionBetaLadderOwner

def rowCount (owner : PiRlcProjectionBetaLadderOwner) : Nat :=
  owner.rowEnd - owner.rowStart

def allocatedCount (owner : PiRlcProjectionBetaLadderOwner) : Nat :=
  owner.allocatedEnd - owner.allocatedStart

def ladderTrace (owner : PiRlcProjectionBetaLadderOwner) :
    ProjectionProgram.LadderTrace :=
  ProjectionProgram.LadderTrace.ofColumns
    owner.betaColumns owner.powerColumns

def rowDefinitions (owner : PiRlcProjectionBetaLadderOwner) :
    List (Nat × Program.Definition) :=
  List.zip (List.range' owner.rowStart owner.rowCount)
    owner.ladderTrace.definitions

/-- Local physical layout only. Input authority and transcript timing remain
outside the artifact predicate. -/
def Valid (owner : PiRlcProjectionBetaLadderOwner)
    (powerCount : Nat) : Prop :=
  owner.stagePath ≠ "" ∧
  1 < powerCount ∧
  owner.rowStart < owner.rowEnd ∧
  owner.allocatedStart < owner.allocatedEnd ∧
  owner.powerColumns.length = powerCount ∧
  owner.rowCount = 2 + 5 * (powerCount - 1) ∧
  owner.allocatedCount = owner.rowCount ∧
  owner.betaColumns.c0 < owner.allocatedStart ∧
  owner.betaColumns.c1 < owner.allocatedStart ∧
  (owner.ladderTrace.definitions.map Program.Definition.output) =
    List.range' owner.allocatedStart owner.allocatedCount ∧
  owner.ladderTrace.LayoutValid ∧
  owner.ladderTrace.definitions.length = owner.rowCount ∧
  owner.rowDefinitions.length = owner.rowCount

instance (owner : PiRlcProjectionBetaLadderOwner) (powerCount : Nat) :
    Decidable (owner.Valid powerCount) := by
  unfold Valid
  infer_instance

theorem Valid.layout {owner : PiRlcProjectionBetaLadderOwner}
    {powerCount : Nat} (valid : owner.Valid powerCount) :
    owner.ladderTrace.LayoutValid := by
  rcases valid with
    ⟨_, _, _, _, _, _, _, _, _, _, layout, _, _⟩
  exact layout

end PiRlcProjectionBetaLadderOwner

end Nightstream.Implementation.R1CS
