import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.TerminalSemantics.GroupCoverage

/-!
Exact group coverage and whole-program terminal composition for the bounded
selective fixed-point `y_zcol` projection slice.

Owns: joining exact group coverage and transported terminal proofs to honest
selected-row completeness.

Does not own: generic symbolic lockstep, direct family algebra, centered-word
packing, producer authority, security events, or permission to remove rows.

Emits constraints: no.

| Leaf | Mathematical obligation | Authority class |
|---|---|---|
| `honest.group_composition` | terminal validity composes across the exact group sequence | derived |
| `honest.complete_assignment` | the independent source boundary constructs satisfying selected rows | computed + derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.TerminalSemantics

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.SourceDecode
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.DirectSemantics
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment

private theorem terminalsHoldFrom_append
    (source : Nat → Nat) (derived : Nat → F) : ∀ left right,
    TerminalsHoldFrom source derived (left ++ right) ↔
      TerminalsHoldFrom source derived left ∧
        TerminalsHoldFrom source (runDerived source derived left) right := by
  intro left
  induction left generalizing derived with
  | nil => intro right; simp [TerminalsHoldFrom, runDerived]
  | cons head tail inductionHypothesis =>
      intro right
      simp only [List.cons_append, TerminalsHoldFrom, runDerived]
      rw [inductionHypothesis]
      constructor
      · intro holds
        exact ⟨⟨holds.1, holds.2.1⟩, holds.2.2⟩
      · intro holds
        exact ⟨holds.1.1, holds.1.2, holds.2⟩

private theorem flattenTerminalsHold
    (source : Nat → Nat) : ∀ groups : List (List DecodedRewriteStep),
    (∀ group ∈ groups, ∀ derived,
      TerminalsHoldFrom source derived group) →
    ∀ derived, TerminalsHoldFrom source derived groups.flatten := by
  intro groups
  induction groups with
  | nil => intro _ derived; trivial
  | cons group rest inductionHypothesis =>
      intro holds derived
      rw [List.flatten_cons, terminalsHoldFrom_append]
      constructor
      · exact holds group (by simp) derived
      · apply inductionHypothesis
        intro candidate member state
        exact holds candidate (by simp [member]) state

theorem rewriteTerminalsHold_of_honestSource
    {source : Nat → Nat} (honest : HonestSourceBoundary source) :
    RewriteTerminalsHold source := by
  constructor
  rw [← rewriteGroupsExact]
  apply flattenTerminalsHold source rewriteGroups
  intro group member derived
  simp only [rewriteGroups, List.mem_append] at member
  rcases member with evaluation | product
  · exact evaluationGroupTerminals honest group evaluation derived
  · exact productGroupTerminals honest group product derived

/-- Fully constructed honest completeness for the exact focused slice.
Every premise belongs to the independent canonical source boundary. -/
theorem exists_selectedRows_of_honestSource
    {source : Nat → Nat} (honest : HonestSourceBoundary source) :
    ∃ assignment,
      assignment Materialized.Checked.constantOneColumn = 1 ∧
      assignment Materialized.Checked.steadySelectorColumn = 1 ∧
      Materialized.Semantics.AssignmentCanonical assignment ∧
      Materialized.Semantics.RowsSatisfied
        Materialized.Artifact.decodedRows assignment := by
  exact HonestAssignment.exists_selectedRows_of_honestSource honest
    (rewriteTerminalsHold_of_honestSource honest)

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.TerminalSemantics
