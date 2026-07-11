import Nightstream.Assurance.FPrimeConcreteNifs
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRows

/-!
Contract: reassemble the exact generated recursive and terminal NIFS parent
rows from `FPrimeConcreteNifs.RecursiveRows` and `.TerminalRows`.

The projection compiler emits one shared prefix followed by one identity tail
per native trace, while the parent manifest stores that prefix once.  The
lemmas below recover that exact storage order from per-trace satisfaction,
then add the independently owned glue rows.  All other parent pieces come
from their named affine, residual-owner, strict-PiDEC, point-binding, and
terminal-authority row predicates.  No accepted protocol conclusion is used
as a row premise.
-/

namespace Nightstream.Assurance.FPrimeFullHistoryNifsReassembly

open Nightstream.Implementation.R1CS

set_option maxRecDepth 1048576

namespace Nifs

abbrev RecursiveRows := FPrimeConcreteNifs.RecursiveRows
abbrev TerminalRows := FPrimeConcreteNifs.TerminalRows

end Nifs

namespace Nested

abbrev recursivePiCcsPieces :=
  FPrimeFullHistoryNestedOwners.recursivePiCcsPieces
abbrev terminalPiCcsPieces :=
  FPrimeFullHistoryNestedOwners.terminalPiCcsPieces
abbrev recursivePiRlcPieces :=
  FPrimeFullHistoryNestedOwners.recursivePiRlcPieces
abbrev terminalPiRlcPieces :=
  FPrimeFullHistoryNestedOwners.terminalPiRlcPieces

end Nested

private theorem satisfies_left {left right : List Row}
    {assignment : Nat → Nat}
    (satisfies : Satisfies (left ++ right) assignment) :
    Satisfies left assignment := by
  intro row member
  exact satisfies row (List.mem_append_left right member)

private theorem satisfies_right {left right : List Row}
    {assignment : Nat → Nat}
    (satisfies : Satisfies (left ++ right) assignment) :
    Satisfies right assignment := by
  intro row member
  exact satisfies row (List.mem_append_right left member)

private theorem recursive_piCcs_residual
    {assignment : Nat → Nat}
    (rows : Nifs.RecursiveRows assignment)
    (owner : OwnerCertificate.Owner)
    (member : owner ∈
      FPrimeFullHistoryNestedOwners.recursivePiCcsResidualOwners) :
    Satisfies owner.rows assignment := by
  exact rows.residual owner (by
    simpa [FPrimeConcreteNifs.recursiveResidualOwners] using
      List.mem_append_left
        FPrimeFullHistoryNestedOwners.recursivePiRlcResidualOwners member)

private theorem recursive_piRlc_residual
    {assignment : Nat → Nat}
    (rows : Nifs.RecursiveRows assignment)
    (owner : OwnerCertificate.Owner)
    (member : owner ∈
      FPrimeFullHistoryNestedOwners.recursivePiRlcResidualOwners) :
    Satisfies owner.rows assignment := by
  exact rows.residual owner (by
    simpa [FPrimeConcreteNifs.recursiveResidualOwners] using
      List.mem_append_right
        FPrimeFullHistoryNestedOwners.recursivePiCcsResidualOwners member)

private theorem terminal_piCcs_residual
    {assignment : Nat → Nat}
    (rows : Nifs.TerminalRows assignment)
    (owner : OwnerCertificate.Owner)
    (member : owner ∈
      FPrimeFullHistoryNestedOwners.terminalPiCcsResidualOwners) :
    Satisfies owner.rows assignment := by
  exact rows.residual owner (by
    simpa [FPrimeConcreteNifs.terminalResidualOwners] using
      List.mem_append_left
        FPrimeFullHistoryNestedOwners.terminalPiRlcResidualOwners member)

private theorem terminal_piRlc_residual
    {assignment : Nat → Nat}
    (rows : Nifs.TerminalRows assignment)
    (owner : OwnerCertificate.Owner)
    (member : owner ∈
      FPrimeFullHistoryNestedOwners.terminalPiRlcResidualOwners) :
    Satisfies owner.rows assignment := by
  exact rows.residual owner (by
    simpa [FPrimeConcreteNifs.terminalResidualOwners] using
      List.mem_append_right
        FPrimeFullHistoryNestedOwners.terminalPiCcsResidualOwners member)

/-- Reassemble the exact recursive PiCCS parent owner, including all nine
compact residual owners in their production emission order. -/
theorem recursivePiCcs_satisfies
    {assignment : Nat → Nat}
    (rows : Nifs.RecursiveRows assignment) :
    Satisfies FPrimeFullHistoryNestedOwners.recursivePiCcsRows assignment := by
  apply (FPrimeFullHistoryNestedOwners.recursivePiCcs_satisfies_iff
    assignment).mpr
  intro piece member
  simp only [FPrimeFullHistoryNestedOwners.recursivePiCcsPieces,
    List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
    rfl | rfl | rfl | rfl
  · exact rows.affine.piCcsAllocation
  · exact rows.affine.piCcsAuthority
  · simpa [FPrimeFullHistoryRecursivePiCcsFreshDigests.rows] using
      recursive_piCcs_residual rows
        FPrimeFullHistoryRecursivePiCcsFreshDigests.owner (by
          simp [FPrimeFullHistoryNestedOwners.recursivePiCcsResidualOwners])
  · simpa [FPrimeFullHistoryRecursivePiCcsRunningAuthority.rows] using
      recursive_piCcs_residual rows
        FPrimeFullHistoryRecursivePiCcsRunningAuthority.owner (by
          simp [FPrimeFullHistoryNestedOwners.recursivePiCcsResidualOwners])
  · simpa [FPrimeFullHistoryRecursivePiCcsTranscript.rows] using
      recursive_piCcs_residual rows
        FPrimeFullHistoryRecursivePiCcsTranscript.owner (by
          simp [FPrimeFullHistoryNestedOwners.recursivePiCcsResidualOwners])
  · simpa [FPrimeFullHistoryRecursivePiCcsFeInitial.rows] using
      recursive_piCcs_residual rows
        FPrimeFullHistoryRecursivePiCcsFeInitial.owner (by
          simp [FPrimeFullHistoryNestedOwners.recursivePiCcsResidualOwners])
  · simpa [FPrimeFullHistoryRecursivePiCcsFeSumcheck.rows] using
      recursive_piCcs_residual rows
        FPrimeFullHistoryRecursivePiCcsFeSumcheck.owner (by
          simp [FPrimeFullHistoryNestedOwners.recursivePiCcsResidualOwners])
  · simpa [FPrimeFullHistoryRecursivePiCcsNcSumcheck.rows] using
      recursive_piCcs_residual rows
        FPrimeFullHistoryRecursivePiCcsNcSumcheck.owner (by
          simp [FPrimeFullHistoryNestedOwners.recursivePiCcsResidualOwners])
  · exact rows.affine.piCcsOutputBinding
  · simpa [FPrimeFullHistoryRecursivePiCcsFeTerminal.rows] using
      recursive_piCcs_residual rows
        FPrimeFullHistoryRecursivePiCcsFeTerminal.owner (by
          simp [FPrimeFullHistoryNestedOwners.recursivePiCcsResidualOwners])
  · simpa [FPrimeFullHistoryRecursivePiCcsNcTerminal.rows] using
      recursive_piCcs_residual rows
        FPrimeFullHistoryRecursivePiCcsNcTerminal.owner (by
          simp [FPrimeFullHistoryNestedOwners.recursivePiCcsResidualOwners])
  · simpa [FPrimeFullHistoryRecursivePiCcsCatchup.rows] using
      recursive_piCcs_residual rows
        FPrimeFullHistoryRecursivePiCcsCatchup.owner (by
          simp [FPrimeFullHistoryNestedOwners.recursivePiCcsResidualOwners])

/-- Reassemble the exact terminal PiCCS parent owner.  Its authority block is
the strict terminal-CE parent followed by the generated affine authority
tail, rather than a caller-supplied digest assertion. -/
theorem terminalPiCcs_satisfies
    {assignment : Nat → Nat}
    (rows : Nifs.TerminalRows assignment) :
    Satisfies FPrimeFullHistoryNestedOwners.terminalPiCcsRows assignment := by
  apply (FPrimeFullHistoryNestedOwners.terminalPiCcs_satisfies_iff
    assignment).mpr
  intro piece member
  simp only [FPrimeFullHistoryNestedOwners.terminalPiCcsPieces,
    List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
    rfl | rfl | rfl | rfl
  · exact rows.affine.piCcsAllocation
  · apply (satisfies_flatten_iff
      [FPrimeFullHistoryPiDec.terminalCeRows,
        FPrimeFullHistoryPiCcsTerminalAuthorityTail.rows] assignment).mpr
    intro authorityPiece authorityMember
    simp only [List.mem_cons, List.not_mem_nil, or_false] at authorityMember
    rcases authorityMember with rfl | rfl
    · exact rows.authority.piDec
    · exact rows.authority.tail
  · simpa [FPrimeFullHistoryTerminalPiCcsFreshDigests.rows] using
      terminal_piCcs_residual rows
        FPrimeFullHistoryTerminalPiCcsFreshDigests.owner (by
          simp [FPrimeFullHistoryNestedOwners.terminalPiCcsResidualOwners])
  · simpa [FPrimeFullHistoryTerminalPiCcsRunningAuthority.rows] using
      terminal_piCcs_residual rows
        FPrimeFullHistoryTerminalPiCcsRunningAuthority.owner (by
          simp [FPrimeFullHistoryNestedOwners.terminalPiCcsResidualOwners])
  · simpa [FPrimeFullHistoryTerminalPiCcsTranscript.rows] using
      terminal_piCcs_residual rows
        FPrimeFullHistoryTerminalPiCcsTranscript.owner (by
          simp [FPrimeFullHistoryNestedOwners.terminalPiCcsResidualOwners])
  · simpa [FPrimeFullHistoryTerminalPiCcsFeInitial.rows] using
      terminal_piCcs_residual rows
        FPrimeFullHistoryTerminalPiCcsFeInitial.owner (by
          simp [FPrimeFullHistoryNestedOwners.terminalPiCcsResidualOwners])
  · simpa [FPrimeFullHistoryTerminalPiCcsFeSumcheck.rows] using
      terminal_piCcs_residual rows
        FPrimeFullHistoryTerminalPiCcsFeSumcheck.owner (by
          simp [FPrimeFullHistoryNestedOwners.terminalPiCcsResidualOwners])
  · simpa [FPrimeFullHistoryTerminalPiCcsNcSumcheck.rows] using
      terminal_piCcs_residual rows
        FPrimeFullHistoryTerminalPiCcsNcSumcheck.owner (by
          simp [FPrimeFullHistoryNestedOwners.terminalPiCcsResidualOwners])
  · exact rows.affine.piCcsOutputBinding
  · simpa [FPrimeFullHistoryTerminalPiCcsFeTerminal.rows] using
      terminal_piCcs_residual rows
        FPrimeFullHistoryTerminalPiCcsFeTerminal.owner (by
          simp [FPrimeFullHistoryNestedOwners.terminalPiCcsResidualOwners])
  · simpa [FPrimeFullHistoryTerminalPiCcsNcTerminal.rows] using
      terminal_piCcs_residual rows
        FPrimeFullHistoryTerminalPiCcsNcTerminal.owner (by
          simp [FPrimeFullHistoryNestedOwners.terminalPiCcsResidualOwners])
  · simpa [FPrimeFullHistoryTerminalPiCcsCatchup.rows] using
      terminal_piCcs_residual rows
        FPrimeFullHistoryTerminalPiCcsCatchup.owner (by
          simp [FPrimeFullHistoryNestedOwners.terminalPiCcsResidualOwners])

private theorem recursive_projection_shared
    {assignment : Nat → Nat}
    (rows : Nifs.RecursiveRows assignment) :
    Satisfies
      FPrimeFullHistoryNestedOwners.recursiveProjectionSharedRows assignment := by
  have nonempty : FPrimeFullHistoryProjection.recursiveTraces ≠ [] := by
    native_decide
  obtain ⟨trace, traceMember⟩ :=
    List.exists_mem_of_ne_nil _ nonempty
  have traceRows := rows.projection trace traceMember
  rw [FPrimeFullHistoryNestedOwners.recursiveTraceRows_partition trace
    traceMember] at traceRows
  exact satisfies_left traceRows

private theorem terminal_projection_shared
    {assignment : Nat → Nat}
    (rows : Nifs.TerminalRows assignment) :
    Satisfies
      FPrimeFullHistoryNestedOwners.terminalProjectionSharedRows assignment := by
  have nonempty : FPrimeFullHistoryProjection.terminalTraces ≠ [] := by
    native_decide
  obtain ⟨trace, traceMember⟩ :=
    List.exists_mem_of_ne_nil _ nonempty
  have traceRows := rows.projection trace traceMember
  rw [FPrimeFullHistoryNestedOwners.terminalTraceRows_partition trace
    traceMember] at traceRows
  exact satisfies_left traceRows

private theorem recursive_projection_identities
    {assignment : Nat → Nat}
    (rows : Nifs.RecursiveRows assignment) :
    Satisfies
      FPrimeFullHistoryNestedOwners.recursiveProjectionIdentityRows
      assignment := by
  apply (satisfies_flatten_iff
    FPrimeFullHistoryNestedOwners.recursiveProjectionIdentityPieces
    assignment).mpr
  intro piece member
  rcases List.mem_append.mp member with tracePiece | gluePiece
  · rcases List.mem_map.mp tracePiece with ⟨trace, traceMember, rfl⟩
    have traceRows := rows.projection trace traceMember
    rw [FPrimeFullHistoryNestedOwners.recursiveTraceRows_partition trace
      traceMember] at traceRows
    exact satisfies_right traceRows
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at gluePiece
    subst piece
    exact rows.projectionGlue

private theorem terminal_projection_identities
    {assignment : Nat → Nat}
    (rows : Nifs.TerminalRows assignment) :
    Satisfies
      FPrimeFullHistoryNestedOwners.terminalProjectionIdentityRows
      assignment := by
  apply (satisfies_flatten_iff
    FPrimeFullHistoryNestedOwners.terminalProjectionIdentityPieces
    assignment).mpr
  intro piece member
  rcases List.mem_append.mp member with tracePiece | gluePiece
  · rcases List.mem_map.mp tracePiece with ⟨trace, traceMember, rfl⟩
    have traceRows := rows.projection trace traceMember
    rw [FPrimeFullHistoryNestedOwners.terminalTraceRows_partition trace
      traceMember] at traceRows
    exact satisfies_right traceRows
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at gluePiece
    subst piece
    exact rows.projectionGlue

/-- Reassemble the recursive PiRLC parent in its exact compact layout: residual
transcript/rho rows, affine shape and folds, residual projection binding, one
shared projection prefix, all 31 identity tails, and the glue tail. -/
theorem recursivePiRlc_satisfies
    {assignment : Nat → Nat}
    (rows : Nifs.RecursiveRows assignment) :
    Satisfies FPrimeFullHistoryNestedOwners.recursivePiRlcRows assignment := by
  apply (FPrimeFullHistoryNestedOwners.recursivePiRlc_satisfies_iff
    assignment).mpr
  intro piece member
  simp only [FPrimeFullHistoryNestedOwners.recursivePiRlcPieces,
    List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl | rfl
  · simpa [FPrimeFullHistoryRecursivePiRlcTranscriptRhos.rows] using
      recursive_piRlc_residual rows
        FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner (by
          simp [FPrimeFullHistoryNestedOwners.recursivePiRlcResidualOwners])
  · exact rows.affine.piRlcShape
  · exact rows.affine.piRlcLinearFolds
  · simpa [FPrimeFullHistoryRecursivePiRlcProjectionBinding.rows] using
      recursive_piRlc_residual rows
        FPrimeFullHistoryRecursivePiRlcProjectionBinding.owner (by
          simp [FPrimeFullHistoryNestedOwners.recursivePiRlcResidualOwners])
  · exact recursive_projection_shared rows
  · exact recursive_projection_identities rows

/-- Terminal-fold counterpart of `recursivePiRlc_satisfies`. -/
theorem terminalPiRlc_satisfies
    {assignment : Nat → Nat}
    (rows : Nifs.TerminalRows assignment) :
    Satisfies FPrimeFullHistoryNestedOwners.terminalPiRlcRows assignment := by
  apply (FPrimeFullHistoryNestedOwners.terminalPiRlc_satisfies_iff
    assignment).mpr
  intro piece member
  simp only [FPrimeFullHistoryNestedOwners.terminalPiRlcPieces,
    List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl | rfl
  · simpa [FPrimeFullHistoryTerminalPiRlcTranscriptRhos.rows] using
      terminal_piRlc_residual rows
        FPrimeFullHistoryTerminalPiRlcTranscriptRhos.owner (by
          simp [FPrimeFullHistoryNestedOwners.terminalPiRlcResidualOwners])
  · exact rows.affine.piRlcShape
  · exact rows.affine.piRlcLinearFolds
  · simpa [FPrimeFullHistoryTerminalPiRlcProjectionBinding.rows] using
      terminal_piRlc_residual rows
        FPrimeFullHistoryTerminalPiRlcProjectionBinding.owner (by
          simp [FPrimeFullHistoryNestedOwners.terminalPiRlcResidualOwners])
  · exact terminal_projection_shared rows
  · exact terminal_projection_identities rows

/-- Exact recursive NIFS parent reassembly. -/
theorem recursiveNifs_satisfies
    {assignment : Nat → Nat}
    (rows : Nifs.RecursiveRows assignment) :
    Satisfies FPrimeFullHistoryRows.recursiveNifsRows assignment := by
  apply (FPrimeFullHistoryRows.recursiveNifs_satisfies_iff assignment).mpr
  intro piece member
  simp only [FPrimeFullHistoryRows.recursiveNifsPieces,
    List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl
  · exact recursivePiCcs_satisfies rows
  · exact recursivePiRlc_satisfies rows
  · exact rows.piDec
  · exact rows.pointBinding

/-- Exact terminal NIFS parent reassembly, including its outer transcript. -/
theorem terminalNifs_satisfies
    {assignment : Nat → Nat}
    (rows : Nifs.TerminalRows assignment) :
    Satisfies FPrimeFullHistoryRows.terminalNifsRows assignment := by
  apply (FPrimeFullHistoryRows.terminalNifs_satisfies_iff assignment).mpr
  intro piece member
  simp only [FPrimeFullHistoryRows.terminalNifsPieces,
    List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl
  · exact rows.transcript
  · exact terminalPiCcs_satisfies rows
  · exact terminalPiRlc_satisfies rows
  · exact rows.piDec
  · exact rows.pointBinding

end Nightstream.Assurance.FPrimeFullHistoryNifsReassembly
