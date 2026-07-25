import Nightstream.Implementation.Lowering.Goldilocks.Compiler

/-!
Contract: exact row-satisfaction projection for receipt-owned programs.

Owns: equivalence between satisfaction of the flattened emitted row list and
satisfaction of every member receipt's complete row list.

Does not own: instruction semantics, typed-program refinement, assignment
construction, Rust emission, or generated artifacts.

No row can enter or leave this theorem outside the nonoptional receipt list.
-/

namespace Nightstream.Implementation.Lowering.Goldilocks

open Nightstream.Implementation.Lowering.Typed

universe u

theorem satisfies_flattened_receipts_iff
    (receipts : List InstructionReceipt)
    (assignment : ColumnId -> Field) :
    Satisfies (receipts.flatMap fun receipt => receipt.rows) assignment ↔
      ∀ receipt, receipt ∈ receipts ->
        Satisfies receipt.rows assignment := by
  induction receipts with
  | nil =>
      simp
  | cons head tail inductionHypothesis =>
      rw [List.flatMap_cons, satisfies_append_iff,
        inductionHypothesis]
      constructor
      · rintro ⟨headHolds, tailHolds⟩ receipt member
        rcases List.mem_cons.mp member with equal | tailMember
        · subst receipt
          exact headHolds
        · exact tailHolds receipt tailMember
      · intro all
        exact ⟨
          all head List.mem_cons_self,
          fun receipt member =>
            all receipt (List.mem_cons_of_mem head member)⟩

namespace Encoding

/-- Satisfaction of a conserved physical encoding projects to the complete
row list of any one of its receipts. -/
theorem receiptSatisfies
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source)
    (assignment : ColumnId -> Field)
    (physical : encoding.PhysicalSatisfies assignment)
    (receipt : InstructionReceipt)
    (member : receipt ∈ encoding.receipts) :
    Satisfies receipt.rows assignment := by
  exact
    (satisfies_flattened_receipts_iff encoding.receipts assignment).mp
      (by simpa only [Encoding.rows] using physical.2)
      receipt member

end Encoding

end Nightstream.Implementation.Lowering.Goldilocks
