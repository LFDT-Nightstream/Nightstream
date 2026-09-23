import NightstreamFPrime.Gadgets.Polynomial.HornerSupport
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain

/-! Variable support for the materialized PiCCS SumCheck chain. -/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.SumCheck
open NightstreamFPrime.Gadgets.Polynomial

abbrev KSupported := Horner.KSupported

def RoundSupported {degree : Nat} (round : FixedChain.Round degree)
    (allowed : Nat → Prop) : Prop :=
  (∀ coefficient, KSupported (round.coefficient coefficient) allowed) ∧
    KSupported round.challenge allowed

private theorem coefficientSum_supported (coefficients : List KExpr)
    (allowed : Nat → Prop)
    (support : ∀ coefficient ∈ coefficients, KSupported coefficient allowed) :
    KSupported (CompactChain.coefficientSum coefficients) allowed := by
  induction coefficients with
  | nil => exact Horner.KSupported.zero allowed
  | cons coefficient rest ih =>
      exact (support coefficient (by simp)).add
        (ih (fun value member => support value (by simp [member])))

private theorem booleanSum_supported (coefficients : List KExpr)
    (allowed : Nat → Prop)
    (support : ∀ coefficient ∈ coefficients, KSupported coefficient allowed) :
    KSupported (CompactChain.booleanSum coefficients) allowed := by
  apply Horner.KSupported.add
  · cases coefficients with
    | nil => exact Horner.KSupported.zero allowed
    | cons coefficient rest => exact support coefficient (by simp)
  · exact coefficientSum_supported coefficients allowed support

private theorem equalities_supported (left right : KExpr)
    (allowed : Nat → Prop) (leftSupport : KSupported left allowed)
    (rightSupport : KSupported right allowed) :
    ∀ expression ∈ KExpr.equalities left right, expression.VarsSatisfy allowed := by
  intro expression member
  simp only [KExpr.equalities, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · exact ⟨leftSupport.1, ⟨trivial, rightSupport.1⟩⟩
  · exact ⟨leftSupport.2, ⟨trivial, rightSupport.2⟩⟩

private theorem compile_supported {degree : Nat} (start : Nat) (current : KExpr)
    (rounds : List (FixedChain.Round degree)) (allowed : Nat → Prop)
    (currentSupport : KSupported current allowed)
    (roundsSupport : ∀ round ∈ rounds, RoundSupported round allowed)
    (locals : ∀ index, start ≤ index →
      index < start + (CompactChain.compile start current rounds).recipes.length → allowed index) :
    (∀ expression ∈ (CompactChain.compile start current rounds).recipes,
      expression.VarsSatisfy allowed) ∧
    KSupported (CompactChain.compile start current rounds).output allowed ∧
    (∀ expression ∈ (CompactChain.compile start current rounds).checks,
      expression.VarsSatisfy allowed) := by
  induction rounds generalizing start current with
  | nil => simp [CompactChain.compile, currentSupport]
  | cons round rounds ih =>
      let evaluation := Horner.compile start round.challenge round.coefficients
      let next := start + evaluation.recipes.length
      have support := roundsSupport round (by simp)
      have coefficients : ∀ coefficient ∈ round.coefficients,
          KSupported coefficient allowed := by
        intro coefficient member
        rw [FixedChain.Round.coefficients, List.mem_ofFn'] at member
        obtain ⟨index, rfl⟩ := member
        exact support.1 index
      have horner := Horner.compile_varsSatisfy start round.challenge round.coefficients
        allowed support.2 coefficients
      have includes : ∀ index, SupportRange.Extend allowed start next index → allowed index := by
        intro index member
        rcases member with existing | ⟨lower, upper⟩
        · exact existing
        · apply locals index lower
          simp only [CompactChain.compile_cons, List.length_append]
          dsimp [next, evaluation] at upper
          omega
      have evaluated : KSupported evaluation.output allowed := horner.2.mono includes
      have tail := ih next evaluation.output evaluated (fun later member =>
        roundsSupport later (by simp [member])) (by
          intro index lower upper
          apply locals index (by dsimp [next] at lower; omega)
          simpa only [CompactChain.compile_cons, List.length_append, next, evaluation,
            Nat.add_assoc] using upper)
      rw [CompactChain.compile_cons]
      refine ⟨?_, tail.2.1, ?_⟩
      · intro expression member
        rcases List.mem_append.mp member with first | later
        · exact Expr.VarsSatisfy.mono expression (horner.1 expression first) includes
        · exact tail.1 expression later
      · intro expression member
        rcases List.mem_append.mp member with first | later
        · exact equalities_supported current (CompactChain.booleanSum round.coefficients)
            allowed currentSupport (booleanSum_supported round.coefficients allowed coefficients)
            expression first
        · exact tail.2.2 expression later

private theorem program_supported {degree : Nat} (interface : Interface degree)
    (offset : Nat) (allowed : Nat → Prop)
    (initialSupport : KSupported (interface.initial offset) allowed)
    (roundSupport : ∀ index, RoundSupported (interface.round offset index) allowed)
    (localSupport : ∀ index, offset ≤ index →
      index < offset + localLength (Circuit.ops (circuit interface).main offset) → allowed index) :
    (∀ expression ∈ (CompactChain.program (coreInterface interface offset) offset).recipes,
      expression.VarsSatisfy allowed) ∧
    KSupported (output interface offset) allowed ∧
    (∀ expression ∈ (CompactChain.program (coreInterface interface offset) offset).checks,
      expression.VarsSatisfy allowed) := by
  apply compile_supported offset (coreInterface interface offset).initial
    (coreInterface interface offset).rounds allowed initialSupport
  · intro round member
    rw [FixedChain.Owned.Interface.rounds, List.mem_ofFn'] at member
    obtain ⟨index, rfl⟩ := member
    exact roundSupport index
  · intro index lower upper
    apply localSupport index lower
    rw [localLength_eq]
    simpa only [CompactChain.compile_recipes_length, FixedChain.Owned.Interface.rounds,
      List.length_ofFn, privateCount] using upper

theorem flatConstraints_varsSatisfy {degree : Nat}
    (interface : Interface degree) (offset : Nat) (allowed : Nat → Prop)
    (initialSupport : KSupported (interface.initial offset) allowed)
    (roundSupport : ∀ index, RoundSupported (interface.round offset index) allowed)
    (localSupport : ∀ index, offset ≤ index →
      index < offset + localLength (Circuit.ops (circuit interface).main offset) → allowed index) :
    ∀ expression ∈ flatConstraints (Circuit.ops (circuit interface).main offset),
      expression.VarsSatisfy allowed := by
  have support := program_supported interface offset allowed initialSupport roundSupport localSupport
  change ∀ expression ∈ flatConstraints (CompactChain.opsAt (coreInterface interface offset) offset), _
  rw [CompactChain.flatConstraints_opsAt]
  intro expression member
  rcases List.mem_append.mp member with recipeMember | checkMember
  · apply Horner.recipeConstraints_varsSatisfy offset _ allowed support.1 _ expression recipeMember
    intro index bound
    apply localSupport (offset + index) (by omega)
    rw [localLength_eq]
    simp only [CompactChain.program, CompactChain.compile_recipes_length,
      FixedChain.Owned.Interface.rounds, List.length_ofFn] at bound
    unfold privateCount
    omega
  · exact support.2.2 expression checkMember

theorem output_varsSatisfy {degree : Nat}
    (interface : Interface degree) (offset : Nat) (allowed : Nat → Prop)
    (initialSupport : KSupported (interface.initial offset) allowed)
    (roundSupport : ∀ index, RoundSupported (interface.round offset index) allowed)
    (localSupport : ∀ index, offset ≤ index →
      index < offset + localLength (Circuit.ops (circuit interface).main offset) → allowed index) :
    KSupported (output interface offset) allowed :=
  (program_supported interface offset allowed initialSupport roundSupport localSupport).2.1

end NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain
