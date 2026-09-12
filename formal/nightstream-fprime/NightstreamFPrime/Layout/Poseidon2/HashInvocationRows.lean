import NightstreamFPrime.Layout.Poseidon2

/-!
Owns the per-invocation recipe-row projection of the existing Poseidon2 hash
compiler. Absorption and final padding retain their source compiler inputs,
outputs, and witness starts. No invocation list or alternate trace is created.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Layout.Poseidon2.HashInvocationRows

open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2

private theorem output_nonempty (start : Nat) (state : Permutation.EState)
    (block : List Expr) (rest : List (List Expr)) :
    (Hash.compileAbsorptions start state (block :: rest)).output =
      Permutation.freshState (start + rest.length * 592 + 584) := by
  induction rest generalizing start state block with
  | nil =>
      simp only [Hash.compileAbsorptions, List.length_nil, Nat.zero_mul,
        Nat.add_zero]
      exact compile_schedule_output_eq start (Hash.absorbE state block)
  | cons next rest induction =>
      change (Hash.compileAbsorptions (start + 592)
        (Permutation.compile start (Hash.absorbE state block)
          Permutation.schedule).output (next :: rest)).output = _
      rw [induction]
      apply congrArg Permutation.freshState
      simp only [List.length_cons]
      omega

/-- Each actual absorption invocation inherits its complete recipe rows from
its containing hash compiler. Its input uses the preceding allocated state. -/
theorem absorption_rows (env : Env) (start : Nat)
    (state : Permutation.EState) (blocks : List (List Expr))
    (rows : ConstraintsHold env (recipeConstraints start
      (Hash.compileAbsorptions start state blocks).recipes))
    (invocation : Nat) (bound : invocation < blocks.length) :
    ConstraintsHold env (recipeConstraints (start + invocation * 592)
      (Permutation.compile (start + invocation * 592)
        (Hash.absorbE
          (if invocation = 0 then state else
            Permutation.freshState (start + (invocation - 1) * 592 + 584))
          (blocks.getD invocation [])) Permutation.schedule).recipes) := by
  induction blocks generalizing start state invocation with
  | nil => simp at bound
  | cons block rest induction =>
      have separated :
          ConstraintsHold env (recipeConstraints start
            (Permutation.compile start (Hash.absorbE state block)
              Permutation.schedule).recipes) ∧
          ConstraintsHold env (recipeConstraints (start + 592)
            (Hash.compileAbsorptions (start + 592)
              (Permutation.compile start (Hash.absorbE state block)
                Permutation.schedule).output rest).recipes) := by
        rw [Hash.compileAbsorptions, Permutation.recipeConstraints_append] at rows
        simpa only [Permutation.compile_schedule_recipe_count] using
          (constraintsHold_append env _ _).mp rows
      cases invocation with
      | zero =>
          simpa using separated.1
      | succ index =>
          have tail := induction (start + 592)
            (Permutation.compile start (Hash.absorbE state block)
              Permutation.schedule).output separated.2 index (by
                simp only [List.length_cons] at bound
                omega)
          rw [compile_schedule_output_eq] at tail
          by_cases first : index = 0
          · subst index
            simpa using tail
          · have nextStart : start + 592 + index * 592 =
                start + (index + 1) * 592 := by omega
            have previousStart : start + 592 + (index - 1) * 592 + 584 =
                start + index * 592 + 584 := by omega
            simp only [if_neg first] at tail
            rw [nextStart, previousStart] at tail
            simpa only [Nat.succ_eq_add_one, Nat.add_sub_cancel,
              Nat.succ_ne_zero, if_false, List.getD_cons_succ] using tail

/-- Every absorption or final-padding permutation inherits its source rows
from the complete hash compiler, with the same allocated previous state. -/
theorem hash_rows (env : Env) (start : Nat) (input : List Expr)
    (rows : ConstraintsHold env
      (recipeConstraints start (Hash.compile start input).recipes))
    (invocation : Nat) (bound : invocation ≤ (Hash.inputChunks input).length) :
    let blocks := Hash.inputChunks input
    let previous := if invocation = 0 then Hash.zeroE else
      Permutation.freshState (start + (invocation - 1) * 592 + 584)
    ConstraintsHold env (recipeConstraints (start + invocation * 592)
      (Permutation.compile (start + invocation * 592)
        (if invocation < blocks.length then
          Hash.absorbE previous (blocks.getD invocation [])
        else Hash.padE previous) Permutation.schedule).recipes) := by
  let blocks := Hash.inputChunks input
  have separated :
      ConstraintsHold env (recipeConstraints start
        (Hash.compileAbsorptions start Hash.zeroE blocks).recipes) ∧
      ConstraintsHold env (recipeConstraints
        (start + (Hash.compileAbsorptions start Hash.zeroE blocks).recipes.length)
        (Permutation.compile
          (start + (Hash.compileAbsorptions start Hash.zeroE blocks).recipes.length)
          (Hash.padE (Hash.compileAbsorptions start Hash.zeroE blocks).output)
          Permutation.schedule).recipes) := by
    rw [Hash.compile, Permutation.recipeConstraints_append] at rows
    exact (constraintsHold_append env _ _).mp rows
  dsimp only
  by_cases absorbing : invocation < blocks.length
  · rw [if_pos absorbing]
    exact absorption_rows env start Hash.zeroE blocks separated.1 invocation absorbing
  · have final : invocation = blocks.length :=
      Nat.le_antisymm bound (Nat.le_of_not_gt absorbing)
    subst invocation
    rw [if_neg (Nat.lt_irrefl _)]
    have previous : (Hash.compileAbsorptions start Hash.zeroE blocks).output =
        if blocks.length = 0 then Hash.zeroE else
          Permutation.freshState (start + (blocks.length - 1) * 592 + 584) := by
      cases blocks with
      | nil => rfl
      | cons block rest =>
          simpa using output_nonempty start Hash.zeroE block rest
    simpa only [Hash.compileAbsorptions_recipes_length, previous] using separated.2

/-- The ordinary hash compiler packet contains its complete recipe packet. -/
theorem recipeRows_of_hashConstraints (interface : Formal.Interface)
    (start : Nat) (env : Env)
    (rows : ConstraintsHold env (hashConstraints interface start)) :
    ConstraintsHold env (recipeConstraints start
      (Hash.compile start (interface.input start)).recipes) := by
  change holdsFlat env (Formal.opsAt interface start) at rows
  have held := holdsFlat_implies_holds env (Formal.opsAt interface start) rows
  exact held (.witness (WitnessBatch.arithmetic start
    (Hash.compile start (interface.input start)).recipes)) (by
      exact List.mem_cons_self)

end NightstreamFPrime.Layout.Poseidon2.HashInvocationRows
