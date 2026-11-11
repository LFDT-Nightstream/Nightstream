import NightstreamFPrime.Export.Generic.AuxRelocation
import NightstreamFPrime.Lifecycle.Nebula.StepRefinement

set_option autoImplicit false

namespace NightstreamFPrime.Export.Generic.ReadOnlyValues

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle.Nebula

theorem word_bits_eq {read write : Fin 32 → F}
    (readBits : ∀ k, IsBit (read k)) (writeBits : ∀ k, IsBit (write k))
    (same : bitsWord read = bitsWord write) : read = write := by
  have numbers := StepWitness.bitsWord_inj (by decide) readBits writeBits same
  have lists : (List.ofFn read).map toBool = (List.ofFn write).map toBool := by
    rw [← bitsLE_bitsNat, ← bitsLE_bitsNat, numbers]
  rw [List.map_ofFn, List.map_ofFn] at lists
  have booleans := List.ofFn_injective lists
  funext k
  have sameBool : toBool (read k) = toBool (write k) := congrFun booleans k
  rw [← natWord_toBool (readBits k), ← natWord_toBool (writeBits k), sameBool]

theorem read_rows_bits_eq {coefficient flag : F} {read write : Fin 32 → F}
    (nonzero : coefficient ≠ 0) (flagRow : coefficient * flag = 0)
    (readBits : ∀ k, IsBit (read k)) (writeBits : ∀ k, IsBit (write k))
    (keeps : (1 - flag) * (bitsWord write - bitsWord read) = 0) : read = write := by
  have zero : flag = 0 :=
    (GoldilocksPrime.baseFieldNoZeroDivisors coefficient flag flagRow).resolve_left nonzero
  rw [zero, sub_zero, one_mul, sub_eq_zero] at keeps
  exact word_bits_eq readBits writeBits keeps.symm

structure Port where
  flag : Nat
  read : Fin 32 → Nat
  write : Fin 32 → Nat

def Port.Holds (port : Port) (env : Env) : Prop :=
  ∃ coefficient : F, coefficient ≠ 0 ∧ coefficient * env port.flag = 0 ∧
    (∀ k, IsBit (env (port.read k))) ∧ (∀ k, IsBit (env (port.write k))) ∧
    (1 - env port.flag) *
      (bitsWord (fun k => env (port.write k)) - bitsWord (fun k => env (port.read k))) = 0

theorem port_bits_eq {port : Port} {env : Env} (holds : port.Holds env) :
    ∀ k, env (port.read k) = env (port.write k) := by
  obtain ⟨coefficient, nonzero, flag, read, write, keeps⟩ := holds
  exact congrFun (read_rows_bits_eq nonzero flag read write keeps)

def Maps (ports : List Port) (map : Nat → Nat) : Prop :=
  ∀ column, map column = column ∨
    ∃ port ∈ ports, ∃ k, column = port.write k ∧ map column = port.read k

theorem agree {ports : List Port} {map : Nat → Nat} (mapped : Maps ports map)
    {env : Env} (holds : ∀ port ∈ ports, port.Holds env) : env ∘ map = env := by
  funext column
  change env (map column) = env column
  rcases mapped column with fixed | ⟨port, member, k, written, read⟩
  · rw [fixed]
  · rw [read, written]
    exact port_bits_eq (holds port member) k

abbrev rename (map : Nat → Nat) : Expr → Expr := AuxRelocation.rename map

theorem renamed_holds (map : Nat → Nat) (source : List Expr) (env : Env) :
    ConstraintsHold env (source.map (rename map)) ↔ ConstraintsHold (env ∘ map) source := by
  constructor
  · intro holds row member
    have := holds _ (List.mem_map_of_mem member)
    simpa only [rename, AuxRelocation.rename, Lifecycle.Nebula.Expr.eval_subst,
      Expr.eval_var, Function.comp_def] using this
  · intro holds row member
    obtain ⟨sourceRow, sourceMember, rfl⟩ := List.mem_map.mp member
    simpa only [rename, AuxRelocation.rename, Lifecycle.Nebula.Expr.eval_subst,
      Expr.eval_var, Function.comp_def] using holds sourceRow sourceMember

inductive Reduction : List Expr → List Expr → Prop where
  | empty : Reduction [] []
  | keep {source kept : List Expr} (prior : Reduction source kept) (row : Expr) :
      Reduction (source ++ [row]) (kept ++ [row])
  | zero {source kept : List Expr} (prior : Reduction source kept) (row : Expr)
      (zero : ∀ env, row.eval env = 0) : Reduction (source ++ [row]) kept
  | duplicate {source kept : List Expr} (prior : Reduction source kept) (row previous : Expr)
      (member : previous ∈ kept) (same : ∀ env, row.eval env = previous.eval env) :
      Reduction (source ++ [row]) kept

theorem reduced_holds {source kept : List Expr} (reduced : Reduction source kept) (env : Env) :
    ConstraintsHold env source ↔ ConstraintsHold env kept := by
  have append (rows : List Expr) (row : Expr) :
      ConstraintsHold env (rows ++ [row]) ↔ ConstraintsHold env rows ∧ row.eval env = 0 := by
    constructor
    · intro holds
      exact ⟨fun previous member => holds previous (List.mem_append_left _ member),
        holds row (List.mem_append_right _ (List.mem_singleton.mpr rfl))⟩
    · rintro ⟨holds, last⟩ previous member
      rcases List.mem_append.mp member with member | member
      · exact holds previous member
      · have equal := List.mem_singleton.mp member
        simpa only [equal] using last
  induction reduced with
  | empty => rfl
  | keep prior row ih =>
      constructor
      · intro holds
        obtain ⟨old, last⟩ := (append _ _).mp holds
        exact (append _ _).mpr ⟨ih.mp old, last⟩
      · intro holds
        obtain ⟨old, last⟩ := (append _ _).mp holds
        exact (append _ _).mpr ⟨ih.mpr old, last⟩
  | zero prior row zero ih =>
      constructor
      · intro holds
        exact ih.mp ((append _ _).mp holds).1
      · intro holds
        exact (append _ _).mpr ⟨ih.mpr holds, zero env⟩
  | duplicate prior row previous member same ih =>
      constructor
      · intro holds
        exact ih.mp ((append _ _).mp holds).1
      · intro holds
        exact (append _ _).mpr ⟨ih.mpr holds, (same env).trans (holds previous member)⟩

theorem reconstruct {map : Nat → Nat} {source kept : List Expr}
    (reduced : Reduction (source.map (rename map)) kept) {env : Env}
    (holds : ConstraintsHold env kept) :
    ConstraintsHold (env ∘ map) source ∧
      ∀ column, map column = column → (env ∘ map) column = env column := by
  refine ⟨(renamed_holds map source env).mp ((reduced_holds reduced env).mpr holds), ?_⟩
  intro column fixed
  change env (map column) = env column
  rw [fixed]

theorem honest_complete {ports : List Port} {map : Nat → Nat} {source kept : List Expr}
    (mapped : Maps ports map) (portsHold : ∀ env, ConstraintsHold env source →
      ∀ port ∈ ports, port.Holds env)
    (reduced : Reduction (source.map (rename map)) kept) {env : Env}
    (holds : ConstraintsHold env source) : ConstraintsHold env kept := by
  apply (reduced_holds reduced env).mp
  apply (renamed_holds map source env).mpr
  rw [agree mapped (portsHold env holds)]
  exact holds

end NightstreamFPrime.Export.Generic.ReadOnlyValues
