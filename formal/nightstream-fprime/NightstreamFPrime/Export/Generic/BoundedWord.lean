import NightstreamFPrime.Export.Generic.PackedBoolean
import NightstreamFPrime.Export.Generic.GuardFusion

set_option autoImplicit false

namespace NightstreamFPrime.Export.Generic.BoundedWord

open NightstreamFPrime.Spec NightstreamFPrime.Circuit NightstreamFPrime.Layout
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.Radix
open Fin.CommRing

def packNat : List Nat → Nat
  | [] => 0
  | x :: xs => x + 2 * packNat xs

theorem packNat_lt (xs : List Nat) (bits : ∀ x ∈ xs, x ≤ 1) :
    packNat xs < 2 ^ xs.length := by
  induction xs with
  | nil => simp [packNat]
  | cons x xs ih =>
    have head := bits x List.mem_cons_self
    have tail := ih (fun v hv => bits v (List.mem_cons_of_mem _ hv))
    simp only [packNat, List.length_cons, pow_succ]
    omega

theorem packNat_injective (xs ys : List Nat) (length : xs.length = ys.length)
    (left : ∀ x ∈ xs, x ≤ 1) (right : ∀ y ∈ ys, y ≤ 1)
    (same : packNat xs = packNat ys) : xs = ys := by
  induction xs generalizing ys with
  | nil => simpa using length.symm
  | cons x xs ih =>
    cases ys with
    | nil => simp at length
    | cons y ys =>
      have hx := left x List.mem_cons_self
      have hy := right y List.mem_cons_self
      simp only [packNat] at same
      have head : x = y := by omega
      have rest : packNat xs = packNat ys := by omega
      have lengths : xs.length = ys.length := by simpa using length
      have tail := ih ys lengths (fun v hv => left v (List.mem_cons_of_mem _ hv))
        (fun v hv => right v (List.mem_cons_of_mem _ hv)) rest
      simp [head, tail]

def bitNat (x : F) : Nat := if x = 1 then 1 else 0

theorem bitNat_le (x : F) : bitNat x ≤ 1 := by
  unfold bitNat
  split <;> omega

theorem bitNat_field (x : F) (bit : x = 0 ∨ x = 1) :
    fieldOfNat (bitNat x) = x := by
  rcases bit with rfl | rfl <;> decide

def pack : List (F × F) → F
  | [] => 0
  | (x,y) :: rest => x - y + 2 * pack rest

def Bits (pairs : List (F × F)) : Prop :=
  ∀ pair ∈ pairs, (pair.1 = 0 ∨ pair.1 = 1) ∧ (pair.2 = 0 ∨ pair.2 = 1)

theorem pack_field (pairs : List (F × F)) (bits : Bits pairs) :
    fieldOfNat (packNat (pairs.map fun p => bitNat p.1)) -
      fieldOfNat (packNat (pairs.map fun p => bitNat p.2)) = pack pairs := by
  induction pairs with
  | nil => simp [packNat, pack, fieldOfNat]
  | cons pair rest ih =>
    have first := bits pair List.mem_cons_self
    have tail := ih (fun p hp => bits p (List.mem_cons_of_mem _ hp))
    simp only [List.map_cons, packNat, fieldOfNat_add, fieldOfNat_mul]
    rw [bitNat_field pair.1 first.1, bitNat_field pair.2 first.2]
    rcases pair with ⟨x,y⟩
    change x + 2 * fieldOfNat (packNat (rest.map fun p => bitNat p.1)) -
      (y + 2 * fieldOfNat (packNat (rest.map fun p => bitNat p.2))) = x - y + 2 * pack rest
    rw [← tail]
    ring

theorem nat_bound (pairs : List (F × F)) (count : pairs.length ≤ 63) (side : F × F → F) :
    packNat (pairs.map fun p => bitNat (side p)) < goldilocksModulus := by
  have small := packNat_lt (pairs.map fun p => bitNat (side p)) (by
    intro v member
    obtain ⟨p, _, rfl⟩ := List.mem_map.mp member
    exact bitNat_le (side p))
  have power : 2 ^ pairs.length ≤ 2 ^ 63 := Nat.pow_le_pow_right (by decide) count
  have bound : 2 ^ 63 < goldilocksModulus := by decide
  simp only [List.length_map] at small
  omega

theorem packed_implies_equal (pairs : List (F × F)) (count : pairs.length ≤ 63)
    (bits : Bits pairs) (zero : pack pairs = 0) : ∀ p ∈ pairs, p.1 = p.2 := by
  have field := pack_field pairs bits
  rw [zero] at field
  have casts := sub_eq_zero.mp field
  have values := congrArg Fin.val casts
  have lb := nat_bound pairs count Prod.fst
  have rb := nat_bound pairs count Prod.snd
  have same : packNat (pairs.map fun p => bitNat p.1) = packNat (pairs.map fun p => bitNat p.2) := by
    simpa only [fieldOfNat, Nat.mod_eq_of_lt lb, Nat.mod_eq_of_lt rb] using values
  have maps := packNat_injective (pairs.map fun p => bitNat p.1) (pairs.map fun p => bitNat p.2)
    (by simp) (by intro x hx; obtain ⟨p,_,rfl⟩ := List.mem_map.mp hx; exact bitNat_le p.1)
    (by intro y hy; obtain ⟨p,_,rfl⟩ := List.mem_map.mp hy; exact bitNat_le p.2) same
  have pointwise : ∀ p ∈ pairs, bitNat p.1 = bitNat p.2 := by
    exact (List.map_inj_left.mp maps)
  intro p hp
  have cast := congrArg fieldOfNat (pointwise p hp)
  rw [bitNat_field p.1 (bits p hp).1, bitNat_field p.2 (bits p hp).2] at cast
  exact cast

theorem equal_implies_packed (pairs : List (F × F)) (equal : ∀ p ∈ pairs, p.1 = p.2) :
    pack pairs = 0 := by
  induction pairs with
  | nil => rfl
  | cons pair rest ih =>
    have first := equal pair List.mem_cons_self
    have tail := ih (fun p hp => equal p (List.mem_cons_of_mem _ hp))
    rcases pair with ⟨x,y⟩
    change x = y at first
    simp only [pack, first, sub_self, tail, mul_zero, add_zero]

theorem common_guard_iff (gate : F) (pairs : List (F × F)) (count : pairs.length ≤ 63)
    (bits : Bits pairs) : gate * pack pairs = 0 ↔ ∀ p ∈ pairs, gate * (p.1 - p.2) = 0 := by
  by_cases zero : gate = 0
  · simp [zero]
  constructor
  · intro holds
    have packed := (GoldilocksPrime.baseFieldNoZeroDivisors _ _ holds).resolve_left zero
    intro p hp
    rw [packed_implies_equal pairs count bits packed p hp, sub_self, mul_zero]
  · intro holds
    have equal : ∀ p ∈ pairs, p.1 = p.2 := by
      intro p hp
      exact sub_eq_zero.mp ((GoldilocksPrime.baseFieldNoZeroDivisors _ _ (holds p hp)).resolve_left zero)
    rw [equal_implies_packed pairs equal, mul_zero]


end NightstreamFPrime.Export.Generic.BoundedWord
