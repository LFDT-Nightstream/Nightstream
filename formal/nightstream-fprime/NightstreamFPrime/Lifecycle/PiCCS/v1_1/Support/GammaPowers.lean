import NightstreamFPrime.Gadgets.Polynomial.HornerSupport
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.GammaPowers

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_1.GammaPowers

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Polynomial

theorem wire_varsSatisfy (gamma : KExpr) (start index : Nat) (allowed : Nat → Prop)
    (gammaSupport : Horner.KSupported gamma allowed) (bounded : index ≤ 16)
    (localSupport : ∀ i, start ≤ i → i < start + 32 → allowed i) :
    Horner.KSupported (wire gamma start index) allowed := by
  cases index with
  | zero => exact gammaSupport
  | succ index =>
    change allowed (start + 2 * index) ∧ allowed (start + 2 * index + 1)
    exact ⟨localSupport _ (by omega) (by omega), localSupport _ (by omega) (by omega)⟩

theorem flatConstraints_varsSatisfy (gamma : KExpr) (start : Nat) (allowed : Nat → Prop)
    (gammaSupport : Horner.KSupported gamma allowed)
    (localSupport : ∀ i, start ≤ i → i < start + 32 → allowed i) :
    ∀ expression ∈ flatConstraints (Circuit.ops (circuit gamma).main start),
      expression.VarsSatisfy allowed := by
  rw [flatConstraints_eq]
  apply Horner.recipeConstraints_varsSatisfy
  · intro expression member
    simp only [recipes, List.mem_ofFn] at member
    obtain ⟨index, rfl⟩ := member
    let step : Fin 16 := ⟨index.val / 2, by omega⟩
    have valid := schedule_valid step
    have left := wire_varsSatisfy gamma start (arguments step).1 allowed gammaSupport
      (by have := step.isLt; omega) localSupport
    have right := wire_varsSatisfy gamma start (arguments step).2 allowed gammaSupport
      (by have := step.isLt; omega) localSupport
    have pair := Horner.KSupported.mul left right
    unfold recipe
    split
    · exact pair.1
    · exact pair.2
  · intro index bound
    rw [recipes_length] at bound
    exact localSupport _ (by omega) (by omega)

end NightstreamFPrime.Lifecycle.PiCCS.v1_1.GammaPowers
