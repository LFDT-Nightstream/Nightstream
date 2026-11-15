import Mathlib.Data.Nat.Count
import NightstreamFPrime.Spec.Algebra

namespace NightstreamFPrime.Spec.AjtaiSetupV1.ReductionBias

private theorem count_blocks (modulus blocks : Nat) (event : Nat → Prop)
    [DecidablePred event] :
    Nat.count (fun value => event (value % modulus)) (modulus * blocks) =
      blocks * Nat.count (fun value => event (value % modulus)) modulus := by
  induction blocks with
  | zero => simp
  | succ blocks ih =>
    rw [Nat.mul_succ, Nat.count_add, ih]
    have shift : (fun value => event ((modulus * blocks + value) % modulus)) =
        (fun value => event (value % modulus)) := by
      funext value
      simp [Nat.add_mod]
    simp only [shift, Nat.succ_mul]

theorem event_count (samples modulus : Nat) (event : Nat → Prop)
    [DecidablePred event] :
    Nat.count (fun value => event (value % modulus)) samples =
      samples / modulus * Nat.count (fun value => event (value % modulus)) modulus +
      Nat.count (fun value => event (value % modulus)) (samples % modulus) := by
  conv_lhs => rw [← Nat.mod_add_div samples modulus, Nat.add_comm]
  rw [Nat.count_add, count_blocks]
  congr 1
  congr 1
  funext value
  simp [Nat.add_mod]

end NightstreamFPrime.Spec.AjtaiSetupV1.ReductionBias
