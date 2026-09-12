import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork
import Mathlib.Analysis.SpecificLimits.Basic

/-!
Finite execution prefixes for the adaptive uniqueness retry in SuperNeo v1.2
Section 6 and Appendix B.2. A response retains its oracle clock. Its Boolean
checker also returns its own clock. Search stops at the first accepted actual
response and charges every rejected call before it.

The declared clock charges three local transitions per visited response:
test dispatch, output/call-count update, and return. Exhaustion costs one.
Arithmetic on the instrumentation counters is bookkeeping, not an additional
machine-operation claim. This module does not sample oracle responses or
identify any response law with the concrete Poseidon2 transcript.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.AcceptedRetry

open NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork (Result)

variable {Value : Type*}

/-- Consume a supplied finite prefix, retaining the returned value, number
of calls and all observed work. Values after the first acceptance are unused. -/
def search (check : Value → Result Bool) : List (Result Value) →
    Result (Option Value × Nat)
  | [] => ⟨(none, 0), 1⟩
  | packet :: following =>
      let checked := check packet.value
      if checked.value then
        ⟨(some packet.value, 1), packet.work + checked.work + 3⟩
      else
        let next := search check following
        ⟨(next.value.1, next.value.2 + 1), packet.work + checked.work + next.work + 3⟩

/-- The finite driver returns the first accepted response itself. Its work
includes the oracle and checker clocks of every preceding rejected response. -/
theorem search_firstHit (check : Value → Result Bool)
    (before : List (Result Value)) (last : Result Value) (after : List (Result Value))
    (rejected : ∀ packet ∈ before, (check packet.value).value = false)
    (accepted : (check last.value).value = true) :
    search check (before ++ last :: after) =
      ⟨(some last.value, before.length + 1),
        (before.map fun packet => packet.work + (check packet.value).work + 3).sum +
          last.work + (check last.value).work + 3⟩ := by
  induction before with
  | nil => simp [search, accepted]
  | cons packet before induction =>
      have headRejected := rejected packet (by simp)
      have tailRejected : ∀ entry ∈ before, (check entry.value).value = false := by
        intro entry member
        exact rejected entry (by simp [member])
      simp [search, headRejected, induction tailRejected,
        Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]

/-- Exhausting a rejected prefix returns no endpoint and retains every call
and checker charge, together with the terminal exhaustion transition. -/
theorem search_exhausted (check : Value → Result Bool) (packets : List (Result Value))
    (rejected : ∀ packet ∈ packets, (check packet.value).value = false) :
    search check packets =
      ⟨(none, packets.length),
        (packets.map fun packet => packet.work + (check packet.value).work + 3).sum + 1⟩ := by
  induction packets with
  | nil => rfl
  | cons packet packets induction =>
      have headRejected := rejected packet (by simp)
      have tailRejected : ∀ entry ∈ packets, (check entry.value).value = false := by
        intro entry member
        exact rejected entry (by simp [member])
      simp [search, headRejected, induction tailRejected,
        Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]

/-- For independent calls with success rate `rate`, an accepted event with
one-call mass `mass` first occurs after a geometrically distributed rejected
prefix. Its conditional first-hit mass is `mass / rate`. The caller must
connect these masses to its actual response law. -/
theorem firstHit_hasSum (rate : ℝ) (positive : 0 < rate) (atMostOne : rate ≤ 1)
    (mass : ℝ) :
    HasSum (fun rejections : Nat => (1 - rate) ^ rejections * mass) (mass / rate) := by
  simpa only [sub_sub_cancel, div_eq_mul_inv, mul_comm] using
    (hasSum_geometric_of_lt_one (sub_nonneg.mpr atMostOne)
      (sub_lt_self 1 positive)).mul_right mass

/-- Entering only after base acceptance cancels the reciprocal retry cost.
The zero-success case enters no retry calls. `work` is the mean of one full
clocked call, including its checker and local transitions. -/
theorem entered_work_hasSum (rate work : ℝ)
    (nonnegative : 0 ≤ rate) (atMostOne : rate ≤ 1) :
    HasSum (fun priorCalls : Nat => rate * (1 - rate) ^ priorCalls * work)
      (if rate = 0 then 0 else work) := by
  by_cases zero : rate = 0
  · simp only [zero, zero_mul, ↓reduceIte]
    exact hasSum_zero
  · have positive : 0 < rate := lt_of_le_of_ne nonnegative (Ne.symm zero)
    have summed := (firstHit_hasSum rate positive atMostOne 1).mul_left (rate * work)
    have cancellation : rate * work * (1 / rate) = work := by
      calc
        _ = (work * rate) / rate := by ring
        _ = work := mul_div_cancel_right₀ work zero
    rw [cancellation] at summed
    simpa only [if_neg zero, mul_one, one_mul, mul_assoc, mul_comm, mul_left_comm] using summed

/-- The mass of entered retries that exhaust a growing observation prefix
tends to zero, including the zero-success case where no retry is entered. -/
theorem exhaustion_tendsTo_zero (rate : ℝ)
    (nonnegative : 0 ≤ rate) (atMostOne : rate ≤ 1) :
    Filter.Tendsto (fun calls : Nat => rate * (1 - rate) ^ calls)
      Filter.atTop (nhds 0) := by
  by_cases zero : rate = 0
  · simpa only [zero, zero_mul] using
      (tendsto_const_nhds : Filter.Tendsto (fun _ : Nat => (0 : ℝ)) Filter.atTop (nhds 0))
  · have positive : 0 < rate := lt_of_le_of_ne nonnegative (Ne.symm zero)
    simpa only [mul_zero] using
      (tendsto_pow_atTop_nhds_zero_of_lt_one (sub_nonneg.mpr atMostOne)
        (sub_lt_self 1 positive)).const_mul rate

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.AcceptedRetry
