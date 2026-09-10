import NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerFiberCount
import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.StoredFiberTables

/-!
Relates stored initial DP counts to the complete scalar comparison fiber.
The fallback is a semantic parameter; no uncharged executable comparison
against a function-valued fallback is implemented. Actual sampling still
aborts. Uniform rank generation and the global inverse remain separate.
-/

namespace NightstreamFPrime.Lifecycle.PiRLC.v1_1.StoredSamplerFiberTables

open NightstreamFPrime.Spec
open Folding.Nifs.NonInteractive.PiRlcSampler
open ProductionAlphabet ProductionStrongSet

/-- The stored initial counts give the exact scalarwise totalized fiber size. -/
theorem fiber_card (target : StoredFiberTables.Target) (fallback : Scalar) :
    Nat.card {fields : FieldShortfall.FieldWindow //
      SamplerTotalizedOutputLaw.totalizedFieldDecode fallback fields = target.get} =
      let initial := (StoredFiberTables.read (StoredFiberTables.build target).value
        (Fin.last FieldShortfall.fieldLaneCount) ⟨0, by decide⟩).value
      initial.success + if target.get = fallback then initial.aborts else 0 := by
  dsimp only
  rw [SamplerFiberCount.totalized_fiber_card,
    (StoredFiberTables.read_build_value target _ _).1,
    (StoredFiberTables.read_build_value target _ _).2,
    List.drop_zero, Nat.sub_zero]
  rfl

/-- Positivity is proved from the complete target length and rectangle recurrences. -/
theorem fiber_positive (target : StoredFiberTables.Target) (fallback : Scalar) :
    0 < Nat.card {fields : FieldShortfall.FieldWindow //
      SamplerTotalizedOutputLaw.totalizedFieldDecode fallback fields = target.get} := by
  rw [fiber_card]
  have positive := StoredFiberTables.initial_success_pos target
  dsimp only
  omega

/-- This is an integer-size bound for the actual normalizer, not a sampling-time bound. -/
theorem fiber_lt_twoPow2048 (target : StoredFiberTables.Target) (fallback : Scalar) :
    Nat.card {fields : FieldShortfall.FieldWindow //
      SamplerTotalizedOutputLaw.totalizedFieldDecode fallback fields = target.get} < 2 ^ 2048 := by
  rw [fiber_card]
  have bounded := StoredFiberTables.build_sum_lt_twoPow2048 target
    (Fin.last FieldShortfall.fieldLaneCount) ⟨0, by decide⟩
  change
    (StoredFiberTables.read (StoredFiberTables.build target).value
      (Fin.last FieldShortfall.fieldLaneCount) ⟨0, by decide⟩).value.success +
    (StoredFiberTables.read (StoredFiberTables.build target).value
      (Fin.last FieldShortfall.fieldLaneCount) ⟨0, by decide⟩).value.aborts < 2 ^ 2048 at bounded
  dsimp only
  split_ifs <;> omega

end NightstreamFPrime.Lifecycle.PiRLC.v1_1.StoredSamplerFiberTables
