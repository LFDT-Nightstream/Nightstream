import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.OracleModel

/-! The 17-read schedule under the ideal block-oracle assumption. A domain
entry adds [4,i,0,0]; the digest advance adds a zero block. Thus every read
has a distinct normalized history. For an initial cache with no scheduled
query, the answers are exactly 17 joint uniform four-field blocks.

This fresh-batch law is not a claim that an adversarially chosen transcript
has fresh queries. Such executions use OracleModel.run_bias_bound with all
oracle calls counted. The extractor below still runs with uniform challenges. -/

namespace NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.ScheduleLaw

open NightstreamFPrime.Spec
open PiRlcSampler.ProductionStrongSet
open OracleModel

/-- Completed scalar entries and digest advances before the selected read. -/
def historyAt (initial : List Draw) (count : Nat) : List Draw :=
  initial ++ (List.range count).flatMap (fun coordinate =>
    [(fun lane => if lane.val = 0 then Poseidon2.ofNat 4
      else if lane.val = 1 then Poseidon2.ofNat coordinate else 0), zeroBlock])

def queryAt (initial : List Draw) (coordinate : Fin 17) : Query :=
  scalarQuery (historyAt initial coordinate.val) coordinate 0

theorem historyAt_length (initial : List Draw) (count : Nat) :
    (historyAt initial count).length = initial.length + 2 * count := by
  simp [historyAt, List.length_flatMap, List.map_const', Nat.mul_comm]

theorem queryAt_length (initial : List Draw) (coordinate : Fin 17) :
    (queryAt initial coordinate).val.length = initial.length + 2 * coordinate.val + 1 := by
  simp only [queryAt, scalarQuery, List.replicate_zero, List.append_nil,
    List.length_append, List.length_cons, List.length_nil, historyAt_length]

theorem queryAt_injective (initial : List Draw) : Function.Injective (queryAt initial) := by
  intro left right same
  have lengths := congrArg (fun query : Query => query.val.length) same
  dsimp only at lengths
  rw [queryAt_length, queryAt_length] at lengths
  exact Fin.ext (by omega)

/-- A finite oracle table for a batch whose scheduled queries were absent
from the incoming cache. The raw four-lane block is the complete answer. -/
def Fresh (initial : List Draw) (cache : Cache) : Prop :=
  ∀ coordinate, cache (queryAt initial coordinate) = none

noncomputable def table (initial : List Draw) (cache : Cache) (_fresh : Fresh initial cache)
    (blocks : Fin 17 → Draw) : Cache :=
  fun query => if found : ∃ coordinate, queryAt initial coordinate = query then
    some (blocks found.choose) else cache query

theorem table_at (initial : List Draw) (cache : Cache) (fresh : Fresh initial cache) (blocks : Fin 17 → Draw)
    (coordinate : Fin 17) : table initial cache fresh blocks (queryAt initial coordinate) = some (blocks coordinate) := by
  unfold table
  rw [dif_pos ⟨coordinate, rfl⟩]
  congr 2
  exact queryAt_injective initial (Exists.choose_spec (show ∃ index, queryAt initial index =
    queryAt initial coordinate from ⟨coordinate, rfl⟩))

theorem table_preserves (initial : List Draw) (cache : Cache) (blocks : Fin 17 → Draw)
    (fresh : Fresh initial cache)
    (query : Query) (saved : Draw) (known : cache query = some saved) :
    table initial cache fresh blocks query = some saved := by
  have absent : ¬∃ coordinate, queryAt initial coordinate = query := by
    rintro ⟨coordinate, rfl⟩
    rw [fresh coordinate] at known
    cases known
  simpa only [table, dif_neg absent] using known

noncomputable def readScalars (initial : List Draw) (cache : Cache) (fresh : Fresh initial cache) (blocks : Fin 17 → Draw) : Fin 17 → Scalar :=
  fun coordinate => sample ((table initial cache fresh blocks (queryAt initial coordinate)).getD (fun _ => 0))

theorem readScalars_eq (initial : List Draw) (cache : Cache) (fresh : Fresh initial cache) (blocks : Fin 17 → Draw) :
    readScalars initial cache fresh blocks = sampleVector blocks := by
  funext coordinate
  simp only [readScalars, table_at, Option.getD_some, sampleVector]

/-- With a fresh joint-block table, this is the exact law used by V6. -/
theorem fresh_batch_law (initial : List Draw) (cache : Cache) (fresh : Fresh initial cache)
    (test : (Fin 17 → Scalar) → ℝ) :
    average (fun blocks => test (readScalars initial cache fresh blocks)) =
      average (fun blocks : Fin 17 → Draw => test (sampleVector blocks)) := by
  simp only [readScalars_eq]

/-- L folds of independent fresh batches have 17*L sampled scalars. This
bound is restricted to that experiment, not general adaptive oracle use. -/
theorem fresh_folds_bias_bound (folds : Nat) (test : (Fin (17 * folds) → Scalar) → ℝ)
    (nonnegative : ∀ vector, 0 ≤ test vector) (atMostOne : ∀ vector, test vector ≤ 1) :
    |average (fun blocks : Fin (17 * folds) → Draw => test (sampleVector blocks)) - average test| ≤
      (17 * folds : Nat) * distance := by
  simpa only [Fintype.card_fin] using vector_average_difference_abs_le test nonnegative atMostOne

open NightstreamFPrime.Spec.Folding.PiRLC
open CoordinateRetry CoordinateOracle CoordinateOracleStar CoordinateTerminalLaw

/-- Compose the exact fresh-batch law with V6. The probability on the right
is the existing uniform-challenge extractor, not a Fiat–Shamir extractor. -/
theorem fresh_blocks_extractor_lower_bound {Assignment : Type*} [Fintype Assignment]
    (initial : List Draw) (cache : Cache) (fresh : Fresh initial cache) (oracle : Oracle (Fin 17) Scalar Assignment)
    (check : (Fin 17 → Scalar) → Assignment → Bool)
    (returns : (Fin 17 → Scalar) → Option Assignment →
      (Fin 17 → CoordinateOracle.Outcome (Challenge := Scalar) (Assignment := Assignment)) → Bool)
    (returnsOnFork : ∀ vector assignment outputs,
      0 < outcomeMass oracle check vector assignment outputs → returns vector assignment outputs = true) :
    average (fun blocks => (line oracle check).acceptance (readScalars initial cache fresh blocks)) -
        17 * distance - (17 : ℝ) / Fintype.card Scalar ≤ returningProbability oracle check returns := by
  rw [fresh_batch_law]
  simpa only [sampledRate, sampleVector, Fintype.card_fin] using
    returningProbability_sampled_lower_bound oracle check returns returnsOnFork

end NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.ScheduleLaw
