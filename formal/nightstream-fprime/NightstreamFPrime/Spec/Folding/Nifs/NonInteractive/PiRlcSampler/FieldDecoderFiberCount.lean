import Mathlib.Data.List.OfFn
import Mathlib.Logic.Embedding.Set
import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.FieldPreimageRectangle
import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.FieldShortfall

/-!
Exact successful and aborting fibers of the existing bounded decoder on
ordered field windows. Counts are arithmetic recurrences, not a stored DP
implementation or a random sampler. Targets retain raw five-symbol indices:
raw index zero is centered coefficient -2, not centered zero. No transcript
distribution, inverse-sampling law, or execution cost is assumed.
-/

namespace NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.FieldDecoderFiberCount

open Sampling ProductionAlphabet FieldPairLaw
open FieldPreimageRectangle (ChunkClass)

abbrev Window (lanes : Nat) := Fin lanes → F

/-- Existing low/high candidates, in the stored lane order. -/
def candidateList {lanes : Nat} (fields : Window lanes) : List Chunk :=
  (List.ofFn fields).flatMap fun value => [(candidates value).1, (candidates value).2]

abbrev SuccessFiber (lanes : Nat) (target : List Coefficient) :=
  {fields : Window lanes //
    FirstAccepted.boundedSample verifier target.length (candidateList fields) = some target}

abbrev AbortFiber (lanes need : Nat) :=
  {fields : Window lanes //
    FirstAccepted.boundedSample verifier need (candidateList fields) = none}

private abbrev weight := FieldPreimageRectangle.size

/-- Successful states are requested suffixes. Reaching the empty suffix
leaves every remaining field unrestricted. -/
def successCount : Nat → List Coefficient → Nat
  | 0, [] => 1
  | 0, _ :: _ => 0
  | lanes + 1, [] => goldilocksModulus * successCount lanes []
  | lanes + 1, [digit] =>
      weight .reject .reject * successCount lanes [digit] +
      weight .reject (.residue digit) * successCount lanes [] +
      weight (.residue digit) .all * successCount lanes []
  | lanes + 1, first :: second :: suffix =>
      weight .reject .reject * successCount lanes (first :: second :: suffix) +
      weight .reject (.residue first) * successCount lanes (second :: suffix) +
      weight (.residue first) .reject * successCount lanes (second :: suffix) +
      weight (.residue first) (.residue second) * successCount lanes suffix

/-- Abort states restrict only the number of missing accepts. They impose
no condition on the accepted digits, including for the eventual fallback. -/
def abortCount : Nat → Nat → Nat
  | 0, need => if 0 < need then 1 else 0
  | _ + 1, 0 => 0
  | lanes + 1, need + 1 =>
      weight .reject .reject * abortCount lanes (need + 1) +
      weight .reject .accepted * abortCount lanes need +
      weight .accepted .reject * abortCount lanes need +
      weight .accepted .accepted * abortCount lanes (need - 1)

private def Matches (target : List Coefficient) (chunks : List Chunk) : Prop :=
  target <+: FirstAccepted.acceptedSymbols verifier chunks

private def Short (need : Nat) (chunks : List Chunk) : Prop :=
  FirstAccepted.acceptedCount verifier chunks < need

private theorem success_iff_matches (target : List Coefficient) (chunks : List Chunk) :
    FirstAccepted.boundedSample verifier target.length chunks = some target ↔
      Matches target chunks := by
  rw [FirstAccepted.boundedSample_eq_some_iff]
  constructor
  · intro success
    exact List.prefix_iff_eq_take.mpr success.2
  · intro matched
    refine ⟨?_, List.prefix_iff_eq_take.mp matched⟩
    have matchedPrefix : target <+: FirstAccepted.acceptedSymbols verifier chunks := matched
    simpa only [FirstAccepted.Enough, FirstAccepted.acceptedSymbols,
      FirstAccepted.acceptedCount, List.length_map] using matchedPrefix.length_le

private theorem abort_iff_short (need : Nat) (chunks : List Chunk) :
    FirstAccepted.boundedSample verifier need chunks = none ↔ Short need chunks :=
  FirstAccepted.boundedSample_eq_none_iff_shortfall

private theorem reject_member (chunk : Chunk) :
    ChunkClass.reject.Member chunk ↔ verifier.accepts chunk = false := by
  have negated := not_congr (accepts_eq_true_iff_ne_rejectionBucket chunk)
  cases accepted : verifier.accepts chunk <;>
    simpa [ChunkClass.Member, accepted] using negated.symm

private theorem reject_not_accepted (chunk : Chunk)
    (rejected : ChunkClass.reject.Member chunk) (accepted : verifier.accepts chunk = true) : False := by
  have falseValue := (reject_member chunk).mp rejected
  rw [accepted] at falseValue
  contradiction

private theorem verifier_symbol_eq : verifier.symbol = symbol := rfl

private theorem matches_one_pair (digit : Coefficient) (low high : Chunk) (tail : List Chunk) :
    Matches [digit] (low :: high :: tail) ↔
      (ChunkClass.reject.Member low ∧ ChunkClass.reject.Member high ∧ Matches [digit] tail) ∨
      (ChunkClass.reject.Member low ∧ (ChunkClass.residue digit).Member high ∧ Matches [] tail) ∨
      ((ChunkClass.residue digit).Member low ∧ ChunkClass.all.Member high ∧ Matches [] tail) := by
  rw [reject_member low, reject_member high]
  cases lowAccepted : verifier.accepts low <;> cases highAccepted : verifier.accepts high <;>
    simp [Matches, FirstAccepted.acceptedSymbols, FirstAccepted.acceptedCandidates,
      ChunkClass.Member, lowAccepted, highAccepted, List.cons_prefix_cons, eq_comm, verifier_symbol_eq]

private theorem matches_two_pair (first second : Coefficient) (suffix : List Coefficient)
    (low high : Chunk) (tail : List Chunk) :
    Matches (first :: second :: suffix) (low :: high :: tail) ↔
      (ChunkClass.reject.Member low ∧ ChunkClass.reject.Member high ∧
        Matches (first :: second :: suffix) tail) ∨
      (ChunkClass.reject.Member low ∧ (ChunkClass.residue first).Member high ∧
        Matches (second :: suffix) tail) ∨
      ((ChunkClass.residue first).Member low ∧ ChunkClass.reject.Member high ∧
        Matches (second :: suffix) tail) ∨
      ((ChunkClass.residue first).Member low ∧ (ChunkClass.residue second).Member high ∧
        Matches suffix tail) := by
  rw [reject_member low, reject_member high]
  cases lowAccepted : verifier.accepts low <;> cases highAccepted : verifier.accepts high <;>
    simp [Matches, FirstAccepted.acceptedSymbols, FirstAccepted.acceptedCandidates,
      ChunkClass.Member, lowAccepted, highAccepted, List.cons_prefix_cons, eq_comm, verifier_symbol_eq]

private theorem short_pair (need : Nat) (low high : Chunk) (tail : List Chunk) :
    Short (need + 1) (low :: high :: tail) ↔
      (ChunkClass.reject.Member low ∧ ChunkClass.reject.Member high ∧ Short (need + 1) tail) ∨
      (ChunkClass.reject.Member low ∧ ChunkClass.accepted.Member high ∧ Short need tail) ∨
      (ChunkClass.accepted.Member low ∧ ChunkClass.reject.Member high ∧ Short need tail) ∨
      (ChunkClass.accepted.Member low ∧ ChunkClass.accepted.Member high ∧ Short (need - 1) tail) := by
  rw [reject_member low, reject_member high]
  cases lowAccepted : verifier.accepts low <;> cases highAccepted : verifier.accepts high <;>
    simp [Short, FirstAccepted.acceptedCount, FirstAccepted.acceptedCandidates,
      ChunkClass.Member, lowAccepted, highAccepted] <;> omega

private theorem candidateList_zero (fields : Window 0) : candidateList fields = [] := rfl

private theorem candidateList_cons {lanes : Nat} (first : F) (tail : Window lanes) :
    candidateList (Fin.cons first tail) =
      (candidates first).1 :: (candidates first).2 :: candidateList tail := by
  simp [candidateList, List.ofFn_succ]

private def headTailEquiv (lanes : Nat) : Window (lanes + 1) ≃ F × Window lanes where
  toFun fields := (fields 0, fun index => fields index.succ)
  invFun fields := Fin.cons fields.1 fields.2
  left_inv fields := by
    funext index
    refine Fin.cases ?_ (fun position => ?_) index <;> rfl
  right_inv fields := by cases fields; rfl

private theorem window_card (lanes : Nat) : Nat.card (Window lanes) = goldilocksModulus ^ lanes := by
  have fieldCard : Nat.card F = goldilocksModulus := Nat.card_fin goldilocksModulus
  rw [Nat.card_fun, fieldCard, Nat.card_fin]

private theorem constant_fiber_card {Carrier : Type*} (condition : Prop) [Decidable condition] :
    Nat.card {_value : Carrier // condition} = if condition then Nat.card Carrier else 0 := by
  by_cases holds : condition <;>
    simp only [holds, if_true, if_false, Nat.card_subtype_true, Nat.card_of_isEmpty]

/-- Exact disjoint-fiber addition. The scalar totalization consumer uses
the same finite operation for its successful and aborting alternatives. -/
theorem disjoint_event_card {Carrier : Type*} [Finite Carrier]
    (left right : Carrier → Prop) (separate : ∀ value, left value → right value → False) :
    Nat.card {value : Carrier // left value ∨ right value} =
      Nat.card {value : Carrier // left value} + Nat.card {value : Carrier // right value} := by
  classical
  have disjoint : Disjoint (left : Set Carrier) (right : Set Carrier) :=
    Set.disjoint_left.mpr separate
  exact (Nat.card_congr (subtypeOrEquiv left right disjoint)).trans Nat.card_sum

private def Branch {Tail : Type*} (low high : ChunkClass) (next : Tail → Prop)
    (input : F × Tail) : Prop :=
  FieldPreimageRectangle.Member low high input.1 ∧ next input.2

private theorem branch_card {Tail : Type*} (low high : ChunkClass) (next : Tail → Prop) :
    Nat.card {input : F × Tail // Branch low high next input} =
      weight low high * Nat.card {tail : Tail // next tail} := by
  have rectangles : Nat.card {value : F // FieldPreimageRectangle.Member low high value} =
      weight low high := FieldPreimageRectangle.rectangle_card low high
  calc
    _ = Nat.card ({value : F // FieldPreimageRectangle.Member low high value} ×
        {tail : Tail // next tail}) := Nat.card_congr Equiv.subtypeProdEquivProd
    _ = _ := by rw [Nat.card_prod, rectangles]

private theorem three_branch_card {Tail : Type*} [Finite Tail]
    (digit : Coefficient) (stays done : Tail → Prop) :
    Nat.card {input : F × Tail //
      Branch .reject .reject stays input ∨
      Branch .reject (.residue digit) done input ∨
      Branch (.residue digit) .all done input} =
      weight .reject .reject * Nat.card {tail : Tail // stays tail} +
      weight .reject (.residue digit) * Nat.card {tail : Tail // done tail} +
      weight (.residue digit) .all * Nat.card {tail : Tail // done tail} := by
  have firstSeparate (input : F × Tail)
      (first : Branch .reject .reject stays input)
      (others : Branch .reject (.residue digit) done input ∨
        Branch (.residue digit) .all done input) : False := by
    rcases others with second | third
    · exact reject_not_accepted _ first.1.2 second.1.2.1
    · exact reject_not_accepted _ first.1.1 third.1.1.1
  have secondSeparate (input : F × Tail)
      (second : Branch .reject (.residue digit) done input)
      (third : Branch (.residue digit) .all done input) : False :=
    reject_not_accepted _ second.1.1 third.1.1.1
  rw [disjoint_event_card _ _ firstSeparate, disjoint_event_card _ _ secondSeparate,
    branch_card, branch_card, branch_card]
  omega

private theorem four_branch_card {Tail : Type*} [Finite Tail]
    (first second : ChunkClass)
    (firstAccepted : ∀ chunk, first.Member chunk → verifier.accepts chunk = true)
    (secondAccepted : ∀ chunk, second.Member chunk → verifier.accepts chunk = true)
    (neither oneLow oneHigh both : Tail → Prop) :
    Nat.card {input : F × Tail //
      Branch .reject .reject neither input ∨
      Branch .reject first oneHigh input ∨
      Branch first .reject oneLow input ∨
      Branch first second both input} =
      weight .reject .reject * Nat.card {tail : Tail // neither tail} +
      weight .reject first * Nat.card {tail : Tail // oneHigh tail} +
      weight first .reject * Nat.card {tail : Tail // oneLow tail} +
      weight first second * Nat.card {tail : Tail // both tail} := by
  have separate0 (input : F × Tail)
      (zero : Branch .reject .reject neither input)
      (others : Branch .reject first oneHigh input ∨
        Branch first .reject oneLow input ∨ Branch first second both input) : False := by
    rcases others with one | one | two
    · exact reject_not_accepted _ zero.1.2 (firstAccepted _ one.1.2)
    · exact reject_not_accepted _ zero.1.1 (firstAccepted _ one.1.1)
    · exact reject_not_accepted _ zero.1.1 (firstAccepted _ two.1.1)
  have separate1 (input : F × Tail)
      (one : Branch .reject first oneHigh input)
      (others : Branch first .reject oneLow input ∨ Branch first second both input) : False := by
    rcases others with other | two
    · exact reject_not_accepted _ one.1.1 (firstAccepted _ other.1.1)
    · exact reject_not_accepted _ one.1.1 (firstAccepted _ two.1.1)
  have separate2 (input : F × Tail)
      (one : Branch first .reject oneLow input) (two : Branch first second both input) : False :=
    reject_not_accepted _ one.1.2 (secondAccepted _ two.1.2)
  rw [disjoint_event_card _ _ separate0, disjoint_event_card _ _ separate1,
    disjoint_event_card _ _ separate2, branch_card, branch_card, branch_card, branch_card]
  omega

private theorem matches_nil_card (lanes : Nat) :
    Nat.card {fields : Window lanes // Matches [] (candidateList fields)} = goldilocksModulus ^ lanes := by
  have same := Nat.card_congr (Equiv.subtypeEquivRight
    (fun fields : Window lanes => show Matches [] (candidateList fields) ↔ True from
      iff_true_intro List.nil_prefix))
  exact same.trans (Nat.card_subtype_true.trans (window_card lanes))

theorem successCount_nil (lanes : Nat) : successCount lanes [] = goldilocksModulus ^ lanes := by
  induction lanes with
  | zero => rfl
  | succ lanes ih => rw [successCount, ih, pow_succ]; exact Nat.mul_comm _ _

private theorem matches_fiber_card (lanes : Nat) (target : List Coefficient) :
    Nat.card {fields : Window lanes // Matches target (candidateList fields)} =
      successCount lanes target := by
  induction lanes generalizing target with
  | zero =>
      cases target with
      | nil => exact matches_nil_card 0
      | cons first tail =>
          have same := Nat.card_congr (Equiv.subtypeEquivRight
            (fun fields : Window 0 => show Matches (first :: tail) (candidateList fields) ↔ False by
              rw [candidateList_zero]
              simp [Matches, FirstAccepted.acceptedSymbols, FirstAccepted.acceptedCandidates]))
          exact same.trans Nat.card_of_isEmpty
  | succ lanes ih =>
      cases target with
      | nil => rw [matches_nil_card, successCount_nil]
      | cons first tail =>
          cases tail with
          | nil =>
              let next := fun (target : List Coefficient) (fields : Window lanes) =>
                Matches target (candidateList fields)
              have split := Nat.card_congr (Equiv.subtypeEquiv (headTailEquiv lanes).symm
                (p := fun input =>
                  Branch .reject .reject (next [first]) input ∨
                  Branch .reject (.residue first) (next []) input ∨
                  Branch (.residue first) .all (next []) input)
                (q := fun fields => Matches [first] (candidateList fields)) (by
                  intro input
                  change _ ↔ Matches [first] (candidateList (Fin.cons input.1 input.2))
                  rw [candidateList_cons]
                  simpa only [Branch, FieldPreimageRectangle.Member, and_assoc] using
                    (matches_one_pair first (candidates input.1).1 (candidates input.1).2
                      (candidateList input.2)).symm))
              rw [← split, three_branch_card]
              simp only [next, ih, successCount]
          | cons second suffix =>
              let next := fun (target : List Coefficient) (fields : Window lanes) =>
                Matches target (candidateList fields)
              have split := Nat.card_congr (Equiv.subtypeEquiv (headTailEquiv lanes).symm
                (p := fun input =>
                  Branch .reject .reject (next (first :: second :: suffix)) input ∨
                  Branch .reject (.residue first) (next (second :: suffix)) input ∨
                  Branch (.residue first) .reject (next (second :: suffix)) input ∨
                  Branch (.residue first) (.residue second) (next suffix) input)
                (q := fun fields => Matches (first :: second :: suffix) (candidateList fields)) (by
                  intro input
                  change _ ↔ Matches (first :: second :: suffix)
                    (candidateList (Fin.cons input.1 input.2))
                  rw [candidateList_cons]
                  simpa only [Branch, FieldPreimageRectangle.Member, and_assoc] using
                    (matches_two_pair first second suffix (candidates input.1).1
                      (candidates input.1).2 (candidateList input.2)).symm))
              rw [← split, four_branch_card (.residue first) (.residue second)
                (fun _ accepted => accepted.1) (fun _ accepted => accepted.1)]
              simp only [next, ih, successCount]

/-- Exact successful fibers, for every lane count and complete raw target. -/
theorem success_fiber_card (lanes : Nat) (target : List Coefficient) :
    Nat.card (SuccessFiber lanes target) = successCount lanes target :=
  (Nat.card_congr (Equiv.subtypeEquivRight
    (fun fields => success_iff_matches target (candidateList fields)))).trans
      (matches_fiber_card lanes target)

private theorem short_zero_card (lanes : Nat) :
    Nat.card {fields : Window lanes // Short 0 (candidateList fields)} = 0 := by
  have same := Nat.card_congr (Equiv.subtypeEquivRight
    (fun fields : Window lanes => show Short 0 (candidateList fields) ↔ False by
      simp only [Short, Nat.not_lt_zero, iff_self]))
  exact same.trans Nat.card_of_isEmpty

theorem abortCount_zero (lanes : Nat) : abortCount lanes 0 = 0 := by cases lanes <;> rfl

private theorem short_fiber_card (lanes need : Nat) :
    Nat.card {fields : Window lanes // Short need (candidateList fields)} = abortCount lanes need := by
  induction lanes generalizing need with
  | zero =>
      have same := Nat.card_congr (Equiv.subtypeEquivRight
        (fun fields : Window 0 => show Short need (candidateList fields) ↔ 0 < need by rfl))
      rw [same, constant_fiber_card, window_card] <;> rfl
  | succ lanes ih =>
      cases need with
      | zero => rw [short_zero_card, abortCount_zero]
      | succ need =>
          let next := fun (needed : Nat) (fields : Window lanes) => Short needed (candidateList fields)
          have split := Nat.card_congr (Equiv.subtypeEquiv (headTailEquiv lanes).symm
            (p := fun input =>
              Branch .reject .reject (next (need + 1)) input ∨
              Branch .reject .accepted (next need) input ∨
              Branch .accepted .reject (next need) input ∨
              Branch .accepted .accepted (next (need - 1)) input)
            (q := fun fields => Short (need + 1) (candidateList fields)) (by
              intro input
              change _ ↔ Short (need + 1) (candidateList (Fin.cons input.1 input.2))
              rw [candidateList_cons]
              simpa only [Branch, FieldPreimageRectangle.Member, and_assoc] using
                (short_pair need (candidates input.1).1 (candidates input.1).2
                  (candidateList input.2)).symm))
          rw [← split, four_branch_card .accepted .accepted (fun _ accepted => accepted)
            (fun _ accepted => accepted)]
          simp only [next, ih, abortCount]

/-- Exact abort fibers count every failed prefix, with no digit constraint. -/
theorem abort_fiber_card (lanes need : Nat) :
    Nat.card (AbortFiber lanes need) = abortCount lanes need :=
  (Nat.card_congr (Equiv.subtypeEquivRight
    (fun fields => abort_iff_short need (candidateList fields)))).trans
      (short_fiber_card lanes need)

/-- Saturation at the last requested digit leaves the second chunk free. -/
theorem successCount_one (lanes : Nat) (digit : Coefficient) :
    successCount (lanes + 1) [digit] =
      (pairModulus - 1) * successCount lanes [digit] +
      ((pairModulus - 1) * acceptedQuotientCount * (chunkModulus + 1) +
        if digit.val = 0 then 1 else 0) * successCount lanes [] := by
  simp only [successCount, weight, FieldPreimageRectangle.size, FieldPreimageRectangle.zeroBonus,
    ChunkClass.size, ChunkClass.HasZero, false_and, and_false, and_true,
    if_false, Nat.mul_one, Nat.one_mul, Nat.add_zero]
  ring

theorem successCount_two (lanes : Nat) (first second : Coefficient) (suffix : List Coefficient) :
    successCount (lanes + 1) (first :: second :: suffix) =
      (pairModulus - 1) * successCount lanes (first :: second :: suffix) +
      2 * (pairModulus - 1) * acceptedQuotientCount * successCount lanes (second :: suffix) +
      ((pairModulus - 1) * acceptedQuotientCount ^ 2 +
        if first.val = 0 ∧ second.val = 0 then 1 else 0) * successCount lanes suffix := by
  simp only [successCount, weight, FieldPreimageRectangle.size, FieldPreimageRectangle.zeroBonus,
    ChunkClass.size, ChunkClass.HasZero, false_and, and_false, if_false,
    Nat.mul_one, Nat.one_mul, Nat.add_zero]
  ring_nf

theorem abortCount_step (lanes need : Nat) :
    abortCount (lanes + 1) (need + 1) =
      (pairModulus - 1) * abortCount lanes (need + 1) +
      2 * (pairModulus - 1) * rejectionBucket * abortCount lanes need +
      ((pairModulus - 1) * rejectionBucket ^ 2 + 1) * abortCount lanes (need - 1) := by
  simp only [abortCount, weight, FieldPreimageRectangle.size, FieldPreimageRectangle.zeroBonus,
    ChunkClass.size, ChunkClass.HasZero, false_and, and_false, true_and, if_false, if_true,
    Nat.mul_one, Nat.one_mul, Nat.add_zero]
  ring

private theorem flatten_pairs {Value : Type*} {lanes : Nat}
    (values : Fin lanes → Fin 2 → Value) :
    List.ofFn (fun index : Fin (lanes * 2) => values index.divNat index.modNat) =
      (List.ofFn fun lane => [values lane 0, values lane 1]).flatten := by
  rw [List.ofFn_mul]
  apply congrArg List.flatten
  apply congrArg List.ofFn
  funext lane
  have components (side : Fin 2) :
      let index : Fin (lanes * 2) := ⟨lane.val * 2 + side.val, by
        have := lane.isLt
        have := side.isLt
        omega⟩
      (index.divNat, index.modNat) = (lane, side) := by
    dsimp only
    apply Prod.ext
    · apply Fin.ext
      change (lane.val * 2 + side.val) / 2 = lane.val
      have := side.isLt
      omega
    · apply Fin.ext
      change (lane.val * 2 + side.val) % 2 = side.val
      have := side.isLt
      omega
  calc
    _ = List.ofFn (values lane) := by
      apply congrArg List.ofFn
      funext side
      exact congrArg (fun pair => values pair.1 pair.2) (components side)
    _ = _ := by simp [List.ofFn_succ]

/-- The generic pair list is exactly the current 32-field/64-candidate view. -/
theorem candidateList_eq_fieldCandidates (fields : FieldShortfall.FieldWindow) :
    candidateList fields = List.ofFn (FieldShortfall.fieldCandidates fields) := by
  symm
  change List.ofFn (fun index : Fin (FieldShortfall.fieldLaneCount * 2) =>
    ((finTwoArrowEquiv Chunk).symm (candidates (fields index.divNat))) index.modNat) = _
  calc
    _ = (List.ofFn fun lane => [(candidates (fields lane)).1, (candidates (fields lane)).2]).flatten :=
      flatten_pairs (fun lane : Fin FieldShortfall.fieldLaneCount =>
        (finTwoArrowEquiv Chunk).symm (candidates (fields lane)))
    _ = _ := by simp only [candidateList, List.flatMap_def, List.map_ofFn, Function.comp_def]

end NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.FieldDecoderFiberCount
