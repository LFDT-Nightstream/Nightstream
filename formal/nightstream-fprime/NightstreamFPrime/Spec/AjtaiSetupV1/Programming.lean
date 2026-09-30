import Mathlib.Algebra.BigOperators.Field
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring
import NightstreamFPrime.Spec.AjtaiSetupV1.ReductionBias
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Binding

/-!
Owns the ideal-model programming step of the SHAKE128 setup argument
(`docs/reviews/ajtai-key-expander/SECURITY_ARGUMENT.md`, section 5).

Each key coefficient is one 32-byte SHAKE128 output chunk, read as an integer
below `2 ^ 256` and reduced modulo the Goldilocks prime (`verifierKey_eq`).
`elementInput_injective` makes the inputs distinct, and `elementLanes_length`
makes the chunks disjoint output slices. Premise P1, SHAKE128 as a random
oracle, makes the chunks independent and uniform, and independent of every
other oracle value. P1 is not proved here. `Extra` holds the other oracle
values that an attacker reads and its coins.

The programmed game samples a uniform matrix, then each chunk uniformly among
the integers with that residue. For every event of the attacker's complete
view, the two games differ by at most `2 * coefficients * r / 2 ^ 256`, with
`r = 2 ^ 256 % q` (`real_sub_programmed_le`). In the programmed game the key is
the uniform matrix, so a binding collision is a short kernel vector of a
uniform matrix, an MSIS solution (`binding_le_solver`). MSIS hardness (premise
P2) and the running time of the simulation are not modeled.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.AjtaiSetupV1.Programming

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Phi81Relation
open NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open Finset

/-! ## Independent coordinates -/

/-- Coordinate weights of hybrid `s`: `ν` on the coordinates in `s`, `μ`
elsewhere. -/
private def hybridWeight {ι α : Type} [DecidableEq ι] (μ ν : α → ℚ) (s : Finset ι) (i : ι)
    (a : α) : ℚ :=
  if i ∈ s then ν a else μ a

private def hybrid {ι α : Type} [Fintype ι] [DecidableEq ι] (μ ν : α → ℚ) (s : Finset ι)
    (values : ι → α) : ℚ :=
  ∏ i, hybridWeight μ ν s i (values i)

private theorem sum_prod_eq_prod_sum {ι α : Type} [Fintype ι] [DecidableEq ι] [Fintype α]
    (g : ι → α → ℚ) : ∑ values : ι → α, ∏ i, g i (values i) = ∏ i, ∑ a, g i a := by
  rw [prod_univ_sum, Fintype.piFinset_univ]

/-- Moving one fresh coordinate into the hybrid costs one coordinate
difference. -/
private theorem hybrid_step {ι α : Type} [Fintype ι] [DecidableEq ι] [Fintype α]
    (μ ν : α → ℚ) (μ_nonneg : ∀ a, 0 ≤ μ a) (ν_nonneg : ∀ a, 0 ≤ ν a)
    (μ_sum : ∑ a, μ a = 1) (ν_sum : ∑ a, ν a = 1) (s : Finset ι) (k : ι) (fresh : k ∉ s) :
    ∑ values : ι → α, |hybrid μ ν s values - hybrid μ ν (insert k s) values| =
      ∑ a, |μ a - ν a| := by
  have weight_nonneg : ∀ i a, 0 ≤ hybridWeight μ ν s i a := by
    intro i a
    unfold hybridWeight
    split
    · exact ν_nonneg a
    · exact μ_nonneg a
  have weight_sum : ∀ i, ∑ a, hybridWeight μ ν s i a = 1 := by
    intro i
    unfold hybridWeight
    split
    · exact ν_sum
    · exact μ_sum
  have same : ∀ i ∈ ({k}ᶜ : Finset ι), ∀ a,
      hybridWeight μ ν (insert k s) i a = hybridWeight μ ν s i a := by
    intro i member a
    have other : i ≠ k := by simpa using member
    simp only [hybridWeight, mem_insert, other, false_or]
  have pointwise : ∀ values : ι → α,
      |hybrid μ ν s values - hybrid μ ν (insert k s) values| =
        ∏ i, (if i = k then |μ (values i) - ν (values i)| else
          hybridWeight μ ν s i (values i)) := by
    intro values
    unfold hybrid
    rw [Fintype.prod_eq_mul_prod_compl k (fun i => hybridWeight μ ν s i (values i)),
      Fintype.prod_eq_mul_prod_compl k (fun i => hybridWeight μ ν (insert k s) i (values i)),
      Fintype.prod_eq_mul_prod_compl k, prod_congr rfl fun i member => same i member (values i)]
    have others : ∏ i ∈ ({k}ᶜ : Finset ι), (if i = k then |μ (values i) - ν (values i)| else
        hybridWeight μ ν s i (values i)) =
          ∏ i ∈ ({k}ᶜ : Finset ι), hybridWeight μ ν s i (values i) := by
      apply prod_congr rfl
      intro i member
      have other : i ≠ k := by simpa using member
      simp only [other, if_false]
    have atK : hybridWeight μ ν s k (values k) = μ (values k) := by
      simp only [hybridWeight, fresh, if_false]
    have atInserted : hybridWeight μ ν (insert k s) k (values k) = ν (values k) := by
      simp only [hybridWeight, mem_insert_self, if_true]
    rw [others, atK, atInserted, if_pos rfl, ← sub_mul, abs_mul,
      abs_of_nonneg (prod_nonneg fun i _ => weight_nonneg i (values i))]
  rw [sum_congr rfl fun values _ => pointwise values,
    sum_prod_eq_prod_sum (fun i a => if i = k then |μ a - ν a| else hybridWeight μ ν s i a),
    Fintype.prod_eq_mul_prod_compl k]
  have ones : ∀ i ∈ ({k}ᶜ : Finset ι),
      ∑ a, (if i = k then |μ a - ν a| else hybridWeight μ ν s i a) = 1 := by
    intro i member
    have other : i ≠ k := by simpa using member
    simp only [other, if_false]
    exact weight_sum i
  rw [prod_congr rfl ones, prod_const_one, mul_one]
  simp only [if_true]

/-- Changing each independent coordinate from weights `μ` to weights `ν`
changes the joint weights, in total absolute difference, by at most the
coordinate count times the difference of one coordinate. -/
theorem product_difference_le {ι α : Type} [Fintype ι] [DecidableEq ι] [Fintype α]
    (μ ν : α → ℚ) (μ_nonneg : ∀ a, 0 ≤ μ a) (ν_nonneg : ∀ a, 0 ≤ ν a)
    (μ_sum : ∑ a, μ a = 1) (ν_sum : ∑ a, ν a = 1) :
    ∑ values : ι → α, |∏ i, μ (values i) - ∏ i, ν (values i)| ≤
      Fintype.card ι * ∑ a, |μ a - ν a| := by
  have hybrids : ∀ s : Finset ι, ∑ values : ι → α,
      |hybrid μ ν ∅ values - hybrid μ ν s values| ≤ s.card * ∑ a, |μ a - ν a| := by
    intro s
    induction s using Finset.induction_on with
    | empty => simp
    | insert k s fresh ih =>
        calc ∑ values : ι → α, |hybrid μ ν ∅ values - hybrid μ ν (insert k s) values|
            ≤ ∑ values : ι → α, (|hybrid μ ν ∅ values - hybrid μ ν s values| +
                |hybrid μ ν s values - hybrid μ ν (insert k s) values|) :=
              sum_le_sum fun values _ => abs_sub_le _ _ _
          _ = ∑ values : ι → α, |hybrid μ ν ∅ values - hybrid μ ν s values| +
                ∑ values : ι → α, |hybrid μ ν s values - hybrid μ ν (insert k s) values| :=
              sum_add_distrib
          _ ≤ s.card * ∑ a, |μ a - ν a| + ∑ a, |μ a - ν a| := by
              rw [hybrid_step μ ν μ_nonneg ν_nonneg μ_sum ν_sum s k fresh]
              linarith
          _ = (insert k s).card * ∑ a, |μ a - ν a| := by
              rw [card_insert_of_notMem fresh]
              push_cast
              ring
  have ends := hybrids univ
  simp only [hybrid, hybridWeight, notMem_empty, if_false, mem_univ, if_true, card_univ] at ends
  exact ends

/-! ## Chunks and residues -/

/-- One 32-byte output chunk read as a little-endian integer. -/
abbrev Chunk := Fin (2 ^ 256)

/-- The key coefficient of one chunk. -/
def residue (chunk : Chunk) : F :=
  ⟨chunk.val % goldilocksModulus, Nat.mod_lt _ (by decide)⟩

/-- The chunks with residue `value`. -/
def fiber (value : F) : Finset Chunk :=
  univ.filter fun chunk => residue chunk = value

/-- Probability of `value` when a uniform chunk is reduced. -/
noncomputable def reducedWeight (value : F) : ℚ :=
  ((fiber value).card : ℚ) / ((2 ^ 256 : ℕ) : ℚ)

private def small (value : F) : Chunk := ⟨value.val, Nat.lt_trans value.isLt (by decide)⟩

private theorem residue_small (value : F) : residue (small value) = value := by
  apply Fin.ext
  exact Nat.mod_eq_of_lt value.isLt

theorem fiber_nonempty (value : F) : (fiber value).Nonempty :=
  ⟨small value, by simp [fiber, residue_small]⟩

/-- A natural number below `q` whose residue lies in `values`. -/
private def residueEvent (values : Finset F) (natural : Nat) : Prop :=
  ∃ bounded : natural < goldilocksModulus, (⟨natural, bounded⟩ : F) ∈ values

private instance (values : Finset F) : DecidablePred (residueEvent values) := by
  intro natural
  unfold residueEvent
  infer_instance

private theorem card_filter_val {count : Nat} (event : Nat → Prop) [DecidablePred event] :
    (univ.filter fun index : Fin count => event index.val).card =
      Nat.count event count := by
  rw [Nat.count_eq_card_filter_range]
  apply card_bij (fun index _ => index.val)
  · intro index member
    simp only [mem_filter, mem_univ, true_and] at member
    simp [member, index.isLt]
  · intro left _ right _ same
    exact Fin.ext same
  · intro natural member
    simp only [mem_filter, mem_range] at member
    exact ⟨⟨natural, member.1⟩, by simp [member.2], rfl⟩

private theorem residue_mem_iff (values : Finset F) (chunk : Chunk) :
    residue chunk ∈ values ↔ residueEvent values (chunk.val % goldilocksModulus) := by
  constructor
  · intro member
    exact ⟨Nat.mod_lt _ (by decide), member⟩
  · rintro ⟨_, member⟩
    exact member

/-- The reduced weights of a residue set are the frequency of that event
under reduction of a uniform 256-bit integer. -/
private theorem sum_reducedWeight (values : Finset F) :
    ∑ value ∈ values, reducedWeight value =
      ReductionBias.frequency (2 ^ 256) goldilocksModulus (residueEvent values) := by
  have cards : ∑ value ∈ values, ((fiber value).card : ℚ) =
      ((univ.filter fun chunk : Chunk => residue chunk ∈ values).card : ℚ) := by
    rw [card_eq_sum_card_fiberwise (s := univ.filter fun chunk : Chunk => residue chunk ∈ values)
      (f := residue) (t := values) (fun chunk member => (mem_filter.mp member).2)]
    push_cast
    apply sum_congr rfl
    intro value member
    have same : fiber value = (univ.filter fun chunk : Chunk => residue chunk ∈ values).filter
        fun chunk => residue chunk = value := by
      ext chunk
      simp only [fiber, mem_filter, mem_univ, true_and]
      constructor
      · rintro rfl
        exact ⟨member, rfl⟩
      · rintro ⟨_, equal⟩
        exact equal
    rw [same]
  have counted : (univ.filter fun chunk : Chunk => residue chunk ∈ values).card =
      Nat.count (fun natural => residueEvent values (natural % goldilocksModulus)) (2 ^ 256) := by
    rw [filter_congr fun chunk _ => residue_mem_iff values chunk]
    exact card_filter_val (fun natural => residueEvent values (natural % goldilocksModulus))
  unfold reducedWeight ReductionBias.frequency
  rw [← sum_div, cards, counted]

/-- The uniform residue weights of a residue set are its frequency among the
residues. -/
private theorem sum_uniformWeight (values : Finset F) :
    ∑ _value ∈ values, (1 / goldilocksModulus : ℚ) =
      ReductionBias.frequency goldilocksModulus goldilocksModulus (residueEvent values) := by
  have counted : Nat.count (fun natural => residueEvent values (natural % goldilocksModulus))
      goldilocksModulus = values.card := by
    rw [Nat.count_eq_card_filter_range]
    apply card_bij (fun natural member => (⟨natural, mem_range.mp (mem_filter.mp member).1⟩ : F))
    · intro natural member
      obtain ⟨bounded, reduced, inside⟩ := mem_filter.mp member
      have same : (⟨natural % goldilocksModulus, reduced⟩ : F) = ⟨natural, mem_range.mp bounded⟩ :=
        Fin.ext (Nat.mod_eq_of_lt (mem_range.mp bounded))
      rwa [same] at inside
    · intro left _ right _ same
      exact congrArg Fin.val same
    · intro value member
      refine ⟨value.val, ?_, rfl⟩
      have same : (⟨value.val % goldilocksModulus, Nat.mod_lt _ (by decide)⟩ : F) = value :=
        Fin.ext (Nat.mod_eq_of_lt value.isLt)
      refine mem_filter.mpr ⟨mem_range.mpr value.isLt, Nat.mod_lt _ (by decide), ?_⟩
      rw [same]
      exact member
  simp only [ReductionBias.frequency, sum_const, nsmul_eq_mul, counted]
  ring

theorem reducedWeight_nonneg (value : F) : 0 ≤ reducedWeight value := by
  unfold reducedWeight
  positivity

theorem reducedWeight_sum : ∑ value, reducedWeight value = 1 := by
  have partition := card_eq_sum_card_fiberwise (s := (univ : Finset Chunk))
    (t := (univ : Finset F)) (f := residue) (fun _ _ => mem_univ _)
  rw [card_univ, Fintype.card_fin] at partition
  have cards : ∑ value, ((fiber value).card : ℚ) = ((2 ^ 256 : ℕ) : ℚ) := by
    exact_mod_cast partition.symm
  unfold reducedWeight
  rw [← sum_div, cards, div_self (by positivity)]

/-- Reducing a uniform 256-bit integer instead of sampling a uniform residue
changes the residue weights, in total absolute difference, by at most
`2 * r / 2 ^ 256`. The bound for each event is
`ReductionBias.wide_frequency_error_le`. -/
theorem reducedWeight_difference_le :
    ∑ value, |reducedWeight value - 1 / goldilocksModulus| ≤ 2 * 4294967295 / 2 ^ 256 := by
  rw [← sum_filter_add_sum_filter_not univ
    (fun value : F => (1 / goldilocksModulus : ℚ) ≤ reducedWeight value)]
  have upper : ∑ value ∈ univ.filter (fun value : F =>
      (1 / goldilocksModulus : ℚ) ≤ reducedWeight value),
      |reducedWeight value - 1 / goldilocksModulus| ≤ 4294967295 / 2 ^ 256 := by
    rw [sum_congr rfl fun value member =>
        abs_of_nonneg (sub_nonneg.mpr (mem_filter.mp member).2),
      sum_sub_distrib, sum_reducedWeight, sum_uniformWeight]
    exact (le_abs_self _).trans (ReductionBias.wide_frequency_error_le _)
  have lower : ∑ value ∈ univ.filter (fun value : F =>
      ¬ (1 / goldilocksModulus : ℚ) ≤ reducedWeight value),
      |reducedWeight value - 1 / goldilocksModulus| ≤ 4294967295 / 2 ^ 256 := by
    rw [sum_congr rfl fun value member => (abs_sub_comm _ _).trans
        (abs_of_nonneg (sub_nonneg.mpr (le_of_lt (not_le.mp (mem_filter.mp member).2)))),
      sum_sub_distrib, sum_reducedWeight, sum_uniformWeight]
    exact (le_abs_self _).trans
      ((abs_sub_comm _ _).le.trans (ReductionBias.wide_frequency_error_le _))
  linarith

/-! ## The real and programmed games -/

section Games

variable {ι Extra : Type} [Fintype ι] [DecidableEq ι] [Fintype Extra] [Nonempty Extra]

/-- The key coefficients that the chunks define. -/
def residues (chunks : ι → Chunk) : ι → F := fun index => residue (chunks index)

/-- The chunk vectors whose residues are `matrix`. -/
def preimages (matrix : ι → F) : Finset (ι → Chunk) :=
  Fintype.piFinset fun index => fiber (matrix index)

theorem mem_preimages {matrix : ι → F} {chunks : ι → Chunk} :
    chunks ∈ preimages matrix ↔ residues chunks = matrix := by
  simp only [preimages, Fintype.mem_piFinset, fiber, mem_filter, mem_univ, true_and]
  exact ⟨fun each => funext each, fun same index => congrFun same index⟩

/-- The attacker views of the programmed game for one matrix. -/
def views (matrix : ι → F) : Finset ((ι → Chunk) × Extra) :=
  preimages matrix ×ˢ univ

/-- Game 1: independent uniform chunks and uniform extra values. -/
noncomputable def real (event : (ι → Chunk) × Extra → Prop) [DecidablePred event] : ℚ :=
  ((univ.filter event).card : ℚ) / Fintype.card ((ι → Chunk) × Extra)

/-- Probability of `event` when the chunks are uniform preimages of `matrix`. -/
noncomputable def programmedFor (event : (ι → Chunk) × Extra → Prop) [DecidablePred event]
    (matrix : ι → F) : ℚ :=
  (((views matrix).filter event).card : ℚ) / (views (Extra := Extra) matrix).card

/-- Game 2: a uniform matrix, then uniform preimage chunks. -/
noncomputable def programmed (event : (ι → Chunk) × Extra → Prop) [DecidablePred event] : ℚ :=
  ∑ matrix : ι → F, programmedFor event matrix / Fintype.card (ι → F)

omit [Nonempty Extra] in
private theorem views_card (matrix : ι → F) :
    ((views (Extra := Extra) matrix).card : ℚ) =
      (∏ index, ((fiber (matrix index)).card : ℚ)) * Fintype.card Extra := by
  unfold views preimages
  rw [card_product, Fintype.card_piFinset, card_univ]
  push_cast
  rfl

private theorem views_card_pos (matrix : ι → F) :
    (0 : ℚ) < (views (Extra := Extra) matrix).card := by
  rw [views_card]
  apply mul_pos
  · exact prod_pos fun index _ => by exact_mod_cast (fiber_nonempty (matrix index)).card_pos
  · exact_mod_cast Fintype.card_pos

private theorem programmedFor_mem (event : (ι → Chunk) × Extra → Prop) [DecidablePred event]
    (matrix : ι → F) : 0 ≤ programmedFor event matrix ∧ programmedFor event matrix ≤ 1 := by
  have positive := views_card_pos (Extra := Extra) matrix
  constructor
  · unfold programmedFor
    positivity
  · unfold programmedFor
    rw [div_le_one positive]
    exact_mod_cast card_filter_le _ _

omit [Nonempty Extra] in
/-- The views of each matrix partition all views. -/
private theorem card_filter_eq_sum (event : (ι → Chunk) × Extra → Prop) [DecidablePred event] :
    ((univ.filter event).card : ℚ) =
      ∑ matrix : ι → F, (((views matrix).filter event).card : ℚ) := by
  rw [card_eq_sum_card_fiberwise (f := fun view : (ι → Chunk) × Extra => residues view.1)
    (t := univ) (fun _ _ => mem_univ _)]
  push_cast
  apply sum_congr rfl
  intro matrix _
  congr 2
  ext view
  simp only [views, mem_filter, mem_univ, true_and, mem_product, mem_preimages, and_true]
  tauto

private theorem real_eq (event : (ι → Chunk) × Extra → Prop) [DecidablePred event] :
    real event = ∑ matrix : ι → F,
      (∏ index, reducedWeight (matrix index)) * programmedFor event matrix := by
  unfold real
  rw [card_filter_eq_sum, sum_div]
  apply sum_congr rfl
  intro matrix _
  have fibers : (0 : ℚ) < ∏ index, ((fiber (matrix index)).card : ℚ) :=
    prod_pos fun index _ => by exact_mod_cast (fiber_nonempty (matrix index)).card_pos
  have extra : (0 : ℚ) < Fintype.card Extra := by exact_mod_cast Fintype.card_pos
  unfold programmedFor
  rw [views_card]
  simp only [reducedWeight, prod_div_distrib, prod_const, card_univ, Fintype.card_prod,
    Fintype.card_fun, Fintype.card_fin]
  push_cast
  field_simp

omit [Nonempty Extra] in
private theorem programmed_eq (event : (ι → Chunk) × Extra → Prop) [DecidablePred event] :
    programmed event = ∑ matrix : ι → F,
      (∏ _index : ι, (1 / goldilocksModulus : ℚ)) * programmedFor event matrix := by
  unfold programmed
  apply sum_congr rfl
  intro matrix _
  simp only [prod_const, card_univ, Fintype.card_fun, Fintype.card_fin]
  push_cast
  rw [one_div, inv_pow, div_eq_mul_inv, mul_comm]

/-- Programming the chunks changes the probability of any event of the
attacker's complete view by at most `2 * coefficients * r / 2 ^ 256`. -/
theorem real_sub_programmed_le (event : (ι → Chunk) × Extra → Prop) [DecidablePred event] :
    |real event - programmed event| ≤ Fintype.card ι * (2 * 4294967295 / 2 ^ 256) := by
  rw [real_eq, programmed_eq, ← sum_sub_distrib]
  calc |∑ matrix : ι → F, ((∏ index, reducedWeight (matrix index)) * programmedFor event matrix -
          (∏ _index : ι, (1 / goldilocksModulus : ℚ)) * programmedFor event matrix)|
      ≤ ∑ matrix : ι → F, |∏ index, reducedWeight (matrix index) -
          ∏ _index : ι, (1 / goldilocksModulus : ℚ)| := by
        refine (abs_sum_le_sum_abs _ _).trans (sum_le_sum fun matrix _ => ?_)
        have bounds := programmedFor_mem event matrix
        rw [← sub_mul, abs_mul, abs_of_nonneg bounds.1]
        exact mul_le_of_le_one_right (abs_nonneg _) bounds.2
    _ ≤ Fintype.card ι * ∑ value, |reducedWeight value - 1 / goldilocksModulus| :=
        product_difference_le reducedWeight (fun _ => (1 / goldilocksModulus : ℚ))
          reducedWeight_nonneg (fun _ => by positivity) reducedWeight_sum (by
            rw [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
            exact mul_one_div_cancel (Nat.cast_ne_zero.mpr (by decide)))
    _ ≤ Fintype.card ι * (2 * 4294967295 / 2 ^ 256) :=
        mul_le_mul_of_nonneg_left reducedWeight_difference_le (Nat.cast_nonneg _)

end Games

/-! ## The setup chunks and binding -/

/-- Key coordinates `(row, block, lane)`. -/
abbrev KeyIndex (rows blocks : Nat) := Fin rows × Fin blocks × Fin ringDegree

/-- The key whose coefficient at `(row, block, lane)` is `matrix (row, block, lane)`. -/
def keyOf {rows blocks : Nat} (matrix : KeyIndex rows blocks → F) :
    Fin rows → Fin blocks → RingF :=
  fun row block lane => matrix (row, block, lane)

/-- The SHAKE128 chunk of key coordinate `(row, block, lane)`. -/
def setupChunks {rows blocks : Nat} (seed : List Nat) (index : KeyIndex rows blocks) : Chunk :=
  ⟨laneWord (elementLanes seed index.1.val index.2.1.val) index.2.2.val, laneWord_lt _ _⟩

/-- The verifier key is the residue key of its SHAKE128 chunks. -/
theorem verifierKey_eq {rows blocks : Nat} (setup : Setup rows blocks) :
    setup.verifierKey = keyOf (residues (setupChunks setup.seed.bytes)) := by
  funext row block lane
  rfl

/-- Two different openings below `bound` with equal commitments. -/
def IsCollision {shape : Phi81Relation.Shape} {rows : Nat} (bound : Nat)
    (key : Commitment.Key shape rows)
    (openings : Phi81Relation.Assignment shape × Phi81Relation.Assignment shape) : Prop :=
  Commitment.commit key openings.1 = Commitment.commit key openings.2 ∧
    assignmentNormBounded bound openings.1 ∧ assignmentNormBounded bound openings.2 ∧
    openings.1 ≠ openings.2

/-- A collision is a short kernel vector of the same key. -/
def IsCollision.shortKernel {shape : Phi81Relation.Shape} {rows bound : Nat}
    {key : Commitment.Key shape rows}
    {openings : Phi81Relation.Assignment shape × Phi81Relation.Assignment shape}
    (collision : IsCollision bound key openings) :
    Binding.ShortKernelVector key (2 * bound) :=
  Binding.bindingCollision_to_shortKernel key (Commitment.commit key openings.1)
    { leftOpening := openings.1
      rightOpening := openings.2
      leftCommits := rfl
      rightCommits := collision.1.symm
      leftNorm := collision.2.1
      rightNorm := collision.2.2.1
      different := collision.2.2.2 }

section Binding

open Classical

variable {shape : Phi81Relation.Shape} {rows : Nat} {Extra : Type} [Fintype Extra] [Nonempty Extra]

/-- Success probability of the MSIS solver built from `attack`: for a uniform
matrix it samples uniform preimage chunks and extra values, runs `attack`,
and converts a collision with `IsCollision.shortKernel`. -/
noncomputable def solverSuccess (bound : Nat)
    (attack : (KeyIndex rows (Phi81ColumnLayout.blockCount shape.carrierWidth) → Chunk) × Extra →
      Phi81Relation.Assignment shape × Phi81Relation.Assignment shape) : ℚ :=
  ∑ matrix : KeyIndex rows (Phi81ColumnLayout.blockCount shape.carrierWidth) → F,
    programmedFor (fun view => IsCollision bound (keyOf matrix) (attack view)) matrix /
      Fintype.card (KeyIndex rows (Phi81ColumnLayout.blockCount shape.carrierWidth) → F)

/-- Binding in the ideal model: an attacker that sees every chunk finds a
collision for the chunk key at most as often as the MSIS solver built from it
succeeds on a uniform matrix, plus `2 * coefficients * r / 2 ^ 256`. -/
theorem binding_le_solver (bound : Nat)
    (attack : (KeyIndex rows (Phi81ColumnLayout.blockCount shape.carrierWidth) → Chunk) × Extra →
      Phi81Relation.Assignment shape × Phi81Relation.Assignment shape) :
    real (fun view => IsCollision bound (keyOf (residues view.1)) (attack view)) ≤
      solverSuccess bound attack +
        Fintype.card (KeyIndex rows (Phi81ColumnLayout.blockCount shape.carrierWidth)) *
          (2 * 4294967295 / 2 ^ 256) := by
  have difference := real_sub_programmed_le
    (fun view => IsCollision bound (keyOf (residues view.1)) (attack view))
  have same : programmed (fun view => IsCollision bound (keyOf (residues view.1)) (attack view)) =
      solverSuccess bound attack := by
    unfold programmed solverSuccess programmedFor
    apply sum_congr rfl
    intro matrix _
    rw [filter_congr (s := views matrix)
      (p := fun view => IsCollision bound (keyOf (residues view.1)) (attack view))
      (q := fun view => IsCollision bound (keyOf matrix) (attack view))
      (fun view member => by rw [mem_preimages.mp (mem_product.mp member).1])]
  rw [same] at difference
  linarith [le_abs_self (real (fun view => IsCollision bound (keyOf (residues view.1))
    (attack view)) - solverSuccess bound attack)]

end Binding

end NightstreamFPrime.Spec.AjtaiSetupV1.Programming
