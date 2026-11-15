import Mathlib.Algebra.BigOperators.Field
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Expect
import Mathlib.Data.Rat.BigOperators
import Mathlib.Basic.Real.Basic
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring
import NightstreamFPrime.Spec.AjtaiSetupV1.Rejection
import NightstreamFPrime.Spec.RingRelation.PiRLCAlgebra.Binding
import NightstreamFPrime.Spec.AjtaiSetupV1

/-! Assumes SHAKE128 is a random oracle (premise P1); P1 is not proved here.

The setup reads one stream of SHAKE128 words per key element (`Rejection.Stream`) and keeps the
words below `q` (`Rejection.element`). `expect_le_programmed` replaces an average over uniform
streams by an average over a uniform key matrix and a uniform preimage of it, at the cost of
`2 · Rejection.fallbackBound` for each key element (`element_bias_le`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.AjtaiSetupV1.Programming

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.RingRelation
open NightstreamFPrime.Spec.RingRelation.PiRLCAlgebra
open NightstreamFPrime.Spec.RingRelation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open Finset

private def hybridWeight {ι α : Type} [DecidableEq ι] (μ ν : α → ℚ) (s : Finset ι) (i : ι)
    (a : α) : ℚ :=
  if i ∈ s then ν a else μ a

private def hybrid {ι α : Type} [Fintype ι] [DecidableEq ι] (μ ν : α → ℚ) (s : Finset ι)
    (values : ι → α) : ℚ :=
  ∏ i, hybridWeight μ ν s i (values i)

private theorem sum_prod_eq_prod_sum {ι α : Type} [Fintype ι] [DecidableEq ι] [Fintype α]
    (g : ι → α → ℚ) : ∑ values : ι → α, ∏ i, g i (values i) = ∏ i, ∑ a, g i a := by
  rw [prod_univ_sum, Fintype.piFinset_univ]

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
      simp only [other, ite_false]
    have atK : hybridWeight μ ν s k (values k) = μ (values k) := by
      simp only [hybridWeight, fresh, ite_false]
    have atInserted : hybridWeight μ ν (insert k s) k (values k) = ν (values k) := by
      simp only [hybridWeight, mem_insert_self, ite_true]
    rw [others, atK, atInserted, ite_eq_left rfl, ← sub_mul, abs_mul,
      abs_of_nonneg (prod_nonneg fun i _ => weight_nonneg i (values i))]
  rw [sum_congr rfl fun values _ => pointwise values,
    sum_prod_eq_prod_sum (fun i a => if i = k then |μ a - ν a| else hybridWeight μ ν s i a),
    Fintype.prod_eq_mul_prod_compl k]
  have ones : ∀ i ∈ ({k}ᶜ : Finset ι),
      ∑ a, (if i = k then |μ a - ν a| else hybridWeight μ ν s i a) = 1 := by
    intro i member
    have other : i ≠ k := by simpa using member
    simp only [other, ite_false]
    exact weight_sum i
  rw [prod_congr rfl ones, prod_const_one, mul_one]
  simp only [ite_true]

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
  simp only [hybrid, hybridWeight, notMem_empty, ite_false, mem_univ, ite_true, card_univ] at ends
  exact ends

section Sampler

variable {Chunk Out : Type} [Fintype Chunk] [Fintype Out] [DecidableEq Out]
  (sample : Chunk → Out)

def fiber (value : Out) : Finset Chunk :=
  univ.filter fun chunk => sample chunk = value

noncomputable def sampledWeight (value : Out) : ℚ :=
  ((fiber sample value).card : ℚ) / Fintype.card Chunk

omit [Fintype Out] in
theorem sampledWeight_nonneg (value : Out) : 0 ≤ sampledWeight sample value := by
  unfold sampledWeight
  positivity

theorem sampledWeight_sum [Nonempty Chunk] : ∑ value, sampledWeight sample value = 1 := by
  have partition := card_eq_sum_card_fiberwise (s := (univ : Finset Chunk))
    (t := (univ : Finset Out)) (f := sample) (fun _ _ => mem_univ _)
  rw [card_univ] at partition
  have cards : ∑ value, ((fiber sample value).card : ℚ) = (Fintype.card Chunk : ℚ) := by
    exact_mod_cast partition.symm
  unfold sampledWeight
  rw [← sum_div, cards, div_self (Nat.cast_ne_zero.mpr Fintype.card_ne_zero)]

omit [Fintype Out] in
theorem onto_of_uniform_good (Bad : Chunk → Prop) [DecidablePred Bad] (good : ℕ)
    (uniform : ∀ value,
      (univ.filter fun chunk => ¬ Bad chunk ∧ sample chunk = value).card = good)
    (positive : 0 < good) (value : Out) : (fiber sample value).Nonempty := by
  rw [← uniform value] at positive
  obtain ⟨chunk, member⟩ := card_pos.mp positive
  exact ⟨chunk, mem_filter.mpr ⟨mem_univ _, (mem_filter.mp member).2.2⟩⟩

/-- When each output is the sample of the same number of chunks outside `Bad`, the law of the
sample of a uniform chunk and the uniform law differ, in total absolute difference, by at most
twice the chance of `Bad`. -/
theorem bias_le_of_uniform_good [Nonempty Chunk] [Nonempty Out] (Bad : Chunk → Prop)
    [DecidablePred Bad] (good : ℕ)
    (uniform : ∀ value,
      (univ.filter fun chunk => ¬ Bad chunk ∧ sample chunk = value).card = good) :
    ∑ value, |sampledWeight sample value - 1 / (Fintype.card Out : ℚ)| ≤
      2 * ((univ.filter Bad).card / (Fintype.card Chunk : ℚ)) := by
  have chunksPositive : (0 : ℚ) < Fintype.card Chunk := by exact_mod_cast Fintype.card_pos
  have outputsPositive : (0 : ℚ) < Fintype.card Out := by exact_mod_cast Fintype.card_pos
  have split : ∀ value : Out, ((fiber sample value).card : ℚ) =
      good + ((fiber sample value).filter Bad).card := by
    intro value
    have parts := card_filter_add_card_filter_not (s := fiber sample value) Bad
    have kept : ((fiber sample value).filter fun chunk => ¬ Bad chunk) =
        univ.filter fun chunk => ¬ Bad chunk ∧ sample chunk = value := by
      ext chunk
      simp only [fiber, mem_filter, mem_univ, true_and, and_comm]
    rw [kept, uniform value] at parts
    have cast := congrArg (Nat.cast (R := ℚ)) parts
    push_cast at cast
    linarith
  have bads : ∑ value : Out, (((fiber sample value).filter Bad).card : ℚ) =
      (univ.filter Bad).card := by
    have fibers := card_eq_sum_card_fiberwise (s := univ.filter Bad)
      (t := (univ : Finset Out)) (f := sample) (fun _ _ => mem_univ _)
    rw [fibers]
    push_cast
    refine sum_congr rfl fun value _ => ?_
    rw [fiber, filter_comm]
  have total : (Fintype.card Out : ℚ) * good + (univ.filter Bad).card = Fintype.card Chunk := by
    have partition := card_eq_sum_card_fiberwise (s := (univ : Finset Chunk))
      (t := (univ : Finset Out)) (f := sample) (fun _ _ => mem_univ _)
    rw [card_univ] at partition
    have cards : ∑ value : Out, ((fiber sample value).card : ℚ) = Fintype.card Chunk := by
      exact_mod_cast partition.symm
    rw [sum_congr rfl fun value _ => split value, sum_add_distrib, bads, sum_const,
      card_univ, nsmul_eq_mul] at cards
    exact cards
  have each : ∀ value : Out,
      |sampledWeight sample value - 1 / (Fintype.card Out : ℚ)| ≤
        ((fiber sample value).filter Bad).card / (Fintype.card Chunk : ℚ) +
          (univ.filter Bad).card / ((Fintype.card Out : ℚ) * Fintype.card Chunk) := by
    intro value
    have difference : sampledWeight sample value - 1 / (Fintype.card Out : ℚ) =
        ((fiber sample value).filter Bad).card / (Fintype.card Chunk : ℚ) -
          (univ.filter Bad).card / ((Fintype.card Out : ℚ) * Fintype.card Chunk) := by
      rw [sampledWeight, split value]
      field_simp
      linarith
    have first : (0 : ℚ) ≤
        ((fiber sample value).filter Bad).card / (Fintype.card Chunk : ℚ) := by
      positivity
    have second : (0 : ℚ) ≤
        (univ.filter Bad).card / ((Fintype.card Out : ℚ) * Fintype.card Chunk) := by
      positivity
    rw [difference, abs_le]
    constructor <;> linarith
  calc ∑ value : Out, |sampledWeight sample value - 1 / (Fintype.card Out : ℚ)|
      ≤ ∑ value : Out,
          (((fiber sample value).filter Bad).card / (Fintype.card Chunk : ℚ) +
            (univ.filter Bad).card / ((Fintype.card Out : ℚ) * Fintype.card Chunk)) :=
        sum_le_sum fun value _ => each value
    _ = 2 * ((univ.filter Bad).card / (Fintype.card Chunk : ℚ)) := by
        rw [sum_add_distrib, ← sum_div, bads, sum_const, card_univ, nsmul_eq_mul]
        field_simp
        ring

variable {ι Extra : Type} [Fintype ι] [DecidableEq ι] [Fintype Extra] [Nonempty Extra]

def residues (chunks : ι → Chunk) : ι → Out := fun index => sample (chunks index)

def preimages (matrix : ι → Out) : Finset (ι → Chunk) :=
  Fintype.piFinset fun index => fiber sample (matrix index)

omit [Fintype Out] in
theorem mem_preimages {matrix : ι → Out} {chunks : ι → Chunk} :
    chunks ∈ preimages sample matrix ↔ residues sample chunks = matrix := by
  simp only [preimages, Fintype.mem_piFinset, fiber, mem_filter, mem_univ, true_and]
  exact ⟨fun each => funext each, fun same index => congrFun same index⟩

def views (matrix : ι → Out) : Finset ((ι → Chunk) × Extra) :=
  preimages sample matrix ×ˢ univ

noncomputable def real (event : (ι → Chunk) × Extra → Prop) [DecidablePred event] : ℚ :=
  ((univ.filter event).card : ℚ) / Fintype.card ((ι → Chunk) × Extra)

noncomputable def programmedFor (event : (ι → Chunk) × Extra → Prop) [DecidablePred event]
    (matrix : ι → Out) : ℚ :=
  (((views sample matrix).filter event).card : ℚ) / (views (Extra := Extra) sample matrix).card

noncomputable def programmed (event : (ι → Chunk) × Extra → Prop) [DecidablePred event] : ℚ :=
  ∑ matrix : ι → Out, programmedFor sample event matrix / Fintype.card (ι → Out)

omit [Fintype Out] [Nonempty Extra] in
private theorem views_card (matrix : ι → Out) :
    ((views (Extra := Extra) sample matrix).card : ℚ) =
      (∏ index, ((fiber sample (matrix index)).card : ℚ)) * Fintype.card Extra := by
  unfold views preimages
  rw [card_product, Fintype.card_piFinset, card_univ]
  push_cast
  rfl

omit [Fintype Out] in
private theorem views_card_pos (onto : ∀ value, (fiber sample value).Nonempty)
    (matrix : ι → Out) : (0 : ℚ) < (views (Extra := Extra) sample matrix).card := by
  rw [views_card]
  apply mul_pos
  · exact prod_pos fun index _ => by exact_mod_cast (onto (matrix index)).card_pos
  · exact_mod_cast Fintype.card_pos

omit [Fintype Out] in
private theorem programmedFor_mem (onto : ∀ value, (fiber sample value).Nonempty)
    (event : (ι → Chunk) × Extra → Prop) [DecidablePred event] (matrix : ι → Out) :
    0 ≤ programmedFor sample event matrix ∧ programmedFor sample event matrix ≤ 1 := by
  have positive := views_card_pos (Extra := Extra) sample onto matrix
  constructor
  · unfold programmedFor
    positivity
  · unfold programmedFor
    rw [div_le_one positive]
    exact_mod_cast card_filter_le _ _

omit [Nonempty Extra] in
private theorem card_filter_eq_sum (event : (ι → Chunk) × Extra → Prop) [DecidablePred event] :
    ((univ.filter event).card : ℚ) =
      ∑ matrix : ι → Out, (((views sample matrix).filter event).card : ℚ) := by
  rw [card_eq_sum_card_fiberwise
    (f := fun view : (ι → Chunk) × Extra => residues sample view.1)
    (t := univ) (fun _ _ => mem_univ _)]
  push_cast
  apply sum_congr rfl
  intro matrix _
  congr 2
  ext view
  simp only [views, mem_filter, mem_univ, true_and, mem_product, mem_preimages, and_true]
  tauto

private theorem real_eq [Nonempty Chunk] (onto : ∀ value, (fiber sample value).Nonempty)
    (event : (ι → Chunk) × Extra → Prop) [DecidablePred event] :
    real event = ∑ matrix : ι → Out,
      (∏ index, sampledWeight sample (matrix index)) * programmedFor sample event matrix := by
  unfold real
  rw [card_filter_eq_sum sample, sum_div]
  apply sum_congr rfl
  intro matrix _
  have fibers : (0 : ℚ) < ∏ index, ((fiber sample (matrix index)).card : ℚ) :=
    prod_pos fun index _ => by exact_mod_cast (onto (matrix index)).card_pos
  have extra : (0 : ℚ) < Fintype.card Extra := by exact_mod_cast Fintype.card_pos
  have chunks : (0 : ℚ) < Fintype.card Chunk := by exact_mod_cast Fintype.card_pos
  unfold programmedFor
  rw [views_card]
  simp only [sampledWeight, prod_div_distrib, prod_const, card_univ, Fintype.card_prod,
    Fintype.card_fun]
  push_cast
  field_simp

omit [Nonempty Extra] in
private theorem programmed_eq (event : (ι → Chunk) × Extra → Prop) [DecidablePred event] :
    programmed sample event = ∑ matrix : ι → Out,
      (∏ _index : ι, (1 / (Fintype.card Out : ℚ))) * programmedFor sample event matrix := by
  unfold programmed
  apply sum_congr rfl
  intro matrix _
  simp only [prod_const, card_univ, Fintype.card_fun]
  push_cast
  rw [one_div, inv_pow, div_eq_mul_inv, mul_comm]

omit [DecidableEq Out] in
private theorem uniform_sum [Nonempty Out] :
    ∑ _value : Out, (1 / (Fintype.card Out : ℚ)) = 1 := by
  rw [sum_const, card_univ, nsmul_eq_mul]
  exact mul_one_div_cancel (Nat.cast_ne_zero.mpr Fintype.card_ne_zero)

theorem real_sub_programmed_le [Nonempty Chunk] [Nonempty Out] {error : ℚ}
    (onto : ∀ value, (fiber sample value).Nonempty)
    (bias : ∑ value, |sampledWeight sample value - 1 / (Fintype.card Out : ℚ)| ≤ error)
    (event : (ι → Chunk) × Extra → Prop) [DecidablePred event] :
    |real event - programmed sample event| ≤ Fintype.card ι * error := by
  rw [real_eq sample onto, programmed_eq, ← sum_sub_distrib]
  calc |∑ matrix : ι → Out,
          ((∏ index, sampledWeight sample (matrix index)) * programmedFor sample event matrix -
          (∏ _index : ι, (1 / (Fintype.card Out : ℚ))) * programmedFor sample event matrix)|
      ≤ ∑ matrix : ι → Out, |∏ index, sampledWeight sample (matrix index) -
          ∏ _index : ι, (1 / (Fintype.card Out : ℚ))| := by
        refine (abs_sum_le_sum_abs _ _).trans (sum_le_sum fun matrix _ => ?_)
        have bounds := programmedFor_mem sample onto event matrix
        rw [← sub_mul, abs_mul, abs_of_nonneg bounds.1]
        exact mul_le_of_le_one_right (abs_nonneg _) bounds.2
    _ ≤ Fintype.card ι * ∑ value, |sampledWeight sample value - 1 / (Fintype.card Out : ℚ)| :=
        product_difference_le (sampledWeight sample) (fun _ => (1 / (Fintype.card Out : ℚ)))
          (sampledWeight_nonneg sample) (fun _ => by positivity) (sampledWeight_sum sample)
          uniform_sum
    _ ≤ Fintype.card ι * error := mul_le_mul_of_nonneg_left bias (Nat.cast_nonneg _)

end Sampler

open scoped BigOperators in
theorem expect_le_programmed {ι Chunk Out : Type} [Fintype ι] [DecidableEq ι] [Fintype Chunk]
    [Nonempty Chunk] [Fintype Out] [Nonempty Out] [DecidableEq Out] (sample : Chunk → Out)
    {error : ℚ} (onto : ∀ value, (fiber sample value).Nonempty)
    (bias : ∑ value, |sampledWeight sample value - 1 / (Fintype.card Out : ℚ)| ≤ error)
    (value : (ι → Chunk) → ℝ)
    (nonnegative : ∀ chunks, 0 ≤ value chunks) (le_one : ∀ chunks, value chunks ≤ 1) :
    𝔼 chunks, value chunks ≤
      (𝔼 matrix : ι → Out, 𝔼 chunks ∈ preimages sample matrix, value chunks) +
        Fintype.card ι * (error : ℝ) := by
  set average : (ι → Out) → ℝ := fun matrix => 𝔼 chunks ∈ preimages sample matrix, value chunks
  have preimagesNonempty (matrix : ι → Out) : (preimages sample matrix).Nonempty := by
    unfold preimages
    exact Fintype.piFinset_nonempty.mpr fun index => onto (matrix index)
  have average_nonnegative (matrix : ι → Out) : 0 ≤ average matrix :=
    Finset.expect_nonneg fun chunks _ => nonnegative chunks
  have average_le_one (matrix : ι → Out) : average matrix ≤ 1 := by
    calc average matrix ≤ 𝔼 _chunks ∈ preimages sample matrix, (1 : ℝ) :=
          Finset.expect_le_expect fun chunks _ => le_one chunks
      _ = 1 := Finset.expect_const (preimagesNonempty matrix) 1
  have preimagesCard (matrix : ι → Out) :
      ((preimages sample matrix).card : ℝ) =
        ∏ index, ((fiber sample (matrix index)).card : ℝ) := by
    unfold preimages
    rw [Fintype.card_piFinset]
    push_cast
    rfl
  have real : 𝔼 chunks, value chunks =
      ∑ matrix : ι → Out,
        (∏ index, (sampledWeight sample (matrix index) : ℝ)) * average matrix := by
    rw [Finset.expect_eq_sum_div_card,
      ← Finset.sum_fiberwise Finset.univ (residues sample) value, Finset.sum_div]
    apply Finset.sum_congr rfl
    intro matrix _
    have fiberEq :
        Finset.univ.filter (fun chunks : ι → Chunk => residues sample chunks = matrix) =
          preimages sample matrix := by
      ext chunks
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, mem_preimages]
    have positive : (0 : ℝ) < (preimages sample matrix).card := by
      exact_mod_cast (preimagesNonempty matrix).card_pos
    rw [fiberEq]
    change (∑ chunks ∈ preimages sample matrix, value chunks) / _ =
      _ * (𝔼 chunks ∈ preimages sample matrix, value chunks)
    rw [Finset.expect_eq_sum_div_card, Finset.card_univ, Fintype.card_fun]
    simp only [sampledWeight]
    push_cast
    have productPositive : (0 : ℝ) < ∏ index, ((fiber sample (matrix index)).card : ℝ) := by
      rw [← preimagesCard]
      exact positive
    rw [preimagesCard, Finset.prod_div_distrib, Finset.prod_const, Finset.card_univ,
      div_mul_div_comm, mul_comm (∏ index, ((fiber sample (matrix index)).card : ℝ)),
      mul_div_mul_right _ _ productPositive.ne']
  have programmed : 𝔼 matrix : ι → Out, average matrix =
      ∑ matrix : ι → Out, (∏ _index : ι, (1 / (Fintype.card Out : ℝ))) * average matrix := by
    rw [Finset.expect_eq_sum_div_card, Finset.sum_div]
    apply Finset.sum_congr rfl
    intro matrix _
    simp only [Finset.prod_const, Finset.card_univ, Fintype.card_fun]
    push_cast
    rw [one_div, inv_pow, div_eq_mul_inv, mul_comm]
  have weights : ∑ matrix : ι → Out, |∏ index, (sampledWeight sample (matrix index) : ℝ) -
      ∏ _index : ι, (1 / (Fintype.card Out : ℝ))| ≤ Fintype.card ι * (error : ℝ) := by
    have rational := (product_difference_le (ι := ι) (sampledWeight sample)
      (fun _ => (1 / (Fintype.card Out : ℚ))) (sampledWeight_nonneg sample)
      (fun _ => by positivity) (sampledWeight_sum sample) uniform_sum).trans
      (mul_le_mul_of_nonneg_left bias (Nat.cast_nonneg _))
    have cast := (Rat.cast_le (K := ℝ)).mpr rational
    push_cast at cast
    exact cast
  rw [real, programmed, ← sub_le_iff_le_add', ← Finset.sum_sub_distrib]
  calc ∑ matrix : ι → Out,
        ((∏ index, (sampledWeight sample (matrix index) : ℝ)) * average matrix -
        (∏ _index : ι, (1 / (Fintype.card Out : ℝ))) * average matrix)
      ≤ ∑ matrix : ι → Out, |∏ index, (sampledWeight sample (matrix index) : ℝ) -
          ∏ _index : ι, (1 / (Fintype.card Out : ℝ))| := by
        refine Finset.sum_le_sum fun matrix _ => ?_
        rw [← sub_mul]
        exact (le_abs_self _).trans (by
          rw [abs_mul, abs_of_nonneg (average_nonnegative matrix)]
          exact mul_le_of_le_one_right (abs_nonneg _) (average_le_one matrix))
    _ ≤ _ := weights

open Rejection

theorem element_onto (value : RingF) : (fiber element value).Nonempty :=
  onto_of_uniform_good element Fallback _ card_element startCount_element_pos value

/-- The element of a uniform stream is within `2 · fallbackBound` of a uniform ring element, in
total absolute difference of the two laws. -/
theorem element_bias_le :
    ∑ value : RingF, |sampledWeight element value - 1 / (Fintype.card RingF : ℚ)| ≤
      2 * fallbackBound :=
  (bias_le_of_uniform_good element Fallback _ card_element).trans
    (mul_le_mul_of_nonneg_left card_fallback_le zero_le_two)

abbrev KeyIndex (rows blocks : Nat) := Fin rows × Fin blocks

def keyOf {rows blocks : Nat} (matrix : KeyIndex rows blocks → RingF) :
    Fin rows → Fin blocks → RingF :=
  fun row block => matrix (row, block)

def setupStreams {rows blocks : Nat} (seed : List Nat) (index : KeyIndex rows blocks) : Stream :=
  streamOfWords (by decide) (elementWords seed index.1.val index.2.val)
    (elementWords_below _ _ _)

theorem verifierKey_eq {rows blocks : Nat} (setup : Setup rows blocks) :
    setup.verifierKey = keyOf (residues element (setupStreams setup.seed.bytes)) := by
  funext row block lane
  apply Fin.ext
  change elementCoefficientNat setup.seed.bytes row.val block.val lane.val =
    keptCoefficientNat (words (setupStreams setup.seed.bytes (row, block))) lane.val
  rw [setupStreams, words_streamOfWords _ _ _ (elementWords_length _ _ _)]
  rfl

def IsCollision {shape : RingRelation.Shape} {rows : Nat} (bound : Nat)
    (key : Commitment.Key shape rows)
    (openings : RingRelation.Assignment shape × RingRelation.Assignment shape) : Prop :=
  Commitment.commit key openings.1 = Commitment.commit key openings.2 ∧
    assignmentNormBounded bound openings.1 ∧ assignmentNormBounded bound openings.2 ∧
    openings.1 ≠ openings.2

def IsCollision.shortKernel {shape : RingRelation.Shape} {rows bound : Nat}
    {key : Commitment.Key shape rows}
    {openings : RingRelation.Assignment shape × RingRelation.Assignment shape}
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

variable {shape : RingRelation.Shape} {rows : Nat} {Extra : Type} [Fintype Extra] [Nonempty Extra]

noncomputable def solverSuccess (bound : Nat)
    (attack : (KeyIndex rows (ConcreteRingColumnLayout.blockCount shape.carrierWidth) → Stream) × Extra →
      RingRelation.Assignment shape × RingRelation.Assignment shape) : ℚ :=
  ∑ matrix : KeyIndex rows (ConcreteRingColumnLayout.blockCount shape.carrierWidth) → RingF,
    programmedFor element (fun view => IsCollision bound (keyOf matrix) (attack view)) matrix /
      Fintype.card (KeyIndex rows (ConcreteRingColumnLayout.blockCount shape.carrierWidth) → RingF)

theorem binding_le_solver (bound : Nat)
    (attack : (KeyIndex rows (ConcreteRingColumnLayout.blockCount shape.carrierWidth) → Stream) × Extra →
      RingRelation.Assignment shape × RingRelation.Assignment shape) :
    real (fun view => IsCollision bound (keyOf (residues element view.1)) (attack view)) ≤
      solverSuccess bound attack +
        Fintype.card (KeyIndex rows (ConcreteRingColumnLayout.blockCount shape.carrierWidth)) *
          (2 * fallbackBound) := by
  have difference := real_sub_programmed_le element element_onto element_bias_le
    (fun view => IsCollision bound (keyOf (residues element view.1)) (attack view))
  have same : programmed element
      (fun view => IsCollision bound (keyOf (residues element view.1)) (attack view)) =
      solverSuccess bound attack := by
    unfold programmed solverSuccess programmedFor
    apply sum_congr rfl
    intro matrix _
    rw [filter_congr (s := views element matrix)
      (p := fun view => IsCollision bound (keyOf (residues element view.1)) (attack view))
      (q := fun view => IsCollision bound (keyOf matrix) (attack view))
      (fun view member => by rw [(mem_preimages element).mp (mem_product.mp member).1])]
  rw [same] at difference
  linarith [le_abs_self (real (fun view => IsCollision bound (keyOf (residues element view.1))
    (attack view)) - solverSuccess bound attack)]

end Binding

end NightstreamFPrime.Spec.AjtaiSetupV1.Programming
