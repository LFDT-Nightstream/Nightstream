import NightstreamFPrime.Spec.Compression.Akita.Extract.Offload
import NightstreamFPrime.Spec.Compression.Akita.Extract.TraceLink
import NightstreamFPrime.Spec.Compression.Akita.Extract.TraceNorm

/-! The stage errors of the checks of one level with the work bits of the Akita code (Akita,
ePrint 2026/1983, revision of 2026-11-23, Remark 10.18 and (146); `transcript_grinding/plan.rs`):
each bound of a draw times `2^{−g}` for the work bits `g = workBits L` of the loss `L` of the site
of the draw in the code.

- `treeCredit`, `foldCredit`, `foldCreditSq`, `tensorCredit` and `setupCredit` are the sums of
  these stage errors for `treeCosts`, `foldCosts`, `foldCostsSq`, `tensorCosts` and `setupCosts`,
  with the losses `treeLosses`, `foldLosses`, `foldLossesSq`, `tensorLosses` and `setupLosses`.
- `treeCredit_eq`, `foldCredit_eq`, `foldCreditSq_eq`, `tensorCredit_eq` and `setupCredit_eq`
  state that `credit` of the bounds and the work bits has these sums. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

variable (E : Type) [Fintype E]

/-- The stage errors of the draws of the range tree before its last stage. -/
noncomputable def treeCredit (μ : ℕ) : List ℕ → ℕ → ℝ
  | [], _ => 0
  | a :: rest, nodes => μ * (((a + 1 : ℕ) : ℝ) / Fintype.card E / 2 ^ workBits (a + 1)) +
      ((nodes * a - 1 : ℕ) : ℝ) / Fintype.card E / 2 ^ workBits (nodes * a - 1) +
        treeCredit μ rest (nodes * a)

theorem treeCredit_eq (μ : ℕ) : ∀ (arities : List ℕ) (nodes : ℕ),
    (credit (treeCosts E μ arities nodes) ((treeLosses μ arities nodes).map workBits)).sum =
      treeCredit E μ arities nodes := by
  intro arities
  induction arities with
  | nil => intro nodes; rfl
  | cons a rest step =>
      intro nodes
      simp only [treeCosts, treeLosses, treeCredit, List.map_append, List.map_replicate,
        List.map_cons]
      rw [credit_append (by simp), credit_replicate, credit_cons, List.sum_append, List.sum_cons,
        List.sum_replicate, step, nsmul_eq_mul, add_assoc]

/-- The stage errors of the draws of `foldCheck` (`foldCosts` with `foldLosses`). -/
noncomputable def foldCredit (bad alpha vars batch base : ℕ) (bin : Bool) : ℝ :=
  (bad : ℝ) / Fintype.card E / 2 ^ workBits alpha +
    ((vars : ℝ) / Fintype.card E / 2 ^ workBits vars +
      ((batch : ℝ) / Fintype.card E / 2 ^ workBits batch +
        (treeCredit E vars (rangeArities base) 1 +
          vars * (((rangeLeaf base + 1 + 1 : ℕ) : ℝ) / Fintype.card E /
            2 ^ workBits (leafDegree base + 1)) +
          ((if bin then ((1 : ℕ) : ℝ) / Fintype.card E / 2 ^ workBits 1 else 0) +
            (((1 : ℕ) : ℝ) / Fintype.card E / 2 ^ workBits 1 +
              vars * (((3 : ℕ) : ℝ) / Fintype.card E / 2 ^ workBits 3))))))

theorem binCredit_eq (bin : Bool) :
    (credit (binCosts E bin) ((binLosses bin).map workBits)).sum =
      if bin then ((1 : ℕ) : ℝ) / Fintype.card E / 2 ^ workBits 1 else 0 := by
  cases bin <;> simp [binCosts, binLosses, credit]

theorem foldCredit_eq (bad alpha vars batch base : ℕ) (bin : Bool) :
    (credit (foldCosts E bad vars batch base bin)
      ((foldLosses alpha vars batch base bin).map workBits)).sum =
        foldCredit E bad alpha vars batch base bin := by
  have range : (credit (rangeCosts E vars base) ((rangeLosses vars base).map workBits)).sum =
      treeCredit E vars (rangeArities base) 1 +
        vars * (((rangeLeaf base + 1 + 1 : ℕ) : ℝ) / Fintype.card E /
          2 ^ workBits (leafDegree base + 1)) := by
    simp only [rangeCosts, rangeLosses, List.map_append, List.map_replicate]
    rw [credit_append (by rw [List.length_map]; exact treeLosses_length (E := E) _ _ _),
      credit_replicate, List.sum_append, treeCredit_eq, List.sum_replicate, nsmul_eq_mul]
  simp only [foldCosts, foldLosses, foldCredit, List.map_cons, List.map_append, List.map_replicate]
  rw [credit_cons, credit_cons, credit_cons,
    credit_append (by rw [List.length_map]; exact rangeLosses_length (E := E) _ _),
    credit_append (by rw [List.length_map, binLosses_length, binCosts_length]), credit_cons,
    credit_replicate]
  simp only [List.sum_cons, List.sum_append, List.sum_replicate, nsmul_eq_mul, range,
    binCredit_eq]

/-- The stage errors of the draws of `foldCheckSq` (`foldCostsSq` with `foldLossesSq`). -/
noncomputable def foldCreditSq (bad alpha vars batch base : ℕ) (bin : Bool) : ℝ :=
  (bad : ℝ) / Fintype.card E / 2 ^ workBits alpha +
    ((vars : ℝ) / Fintype.card E / 2 ^ workBits vars +
      ((batch : ℝ) / Fintype.card E / 2 ^ workBits batch +
        (treeCredit E vars (rangeArities base) 1 +
          (((1 : ℕ) : ℝ) / Fintype.card E / 2 ^ workBits 1 +
            vars * (((max (rangeLeaf base + 1) 2 + 1 : ℕ) : ℝ) / Fintype.card E /
              2 ^ workBits (leafDegree base + 1))) +
          (((1 : ℕ) : ℝ) / Fintype.card E / 2 ^ workBits 1 +
            ((if bin then ((1 : ℕ) : ℝ) / Fintype.card E / 2 ^ workBits 1 else 0) +
              (((1 : ℕ) : ℝ) / Fintype.card E / 2 ^ workBits 1 +
                vars * (((3 : ℕ) : ℝ) / Fintype.card E / 2 ^ workBits 3)))))))

theorem foldCreditSq_eq (bad alpha vars batch base : ℕ) (bin : Bool) :
    (credit (foldCostsSq E bad vars batch base bin)
      ((foldLossesSq alpha vars batch base bin).map workBits)).sum =
        foldCreditSq E bad alpha vars batch base bin := by
  simp only [foldCostsSq, foldLossesSq, foldCreditSq, rangeCostsSq, List.map_cons,
    List.map_append, List.map_replicate]
  rw [credit_cons, credit_cons, credit_cons,
    credit_append (by
      simp only [List.length_append, List.length_cons, List.length_replicate, List.length_map,
        treeLosses_length (E := E)]),
    credit_append (by rw [List.length_map]; exact treeLosses_length (E := E) _ _ _), credit_cons,
    credit_replicate, credit_cons,
    credit_append (by rw [List.length_map, binLosses_length, binCosts_length]), credit_cons,
    credit_replicate]
  simp only [List.sum_cons, List.sum_append, List.sum_replicate, nsmul_eq_mul, treeCredit_eq,
    binCredit_eq]

/-- The stage errors of the draws of `tensorCheck` (`tensorCosts` with `tensorLosses`). -/
noncomputable def tensorCredit (ν : ℕ) : ℝ :=
  ((1 : ℕ) : ℝ) / Fintype.card E / 2 ^ workBits 1 +
    ν * (((2 : ℕ) : ℝ) / Fintype.card E / 2 ^ workBits 2)

theorem tensorCredit_eq (ν : ℕ) :
    (credit (tensorCosts E ν) ((tensorLosses ν).map workBits)).sum = tensorCredit E ν := by
  simp only [tensorCosts, tensorLosses, tensorCredit, List.map_cons, List.map_replicate]
  rw [credit_cons, credit_replicate, List.sum_cons, List.sum_replicate, nsmul_eq_mul]

/-- The stage errors of the draws of `setupCheck` (`setupCosts` with `setupLosses`). -/
noncomputable def setupCredit (ν : ℕ) : ℝ :=
  ν * (((2 : ℕ) : ℝ) / Fintype.card E / 2 ^ workBits 2)

theorem setupCredit_eq (ν : ℕ) :
    (credit (setupCosts E ν) ((setupLosses ν).map workBits)).sum = setupCredit E ν := by
  simp only [setupCosts, setupLosses, setupCredit, List.map_replicate]
  rw [credit_replicate, List.sum_replicate, nsmul_eq_mul]

end NightstreamFPrime.Spec.Compression.Akita.Extract
