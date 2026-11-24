import NightstreamFPrime.Spec.Compression.Akita.Extract.Present
import NightstreamFPrime.Spec.Compression.Akita.NormCheck
import NightstreamFPrime.Spec.Compression.Akita.Tensor
import NightstreamFPrime.Spec.Compression.Akita.Extract.Offload

/-! The checks of the levels of Akita are presentable (`Extract/Present.lean`) for the answers
`Coord → E` and the chunks `List E` (Akita, ePrint 2026/1983, revision of 2026-11-23, Figure 6,
Sections 5.2, 6.1 and 6.3): each draw of one field element reads one slot (`exists_fair_slot`),
each draw of a vector of `n ≤ |Coord|` field elements reads `n` slots (`exists_fair_slots`), and
the lists of `E` map onto each message type (`listChunk_onto`). So `exists_presents` gives a
presented segment for each run of `foldCheck`, `foldCheckSq`, `offloadCheck` and `tensorCheck`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open NightstreamFPrime.Spec.Compression (Game)

section Rounds

variable {K : Type} [Field K] [Fintype K] [DecidableEq K] {Coord : Type} [Fintype Coord]
  [DecidableEq Coord] [Nonempty Coord]

theorem presentable_eqSumcheck (D : ℕ) (round : List K → Polynomial K)
    (extra : List K → Finset K) (τ : List K) :
    ∀ (rounds : ℕ) (point : List K) (claim : K),
      Presentable (Coord → K) (List K) (eqSumcheck D round extra τ rounds point claim) := by
  intro rounds
  induction rounds with
  | zero => exact fun _ _ => trivial
  | succ rounds induction =>
      exact fun _ _ => ⟨(listChunk_onto D).2.2.2, fun _ => ⟨exists_fair_slot, fun _ =>
        induction _ _⟩⟩

theorem presentable_sumcheckWith (d : ℕ) (round : List K → Polynomial K)
    (extra : List K → Finset K) :
    ∀ (rounds : ℕ) (point : List K) (claim : K),
      Presentable (Coord → K) (List K) (sumcheckWith d round extra rounds point claim) := by
  intro rounds
  induction rounds with
  | zero => exact fun _ _ => trivial
  | succ rounds induction =>
      exact fun _ _ => ⟨(listChunk_onto (d + 1)).2.2.2, fun _ =>
        Presentable.ite ⟨exists_fair_slot, fun _ => induction _ _⟩ trivial⟩

end Rounds

section Tree

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] {Coord : Type}
  [Fintype Coord] [DecidableEq Coord] [Nonempty Coord]

theorem presentable_rangeTree {β : Type} (leaf μ : ℕ) (w : Table)
    (last : ℕ → (ℕ → E) → List E → E → Game (Option β))
    (lasts : ∀ nodes weights point claim,
      Presentable (Coord → E) (List E) (last nodes weights point claim)) :
    ∀ (arities : List ℕ) (nodes : ℕ) (weights : ℕ → E) (point : List E) (claim : E),
      Presentable (Coord → E) (List E) (rangeTree leaf μ w last arities nodes weights point claim) := by
  intro arities
  induction arities with
  | nil => exact lasts
  | cons a rest induction =>
      intro nodes weights point claim
      simp only [rangeTree]
      exact Presentable.bind (presentable_eqSumcheck _ _ _ _ _ _ _) fun _ =>
        ⟨(listChunk_onto (nodes * a)).2.2.2, fun _ =>
          Presentable.ite ⟨exists_fair_slot, fun _ => induction _ _ _ _⟩ trivial⟩

theorem presentable_leafStage (leaf μ : ℕ) (support : ℕ → Prop) [DecidablePred support]
    (w : Table) (nodes : ℕ) (weights : ℕ → E) (point : List E) (claim : E) :
    Presentable (Coord → E) (List E) (leafStage leaf μ support w nodes weights point claim) := by
  unfold leafStage
  exact Presentable.bind (presentable_eqSumcheck _ _ _ _ _ _ _) fun _ =>
    ⟨(listChunk_onto 0).1, fun _ => Presentable.ite trivial trivial⟩

theorem presentable_normLeafStage (leaf μ limit : ℕ) (kernel : ℕ → ℕ → Zq)
    (support : ℕ → Prop) [DecidablePred support] (w : Table) (nodes : ℕ) (weights : ℕ → E)
    (point : List E) (claim : E) :
    Presentable (Coord → E) (List E)
      (normLeafStage leaf μ limit kernel support w nodes weights point claim) := by
  unfold normLeafStage
  refine ⟨(listChunk_onto 0).2.2.1, fun _ => Presentable.ite ⟨exists_fair_slot, fun _ =>
    Presentable.bind (presentable_sumcheckWith _ _ _ _ _ _) fun result => ?_⟩ trivial⟩
  rcases result with _ | ⟨r, value⟩
  · trivial
  · exact ⟨(listChunk_onto 0).2.1, fun _ => Presentable.ite trivial trivial⟩

theorem presentable_rangeCheck (b μ : ℕ) (support : ℕ → Prop) [DecidablePred support]
    (w : Table) (τ0 : List E) :
    Presentable (Coord → E) (List E) (rangeCheck b μ support w τ0) :=
  presentable_rangeTree _ _ _ _ (presentable_leafStage _ _ _ _) _ _ _ _ _

omit [DecidableEq E] [Algebra Zq E] in
theorem presentable_drawBin {β : Type} (bin : Bool) {bad : Finset E} {next : E → Game β}
    (nexts : ∀ ζ, Presentable (Coord → E) (List E) (next ζ)) :
    Presentable (Coord → E) (List E) (drawBin bin bad next) := by
  cases bin
  · exact nexts 0
  · exact ⟨exists_fair_slot, nexts⟩

theorem presentable_fusedCheck (μ : ℕ) (support : ℕ → Prop) [DecidablePred support] (w : Table)
    (weight : ℕ → E) (target : E) (r : List E) (sClaim ζ γ : E) :
    Presentable (Coord → E) (List E) (fusedCheck μ support w weight target r sClaim ζ γ) := by
  unfold fusedCheck
  refine Presentable.bind (presentable_sumcheck _ _ _ _ _) fun result => ?_
  rcases result with _ | ⟨r2, value⟩
  · trivial
  · exact ⟨(listChunk_onto 0).1, fun _ => Presentable.ite trivial trivial⟩

/-- `foldCheck` is presentable when an answer has a slot for each coordinate of a draw. -/
theorem presentable_foldCheck (R : Reduction E) (b : ℕ) (support : ℕ → Prop)
    [DecidablePred support] (bin : Bool) (w : Table) (vars : R.vars ≤ Fintype.card Coord)
    (batch : R.batch ≤ Fintype.card Coord) :
    Presentable (Coord → E) (List E) (foldCheck R b support bin w) := by
  refine ⟨exists_fair_slot, fun _ => ⟨exists_fair_slots vars, fun _ => ⟨exists_fair_slots batch,
    fun _ => Presentable.bind (presentable_rangeCheck _ _ _ _ _) fun result => ?_⟩⟩⟩
  rcases result with _ | ⟨r, sClaim⟩
  · trivial
  · exact presentable_drawBin _ fun _ => ⟨exists_fair_slot, fun _ =>
      presentable_fusedCheck _ _ _ _ _ _ _ _ _⟩

/-- `foldCheckSq` is presentable when an answer has a slot for each coordinate of a draw. -/
theorem presentable_foldCheckSq (R : Reduction E) (b : ℕ) (support : ℕ → Prop)
    [DecidablePred support] (bin : Bool) (limit : ℕ) (kernel : ℕ → ℕ → Zq) (w : Table)
    (vars : R.vars ≤ Fintype.card Coord) (batch : R.batch ≤ Fintype.card Coord) :
    Presentable (Coord → E) (List E) (foldCheckSq R b support bin limit kernel w) := by
  refine ⟨exists_fair_slot, fun _ => ⟨exists_fair_slots vars, fun _ => ⟨exists_fair_slots batch,
    fun _ => Presentable.bind (presentable_rangeTree _ _ _ _
      (presentable_normLeafStage _ _ _ _ _ _) _ _ _ _ _) fun result => ?_⟩⟩⟩
  rcases result with _ | ⟨r, sClaim, y⟩
  · trivial
  · exact ⟨exists_fair_slot, fun _ => presentable_drawBin _ fun _ => ⟨exists_fair_slot, fun _ =>
      presentable_fusedCheck _ _ _ _ _ _ _ _ _⟩⟩

theorem presentable_setupCheck (ν : ℕ) (flat : Table) (weight : ℕ → E) (σ : E) :
    Presentable (Coord → E) (List E) (setupCheck ν flat weight σ) := by
  unfold setupCheck
  refine Presentable.bind (presentable_sumcheck _ _ _ _ _) fun result => ?_
  rcases result with _ | ⟨ρ, final⟩
  · trivial
  · exact ⟨(listChunk_onto 0).1, fun _ => Presentable.ite trivial trivial⟩

/-- `offloadCheck` is presentable when an answer has a slot for each coordinate of a draw. -/
theorem presentable_offloadCheck (R : Reduction E) (b : ℕ) (support : ℕ → Prop)
    [DecidablePred support] (bin : Bool) (w : Table) (ν : ℕ) (flat : Table)
    (scan : SetupScan R ν flat) (vars : R.vars ≤ Fintype.card Coord)
    (batch : R.batch ≤ Fintype.card Coord) :
    Presentable (Coord → E) (List E) (offloadCheck R b support bin w ν flat scan) := by
  refine ⟨exists_fair_slot, fun _ => ⟨exists_fair_slots vars, fun _ => ⟨exists_fair_slots batch,
    fun _ => Presentable.bind (presentable_rangeCheck _ _ _ _ _) fun result => ?_⟩⟩⟩
  rcases result with _ | ⟨r, sClaim⟩
  · trivial
  · refine presentable_drawBin _ fun _ => ⟨exists_fair_slot, fun _ =>
      Presentable.bind (presentable_sumcheck _ _ _ _ _) fun ended => ?_⟩
    rcases ended with _ | ⟨r2, final⟩
    · trivial
    · exact ⟨(listChunk_onto 0).1, fun _ => ⟨(listChunk_onto 0).1, fun _ =>
        Presentable.ite (Presentable.bind (presentable_setupCheck _ _ _ _) fun claim => by
          rcases claim with _ | _ <;> trivial) trivial⟩⟩

theorem presentable_tensorCheck [Degree2 E] (ν : ℕ) (w : Table) :
    ∀ (point : List E) (v : E), Presentable (Coord → E) (List E) (tensorCheck ν w point v)
  | [], _ => trivial
  | _ :: _, _ => Presentable.ite ⟨(listChunk_onto 0).2.1, fun _ => Presentable.ite
      ⟨exists_fair_slot, fun _ => Presentable.bind (presentable_sumcheck _ _ _ _ _) fun result => by
        rcases result with _ | ⟨ρ, final⟩
        · trivial
        · exact ⟨(listChunk_onto 0).1, fun _ => Presentable.ite trivial trivial⟩⟩ trivial⟩ trivial

end Tree

end NightstreamFPrime.Spec.Compression.Akita.Extract
