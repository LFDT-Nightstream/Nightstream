import NightstreamFPrime.Spec.Compression.Akita.Replay.SumCheck
import NightstreamFPrime.Spec.Compression.Akita.NormCheck

/-! The range proof of one fold of the pinned Akita verifier on words (Akita revision `d50f700`,
`akita-verifier/src/stages/stage1.rs`, `akita-params/src/layout/digit_range.rs`), and its relation
to the range tree of the specification (`RangeTree.lean`; `docs/akita-work/match-design.md`
section 7, item 7).

- `Ext.same a b` is the test `a = b` in `E`. The words of `a` and `b` need not be below `q`.
- `stageEnd` is the end claim of a product stage, `Σ_m weight_m·Π_t child_{m·a+t}`
  (`Stage1Verifier::verify_product_prefix`), `batched` is `Σ_i γ^i·c_i`
  (`interstage_batch_weights`, `batch_claims`), and `leafEnd` is the end claim of the leaf stage,
  `Σ_m weight_m·Π_j (s − k(k + 1))` for `k = m·leaf + j` (`batch_leaf_polynomials`,
  `evaluate_leaf_polynomial`).
- `treeRun` is the verifier of the product stages and `leafRun` the verifier of the leaf stage on
  the coefficient route. `none` is a rejection, or rounds whose count is not the count of the
  coordinates of the point before.
- `treeRun_runs` and `leafRun_runs`: on the path of the messages and the challenges of the code,
  `rangeTree` and `leafStage` of the specification end in the result of the code. `rangeRun_runs`
  is this statement for `rangeCheck`. A result `none` is a rejection of the specification on the
  same path, and a result `some` is its end.
- `normRun` is the last stage on the Euclidean route (`stages/physical_l2_norm.rs`, direct shape):
  the integer checks `E_resp ≤ S_max` and `E_resp < q` of `validate_integer_claim`
  (`energyFits`), the plain sum-check for `C_rng + ζ·E_resp`, and the end check
  `end = ẽq(point, r)·leaf(s) + ζ·z²` (`expected_output_claim`). `normRun_runs` is the statement
  for `normLeafStage`, which checks only `E_resp ≤ S_max`: for `S_max < q` the two checks give the
  same result on each `E_resp` (`energyFits_iff`). The shape check of `validate_integer_claim`
  reads only the schedule. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game Polynomial
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita

def Ext.same (a b : Ext) : Bool := decide (a - b = Ext.zero)

theorem Ext.same_iff (a b : Ext) : a.same b = true ↔ a.toE = b.toE := by
  rw [Ext.same, decide_eq_true_iff]
  constructor
  · intro zero
    have value := congrArg Ext.toE zero
    rw [Ext.toE_sub, Ext.toE_zero] at value
    exact sub_eq_zero.mp value
  · intro equal
    refine Ext.toE_injective (Fq.ofNat_canonical _) (Fq.ofNat_canonical _) (by decide) (by decide)
      ?_
    rw [Ext.toE_sub, Ext.toE_zero, equal, sub_self]

def natExt : ℕ → Ext
  | 0 => Ext.zero
  | n + 1 => natExt n + Ext.one

def powExt (γ : Ext) : ℕ → Ext
  | 0 => Ext.one
  | n + 1 => powExt γ n * γ

/-- `Σ_{i<n} f i`. -/
def sumTo (f : ℕ → Ext) : ℕ → Ext
  | 0 => Ext.zero
  | n + 1 => sumTo f n + f n

/-- `Π_{i<n} f i`. -/
def prodTo (f : ℕ → Ext) : ℕ → Ext
  | 0 => Ext.one
  | n + 1 => prodTo f n * f n

/-- `Π_{j<leaf} (s − k(k + 1))` for `k = m·leaf + j`. -/
def nodeEval (leaf m : ℕ) (s : Ext) : Ext :=
  prodTo (fun j => s - natExt (m * leaf + j) * (natExt (m * leaf + j) + Ext.one)) leaf

def stageEnd (nodes arity : ℕ) (weights : ℕ → Ext) (children : List Ext) : Ext :=
  sumTo (fun m => weights m * prodTo (fun t => children.getD (m * arity + t) Ext.zero) arity) nodes

def batched (count : ℕ) (children : List Ext) (γ : Ext) : Ext :=
  sumTo (fun i => powExt γ i * children.getD i Ext.zero) count

def leafEnd (leaf nodes : ℕ) (weights : ℕ → Ext) (image : Ext) : Ext :=
  sumTo (fun m => weights m * nodeEval leaf m image) nodes

/-- One product stage of the range proof: its degree, its rounds, the claims of the children and
the batch challenge. -/
structure Product where
  arity : ℕ
  rounds : List (List Ext × Ext)
  children : List Ext
  gamma : Ext

def treeRun {β : Type} (last : ℕ → (ℕ → Ext) → List Ext → Ext → Option β) :
    List Product → ℕ → (ℕ → Ext) → List Ext → Ext → Option β
  | [], nodes, weights, point, claim => last nodes weights point claim
  | S :: rest, nodes, weights, point, claim =>
      match eqRun claim Ext.one point S.rounds with
      | none => none
      | some ended =>
          if ended.claim.same (stageEnd nodes S.arity weights S.children) then
            treeRun last rest (nodes * S.arity) (powExt S.gamma) ended.point
              (batched (nodes * S.arity) S.children S.gamma)
          else none

def leafRun (leaf : ℕ) (rounds : List (List Ext × Ext)) (image : Ext) (nodes : ℕ)
    (weights : ℕ → Ext) (point : List Ext) (claim : Ext) : Option (List Ext × Ext) :=
  match eqRun claim Ext.one point rounds with
  | none => none
  | some ended =>
      if ended.claim.same (leafEnd leaf nodes weights image) then some (ended.point, image)
      else none

/-- The range proof on the coefficient route for the base `b`, from the range point. -/
def rangeRun (b : ℕ) (stages : List Product) (rounds : List (List Ext × Ext)) (image : Ext)
    (point : List Ext) : Option (List Ext × Ext) :=
  treeRun (leafRun (rangeLeaf b) rounds image) stages 1 (fun _ => Ext.one) point Ext.zero

theorem entry_listFn {K : Type} [Field K] {n i : ℕ} (l : List K) (inside : i < n) :
    entry (listFn n l) i = l.getD i 0 := by
  rw [entry, dite_eq_left inside]
  rfl

section Spec

variable {K : Type} [Field K] {φ : Ext → K} (hom : ExtHom φ)
include hom

theorem map_natExt : ∀ n : ℕ, φ (natExt n) = (n : K)
  | 0 => by rw [natExt, hom.zero, Nat.cast_zero]
  | n + 1 => by rw [natExt, hom.add, map_natExt n, hom.one, Nat.cast_succ]

theorem map_powExt (γ : Ext) : ∀ n : ℕ, φ (powExt γ n) = φ γ ^ n
  | 0 => by rw [powExt, hom.one, pow_zero]
  | n + 1 => by rw [powExt, hom.mul, map_powExt γ n, pow_succ]

theorem map_sumTo (f : ℕ → Ext) : ∀ n : ℕ, φ (sumTo f n) = ∑ i ∈ Finset.range n, φ (f i)
  | 0 => by rw [sumTo, hom.zero, Finset.sum_range_zero]
  | n + 1 => by rw [sumTo, hom.add, map_sumTo f n, Finset.sum_range_succ]

theorem map_prodTo (f : ℕ → Ext) : ∀ n : ℕ, φ (prodTo f n) = ∏ i ∈ Finset.range n, φ (f i)
  | 0 => by rw [prodTo, hom.one, Finset.prod_range_zero]
  | n + 1 => by rw [prodTo, hom.mul, map_prodTo f n, Finset.prod_range_succ]

theorem map_getD (l : List Ext) (i : ℕ) : φ (l.getD i Ext.zero) = (l.map φ).getD i 0 := by
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_map]
  cases l[i]? <;> simp [hom.zero]

theorem map_nodeEval (leaf m : ℕ) (s : Ext) : φ (nodeEval leaf m s) = nodeIn leaf m (φ s) := by
  rw [nodeEval, map_prodTo hom, nodeIn]
  refine Finset.prod_congr rfl fun j _ => ?_
  rw [hom.sub, hom.mul, hom.add, map_natExt hom, hom.one]

theorem map_stageEnd (nodes arity : ℕ) (weights : ℕ → Ext) (children : List Ext) :
    φ (stageEnd nodes arity weights children) = ∑ m ∈ Finset.range nodes, φ (weights m) *
      ∏ t ∈ Finset.range arity, entry (listFn (nodes * arity) (children.map φ)) (m * arity + t) := by
  rw [stageEnd, map_sumTo hom]
  refine Finset.sum_congr rfl fun m member => ?_
  rw [hom.mul, map_prodTo hom]
  refine congrArg _ (Finset.prod_congr rfl fun t inside => ?_)
  have small : m * arity + t < nodes * arity := by
    have row := Finset.mem_range.mp member
    have column := Finset.mem_range.mp inside
    calc m * arity + t < m * arity + arity := by omega
      _ = (m + 1) * arity := by ring
      _ ≤ nodes * arity := Nat.mul_le_mul_right _ row
  rw [entry_listFn _ small, map_getD hom]

theorem map_batched (count : ℕ) (children : List Ext) (γ : Ext) :
    φ (batched count children γ) =
      (rowPoly count (entry (listFn count (children.map φ)))).eval (φ γ) := by
  rw [batched, map_sumTo hom, rowPoly_eval]
  refine Finset.sum_congr rfl fun i member => ?_
  rw [hom.mul, map_powExt hom, entry_listFn _ (Finset.mem_range.mp member), map_getD hom]

theorem map_leafEnd (leaf nodes : ℕ) (weights : ℕ → Ext) (image : Ext) :
    φ (leafEnd leaf nodes weights image) =
      ∑ m ∈ Finset.range nodes, φ (weights m) * nodeIn leaf m (φ image) := by
  rw [leafEnd, map_sumTo hom]
  exact Finset.sum_congr rfl fun m _ => by rw [hom.mul, map_nodeEval hom]

end Spec

section Paths

variable {K : Type} [Field K] (φ : Ext → K)

/-- The path of `rangeTree` for the messages and the challenges of the code: the rounds of each
product stage, the claims of its children and, when its end check passes, its batch challenge;
then the path `last` of the last stage. -/
noncomputable def treePath (last : ℕ → (ℕ → Ext) → List Ext → Ext → List Move) :
    List Product → ℕ → (ℕ → Ext) → List Ext → Ext → List Move
  | [], nodes, weights, point, claim => last nodes weights point claim
  | S :: rest, nodes, weights, point, claim =>
      eqCodePath φ S.arity S.rounds ++
        Move.recv (listFn (nodes * S.arity) (S.children.map φ)) ::
          match eqRun claim Ext.one point S.rounds with
          | none => []
          | some ended =>
              if ended.claim.same (stageEnd nodes S.arity weights S.children) then
                Move.draw (φ S.gamma) :: treePath last rest (nodes * S.arity) (powExt S.gamma)
                  ended.point (batched (nodes * S.arity) S.children S.gamma)
              else []

/-- The path of `leafStage`: its rounds and the value `s_claim`. -/
noncomputable def leafPath (leaf : ℕ) (rounds : List (List Ext × Ext)) (image : Ext) :
    List Move :=
  eqCodePath φ leaf rounds ++ [Move.recv (φ image)]

end Paths

section Runs

variable {K : Type} [Field K] [Fintype K] [DecidableEq K] [Algebra Zq K] {φ : Ext → K}
  (hom : ExtHom φ) (same : ∀ a b : Ext, a.same b = true ↔ φ a = φ b)
include hom same

theorem leafRun_runs (leaf μ : ℕ) (support : ℕ → Prop) [DecidablePred support] (w : Table)
    (rounds : List (List Ext × Ext)) (image : Ext) (nodes : ℕ) (weights : ℕ → Ext)
    (point : List Ext) (claim : Ext) (count : rounds.length = μ) (long : point.length = μ)
    (lengths : ∀ x ∈ rounds, x.1.length = leaf) :
    (leafStage leaf μ support w nodes (fun m => φ (weights m)) (point.map φ) (φ claim)).Runs
      (leafPath φ leaf rounds image)
      ((leafRun leaf rounds image nodes weights point claim).map fun out =>
        (out.1.map φ, φ out.2)) := by
  obtain ⟨ended, ran⟩ : ∃ ended, eqRun claim Ext.one point rounds = some ended :=
    Option.ne_none_iff_exists'.mp fun none =>
      (eqRun_none point rounds claim Ext.one).mp none (long.trans count.symm)
  have rounds_run := eqRun_eqSumcheck hom leaf
    (combRound μ (rangeTables (point.map φ) w)
      (leafComb K leaf nodes fun m => φ (weights m)))
    (fun bound => bindBad (μ - bound.length - 1) (fold bound (supportTable support w)))
    (point.map φ) point rounds [] claim Ext.one ended lengths ran rfl
  rw [count] at rounds_run
  refine runs_bind rounds_run (runs_recv ?_)
  simp only [List.nil_append, leafRun, ran]
  by_cases passes : ended.claim.same (leafEnd leaf nodes weights image) = true
  · rw [ite_eq_left passes, ite_eq_left (((same _ _).mp passes).trans (map_leafEnd hom _ _ _ _))]
    rfl
  · rw [ite_eq_right passes, ite_eq_right fun equal =>
      passes ((same _ _).mpr (equal.trans (map_leafEnd hom _ _ _ _).symm))]
    rfl

theorem treeRun_runs {βc β : Type} (leaf μ : ℕ) (w : Table) (ψ : βc → β)
    (lastGame : ℕ → (ℕ → K) → List K → K → Game (Option β))
    (last : ℕ → (ℕ → Ext) → List Ext → Ext → Option βc)
    (lastPath : ℕ → (ℕ → Ext) → List Ext → Ext → List Move)
    (lastRuns : ∀ nodes weights point claim, point.length = μ →
      (lastGame nodes (fun m => φ (weights m)) (point.map φ) (φ claim)).Runs
        (lastPath nodes weights point claim) ((last nodes weights point claim).map ψ)) :
    ∀ (stages : List Product) (nodes : ℕ) (weights : ℕ → Ext) (point : List Ext) (claim : Ext),
      point.length = μ →
      (∀ S ∈ stages, S.rounds.length = μ ∧ ∀ x ∈ S.rounds, x.1.length = S.arity) →
      (rangeTree leaf μ w lastGame (stages.map (·.arity)) nodes (fun m => φ (weights m))
        (point.map φ) (φ claim)).Runs (treePath φ lastPath stages nodes weights point claim)
        ((treeRun last stages nodes weights point claim).map ψ)
  | [], nodes, weights, point, claim, long, _ => by
      simpa [rangeTree, treePath, treeRun] using lastRuns nodes weights point claim long
  | S :: rest, nodes, weights, point, claim, long, shaped => by
      obtain ⟨count, lengths⟩ := shaped S List.mem_cons_self
      obtain ⟨ended, ran⟩ : ∃ ended, eqRun claim Ext.one point S.rounds = some ended :=
        Option.ne_none_iff_exists'.mp fun none =>
          (eqRun_none point S.rounds claim Ext.one).mp none (long.trans count.symm)
      have rounds_run := eqRun_eqSumcheck hom S.arity
        (combRound μ (stageTables ((rest.map (·.arity)).prod * leaf) (point.map φ) w)
          (stageComb K S.arity nodes fun m => φ (weights m)))
        (fun _ => ∅) (point.map φ) point S.rounds [] claim Ext.one ended lengths ran rfl
      rw [count] at rounds_run
      simp only [List.map_cons, rangeTree, treePath, treeRun, ran]
      refine runs_bind rounds_run (runs_recv ?_)
      simp only [List.nil_append]
      by_cases passes : ended.claim.same (stageEnd nodes S.arity weights S.children) = true
      · rw [ite_eq_left passes, ite_eq_left passes,
          ite_eq_left (((same _ _).mp passes).trans (map_stageEnd hom _ _ _ _))]
        refine runs_draw ?_
        have next := treeRun_runs leaf μ w ψ lastGame last lastPath lastRuns rest
          (nodes * S.arity) (powExt S.gamma) ended.point
          (batched (nodes * S.arity) S.children S.gamma)
          ((eqRun_factor hom point S.rounds claim Ext.one ended ran).1.trans long)
          fun T member => shaped T (List.mem_cons_of_mem _ member)
        rw [map_batched hom] at next
        simpa only [map_powExt hom] using next
      · rw [ite_eq_right passes, ite_eq_right passes, ite_eq_right fun equal =>
          passes ((same _ _).mpr (equal.trans (map_stageEnd hom _ _ _ _).symm))]
        rfl

/-- `rangeCheck` of the specification, on the path of the messages and the challenges of the code
from the range point, ends in the result of `rangeRun`: `none` when an end check of a stage fails,
and otherwise the point `r_virt` and the value `s_claim`. -/
theorem rangeRun_runs (b μ : ℕ) (support : ℕ → Prop) [DecidablePred support] (w : Table)
    (stages : List Product) (rounds : List (List Ext × Ext)) (image : Ext) (point : List Ext)
    (arities : stages.map (·.arity) = rangeArities b) (long : point.length = μ)
    (shaped : ∀ S ∈ stages, S.rounds.length = μ ∧ ∀ x ∈ S.rounds, x.1.length = S.arity)
    (count : rounds.length = μ) (lengths : ∀ x ∈ rounds, x.1.length = rangeLeaf b) :
    (rangeCheck b μ support w (point.map φ)).Runs
      (treePath φ (fun _ _ _ _ => leafPath φ (rangeLeaf b) rounds image) stages 1
        (fun _ => Ext.one) point Ext.zero)
      ((rangeRun b stages rounds image point).map fun out => (out.1.map φ, φ out.2)) := by
  have runs := treeRun_runs hom same (rangeLeaf b) μ w (fun out => (out.1.map φ, φ out.2))
    (leafStage (rangeLeaf b) μ support w) (leafRun (rangeLeaf b) rounds image)
    (fun _ _ _ _ => leafPath φ (rangeLeaf b) rounds image)
    (fun nodes weights point claim long =>
      leafRun_runs hom same (rangeLeaf b) μ support w rounds image nodes weights point claim count
        long lengths)
    stages 1 (fun _ => Ext.one) point Ext.zero long shaped
  rw [arities] at runs
  simpa only [rangeCheck, rangeRun, hom.one, hom.zero] using runs

end Runs

/-- `ẽq(τ, r) = Π_i eq(τ_i, r_i)`. -/
def eqList : List Ext → List Ext → Ext
  | τ :: before, r :: point => eqVal τ r * eqList before point
  | _, _ => Ext.one

/-- The integer checks of `validate_integer_claim` for the direct shape: `E_resp ≤ S_max` and
`E_resp < q`. -/
def energyFits (energy limit : ℕ) : Bool := decide (energy ≤ limit) && decide (energy < fieldModulus)

/-- For `S_max < q` the checks of the code are the check `E_resp ≤ S_max` of `normLeafStage`. -/
theorem energyFits_iff {energy limit : ℕ} (small : limit < fieldModulus) :
    energyFits energy limit = true ↔ energy ≤ limit := by
  rw [energyFits, Bool.and_eq_true, decide_eq_true_iff, decide_eq_true_iff]
  exact ⟨And.left, fun fits => ⟨fits, fits.trans_lt small⟩⟩

def normRun (leaf limit energy : ℕ) (merge : Ext) (rounds : List (List Ext × Ext))
    (virtual image : Ext) (nodes : ℕ) (weights : ℕ → Ext) (point : List Ext) (claim : Ext) :
    Option (List Ext × Ext × Ext) :=
  if energyFits energy limit then
    let ended := plainRun (claim + merge * Ext.ofBase (Fq.ofNat energy)) rounds
    if ended.claim.same (eqList point ended.point * leafEnd leaf nodes weights image +
        merge * (virtual * virtual)) then
      some (ended.point, image, virtual)
    else none
  else none

theorem plainRun_point_length : ∀ (rounds : List (List Ext × Ext)) (claim : Ext),
    (plainRun claim rounds).point.length = rounds.length
  | [], _ => rfl
  | (message, r) :: rounds, claim => by
      simp [plainRun, plainRun_point_length rounds]

/-- The path of `normLeafStage`: `E_resp` and, when it is at most `S_max`, the challenge `ζ`, the
rounds and the values `z̃(r_virt)` and `s_claim`. -/
noncomputable def normPath {K : Type} [Field K] (φ : Ext → K) (leaf limit energy : ℕ)
    (merge : Ext) (rounds : List (List Ext × Ext)) (virtual image claim : Ext) : List Move :=
  Move.recv energy ::
    if energy ≤ limit then
      Move.draw (φ merge) ::
        (plainPath φ leaf (claim + merge * Ext.ofBase (Fq.ofNat energy)) rounds ++
          [Move.recv (φ virtual, φ image)])
    else []

section Norm

variable {K : Type} [Field K] [Fintype K] [DecidableEq K] [Algebra Zq K] {φ : Ext → K}
  (hom : ExtHom φ) (same : ∀ a b : Ext, a.same b = true ↔ φ a = φ b)
  (nat : ∀ n : ℕ, φ (Ext.ofBase (Fq.ofNat n)) = (n : K))
include hom

omit [Fintype K] [DecidableEq K] [Algebra Zq K] in
theorem map_eqList : ∀ τ r : List Ext, φ (eqList τ r) = eqEval (τ.map φ) (r.map φ)
  | [], _ => by simp [eqList, eqEval, hom.one]
  | _ :: _, [] => by simp [eqList, eqEval, hom.one]
  | τ :: before, r :: point => by
      simp only [eqList, eqEval, List.map_cons]
      rw [hom.mul, map_eqList before point, eqVal, hom.add, hom.mul, hom.mul, hom.sub, hom.sub,
        hom.one]

include same nat

/-- `normLeafStage` of the specification, on the path of the messages and the challenges of the
code, ends in the result of `normRun`. -/
theorem normRun_runs (leaf μ limit : ℕ) (kernel : ℕ → ℕ → Zq) (support : ℕ → Prop)
    [DecidablePred support] (w : Table) (energy : ℕ) (merge : Ext)
    (rounds : List (List Ext × Ext)) (virtual image : Ext) (nodes : ℕ) (weights : ℕ → Ext)
    (point : List Ext) (claim : Ext) (positive : 1 ≤ leaf) (count : rounds.length = μ)
    (long : point.length = μ) (lengths : ∀ x ∈ rounds, x.1.length = leaf + 1)
    (small : limit < fieldModulus) :
    (normLeafStage leaf μ limit kernel support w nodes (fun m => φ (weights m)) (point.map φ)
      (φ claim)).Runs (normPath φ leaf limit energy merge rounds virtual image claim)
      ((normRun leaf limit energy merge rounds virtual image nodes weights point claim).map
        fun out => (out.1.map φ, φ out.2.1, φ out.2.2)) := by
  have degree : max (leaf + 1) 2 = leaf + 1 := by omega
  unfold normLeafStage normPath normRun
  refine runs_recv ?_
  by_cases fits : energy ≤ limit
  · simp only [ite_eq_left fits, ite_eq_left ((energyFits_iff small).mpr fits)]
    refine runs_draw ?_
    have rounds_run := plainRun_runs_with hom leaf
      (combRound μ (normLeafTables μ kernel (point.map φ) w)
        (normLeafComb K leaf nodes (fun m => φ (weights m)) (φ merge)))
      (fun bound => bindBad (μ - bound.length - 1) (fold bound (supportTable support w)))
      rounds [] (claim + merge * Ext.ofBase (Fq.ofNat energy)) lengths
    rw [hom.add, hom.mul, nat, count] at rounds_run
    rw [degree]
    refine runs_bind rounds_run (runs_recv ?_)
    have factor : mle ((plainRun (claim + merge * Ext.ofBase (Fq.ofNat energy)) rounds).point.map
        φ) (eqAt (point.map φ)) = φ (eqList point
          (plainRun (claim + merge * Ext.ofBase (Fq.ofNat energy)) rounds).point) := by
      rw [map_eqList hom, mle_eqAt _ _ (by simp [plainRun_point_length, count, long])]
    have value : φ (eqList point
          (plainRun (claim + merge * Ext.ofBase (Fq.ofNat energy)) rounds).point *
        leafEnd leaf nodes weights image + merge * (virtual * virtual)) =
        mle ((plainRun (claim + merge * Ext.ofBase (Fq.ofNat energy)) rounds).point.map φ)
          (eqAt (point.map φ)) * ∑ m ∈ Finset.range nodes, φ (weights m) *
            nodeIn leaf m (φ image) + φ merge * (φ virtual * φ virtual) := by
      rw [hom.add, hom.mul, hom.mul, hom.mul, map_leafEnd hom, factor]
    simp only [List.nil_append]
    by_cases passes : (plainRun (claim + merge * Ext.ofBase (Fq.ofNat energy)) rounds).claim.same
        (eqList point (plainRun (claim + merge * Ext.ofBase (Fq.ofNat energy)) rounds).point *
          leafEnd leaf nodes weights image + merge * (virtual * virtual)) = true
    · rw [ite_eq_left passes, ite_eq_left (((same _ _).mp passes).trans value)]
      rfl
    · rw [ite_eq_right passes, ite_eq_right fun equal =>
        passes ((same _ _).mpr (equal.trans value.symm))]
      rfl
  · simp only [ite_eq_right fits, ite_eq_right fun passes => fits ((energyFits_iff small).mp passes)]
    rfl

end Norm

section Instances

variable {E : Type} [CommRing E] [Algebra Zq E] [Degree2 E]

/-- The hypothesis `same` of the stage theorems for the map of `Ext` to a field with
`Degree2`. -/
theorem same_degree2 (a b : Ext) :
    a.same b = true ↔
      (Degree2.coords (E := E)).symm a.toE = (Degree2.coords (E := E)).symm b.toE :=
  (Ext.same_iff a b).trans (Degree2.coords (E := E)).symm.injective.eq_iff.symm

/-- The hypothesis `nat` of `normRun_runs` for the map of `Ext` to a field with `Degree2`. -/
theorem nat_degree2 (n : ℕ) :
    (Degree2.coords (E := E)).symm (Ext.ofBase (Fq.ofNat n)).toE = (n : E) := by
  rw [Ext.toE_ofBase, Fq.toZq_ofNat, map_natCast, map_natCast]

end Instances

end NightstreamFPrime.Spec.Compression.Akita.Replay
