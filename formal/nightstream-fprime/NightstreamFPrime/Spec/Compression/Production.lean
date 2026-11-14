import NightstreamFPrime.Spec.Compression.Verifier.Bound
import NightstreamFPrime.Spec.Compression.Verifier.Pass
import NightstreamFPrime.Spec.FieldExtensionRing
import NightstreamFPrime.Spec.Folding.Schedule
import NightstreamFPrime.Spec.RingRelation.Semantics

/-! The production instance of the argument (SPEC.md section 3, Algorithms 9 and 11). `F` is `F_q`
with `q = 2^64 − 59` and `K` is `F_q[u]/(u² − 7)`, a ring block has `D = 64` lanes, the key has
`κ = 18` rows below `2^5`, a relation has `t = 4` matrices, there are `L = 4` lookup challenges,
and the claims are slot 0, slot 1 and the fresh instance, in this order. `params` builds `Params`
from the Ajtai key, the CCS matrices of each claim's relation, the depths of the slots and the
layout of Rust `Layout::new`, which Lean does not model. A claim has the blocks of its relation's
cells, and slot `s` has the table `[−accumulatorBound(depth_s), accumulatorBound(depth_s)]`.
`params_error_lt`: at the production bounds of the layout, `error < 2^−119`.

`verifierData` is the verifier of Rust `verify` with the one pass of Rust `matrix_part` over the
factored rows of each claim's relation. With its matrices, `MatrixSpec` holds
(`verifierData_matrixSpec`), and `soundness` needs of the layout only `WellFormed`, the bound of
the key blocks and the bound `2^25` on the cells of a slot (`params_soundness`).

`slot_normBounded`: `Relation` puts each cell of the witness slice of slot `s` in its table, so
the slice has the norm `accumulatorBound(depth_s) + 1` of `Spec.CE.Holds`, which is at most `B`. -/

namespace NightstreamFPrime.Spec.Compression.Production

open NightstreamFPrime.Spec NightstreamFPrime.Spec.FieldExtensionRing
open NightstreamFPrime.Spec.Compression.Verifier
open NightstreamFPrime.Spec.Folding.Schedule (productionLightFoldCap)
open NightstreamFPrime.Spec.Compression.Game

scoped instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

scoped instance fieldF : Field F := inferInstanceAs (Field (ZMod fieldModulus))

noncomputable scoped instance fieldK : Field K := Fintype.fieldOfDomain K

/-- `F → K`, `x ↦ x + 0·u` (`K.embed`). -/
def embedHom : F →+* K where
  toFun := K.embed
  map_one' := rfl
  map_mul' left right := by
    rw [mul_eq]
    simp only [K.embed, K.mul, mul_zero, zero_mul, add_zero]
  map_zero' := rfl
  map_add' left right := by
    rw [add_eq]
    simp only [K.embed, K.add, add_zero]

noncomputable scoped instance : Algebra F K := embedHom.toAlgebra

theorem card_F : Fintype.card F = fieldOrder := Fintype.card_fin _

/-- Lane `h` of key element `(i, j)` of an Ajtai key with `rows` rows and `blocks` blocks, and 0
outside it (Algorithm 11, step 3). -/
def keyOf {rows blocks : ℕ} (A : Fin rows → Fin blocks → RingF) (i j h : ℕ) : F :=
  if hi : i < rows then
    if hj : j < blocks then if hh : h < ringDegree then A ⟨i, hi⟩ ⟨j, hj⟩ ⟨h, hh⟩ else 0 else 0
  else 0

/-- What Rust `Layout::new` gives for the three claims (Algorithm 9, step 3). Lean does not model
that function. -/
structure Layout where
  witness : Fin 3 → Slice
  counts : Fin 3 → Slice
  witnessVars : ℕ
  spanVars : ℕ
  keyBlockVars : ℕ

/-- The table bound of claim `c`: `accumulatorBound(depth_s)` for slot `s`, and 0 for the fresh
instance, which has no table. -/
def bound (depths : Fin 2 → ℕ) (c : Fin 3) : ℕ :=
  if slot : (c : ℕ) < 2 then productionGlobalParams.accumulatorBound (depths ⟨c, slot⟩) else 0

/-- The production `Params`: the key `key`, the CCS matrices `matrices c` of the relation of claim
`c`, the depths of the slots and the layout. -/
noncomputable def params (key : ℕ → ℕ → ℕ → F) (matrices : Fin 3 → Matrices F)
    (depths : Fin 2 → ℕ) (layout : Layout) : Params F K where
  claims := 3
  shape c :=
    { blocks := (matrices c).columns / ringDegree
      witness := layout.witness c
      publicCells := ringDegree * 5
      lookup := decide ((c : ℕ) < 2)
      bound := bound depths c
      counts := layout.counts c }
  witnessVars := layout.witnessVars
  spanVars := layout.spanVars
  lookups := 4
  laneVars := 6
  rows := productionProfile.commitmentWidth
  rowVars := 5
  keyBlockVars := layout.keyBlockVars
  key := key
  matrices := productionProfile.ccsMatrices
  matrixForm c := (matrices c).form ringDegree

section Params

variable (key : ℕ → ℕ → ℕ → F) (matrices : Fin 3 → Matrices F) (depths : Fin 2 → ℕ)
  (layout : Layout)

theorem params_lanes : (params key matrices depths layout).lanes = ringDegree := rfl

theorem params_equations : (params key matrices depths layout).equations = 23 := rfl

/-- Each table of a slot below the cap has at most `20,737` values. -/
theorem slot_table_le {depth : ℕ} (le : depth ≤ productionLightFoldCap) :
    2 * productionGlobalParams.accumulatorBound depth + 1 ≤ 20737 := by
  simp only [GlobalParams.accumulatorBound, productionGlobalParams, productionLightFoldCap] at le ⊢
  omega

/-- At the production bounds of the layout, the error of `soundness` is below `2^−119`. -/
theorem params_error_lt (vf : VerifierData (params key matrices depths layout))
    (columns : vf.columnVars = 2) (depths_le : ∀ s, depths s ≤ productionLightFoldCap)
    (slots_le : ∀ c : Fin 3, (c : ℕ) < 2 → (layout.witness c).vars ≤ 25)
    (span_le : layout.spanVars ≤ 27) : error vf < 1 / 2 ^ 119 := by
  refine lt_of_le_of_lt (error_le_production vf card_F card_K rfl rfl rfl
    (Finset.sup_le fun _ _ => le_refl _) span_le columns rfl fun c table => ?_)
    production_error_lt
  have slot : (c : ℕ) < 2 := of_decide_eq_true table
  have cells : 2 ^ (layout.witness c).vars ≤ 2 ^ 25 :=
    Nat.pow_le_pow_right two_pos (slots_le c slot)
  have values := slot_table_le (depths_le ⟨c, slot⟩)
  have slot_bound : bound depths c = productionGlobalParams.accumulatorBound (depths ⟨c, slot⟩) :=
    by simp only [bound, slot, dite_true]
  change 2 ^ (layout.witness c).vars + (2 * bound depths c + 1) ≤ 2 ^ 25 + 20737
  omega

/-- A slot of at most `2^25` cells has nonzero counts in `F_q`. -/
theorem params_countsNonzero (slots_le : ∀ c : Fin 3, (c : ℕ) < 2 → (layout.witness c).vars ≤ 25) :
    CountsNonzero (params key matrices depths layout) := by
  intro c table n positive le
  have slot : (c : ℕ) < 2 := of_decide_eq_true table
  have cells : 2 ^ (layout.witness c).vars ≤ 2 ^ 25 :=
    Nat.pow_le_pow_right two_pos (slots_le c slot)
  have below : (2 : ℕ) ^ 25 < fieldModulus := by decide
  change ((n : ZMod fieldModulus)) ≠ 0
  rw [Ne, CharP.cast_eq_zero_iff (ZMod fieldModulus) fieldModulus]
  intro divides
  have := Nat.le_of_dvd positive divides
  change n ≤ 2 ^ (layout.witness c).vars at le
  omega

end Params

section Pass

variable (key : ℕ → ℕ → ℕ → F) (rel : ℕ → Factored F) (which : Fin 3 → ℕ) (depths : Fin 2 → ℕ)
  (layout : Layout) (zeros : ℕ)

/-- The verifier of Rust `verify`: the one pass of Rust `matrix_part` over the factored rows
`rel (which c)` of the relation of claim `c`, `LOOKUP_VARS = 2` and `zeros` zeros in the opening
point of `V`. -/
noncomputable def verifierData :
    VerifierData (params key (fun c => (rel (which c)).matrices) depths layout) :=
  ⟨matrixPass _ rel which, 2, zeros⟩

/-- The one pass computes the matrix part of the honest weights of the production parameters. -/
theorem verifierData_matrixSpec : MatrixSpec (verifierData key rel which depths layout zeros) := by
  have form : ∀ c, (params key (fun c => (rel (which c)).matrices) depths layout).matrixForm c =
      (rel (which c)).matrices.form (params key (fun c => (rel (which c)).matrices) depths
        layout).lanes := fun c => by
    rw [params_lanes]
    rfl
  have blocks : ∀ c c', which c = which c' →
      ((params key (fun c => (rel (which c)).matrices) depths layout).shape c).blocks =
        ((params key (fun c => (rel (which c)).matrices) depths layout).shape c').blocks :=
    fun c c' same => by
      change (rel (which c)).columns / ringDegree = (rel (which c')).columns / ringDegree
      rw [same]
  exact matrixPass_spec (params key (fun c => (rel (which c)).matrices) depths layout) rel which
    form blocks 2 zeros

/-- `soundness` at the production parameters with the one pass of Rust `matrix_part`. -/
theorem params_soundness
    (wf : WellFormed (params key (fun c => (rel (which c)).matrices) depths layout))
    (slots_le : ∀ c : Fin 3, (c : ℕ) < 2 → (layout.witness c).vars ≤ 25)
    (key_fits : ∀ c : Fin 3, (layout.witness c).vars - 6 ≤ layout.keyBlockVars)
    (s : Strategy (Option (Option (Statement (params key (fun c => (rel (which c)).matrices)
      depths layout) × (ℕ → F))))) (q : ℕ) :
    prob s (restore (game (verifierData key rel which depths layout zeros)) q [])
        (fun r => ¬ ∀ a, r = some a → ∀ o, a = some o → Relation _ o.1 o.2) ≤
      q * error (verifierData key rel which depths layout zeros) :=
  soundness _ wf (params_countsNonzero key _ depths layout slots_le)
    (verifierData_matrixSpec key rel which depths layout zeros) (le_of_eq rfl)
    (Nat.le_of_ble_eq_true rfl) key_fits s q

end Pass

section Norm

/-- A value of the table `[−bound, bound]` has centered magnitude at most `bound`. -/
theorem centeredMagnitude_tableValue {bound v : ℕ} (small : 2 * bound < fieldModulus)
    (value : v < 2 * bound + 1) : centeredMagnitude (tableValue bound v : F) ≤ bound := by
  unfold tableValue centeredMagnitude
  change min (ZMod.val ((v : ZMod fieldModulus) - (bound : ZMod fieldModulus)))
    (fieldModulus - ZMod.val ((v : ZMod fieldModulus) - (bound : ZMod fieldModulus))) ≤ bound
  rcases le_or_gt bound v with above | below
  · rw [← Nat.cast_sub above, ZMod.val_natCast,
      Nat.mod_eq_of_lt (show v - bound < fieldModulus by omega)]
    omega
  · rw [show (v : ZMod fieldModulus) - (bound : ZMod fieldModulus) =
          -((bound - v : ℕ) : ZMod fieldModulus) by
        rw [Nat.cast_sub below.le]
        ring,
      ZMod.neg_val', ZMod.val_natCast, Nat.mod_eq_of_lt (show bound - v < fieldModulus by omega),
      Nat.mod_eq_of_lt (show fieldModulus - (bound - v) < fieldModulus by omega)]
    omega

variable (key : ℕ → ℕ → ℕ → F) (matrices : Fin 3 → Matrices F) (depths : Fin 2 → ℕ)
  (layout : Layout)

/-- `Relation` gives each cell of the witness slice of slot `s` a centered magnitude below
`accumulatorBound(depth_s) + 1`, and that bound is at most `B`. -/
theorem slot_norm {st : Statement (params key matrices depths layout)} {W : ℕ → F}
    (holds : Relation (params key matrices depths layout) st W) (s : Fin 2)
    (depth_le : depths s ≤ productionLightFoldCap) :
    (∀ y < 2 ^ (layout.witness s.castSucc).vars,
      centeredMagnitude (W ((layout.witness s.castSucc).offset + y)) <
        productionGlobalParams.accumulatorBound (depths s) + 1) ∧
    productionGlobalParams.accumulatorBound (depths s) + 1 ≤ productionGlobalParams.bigB := by
  have shape_bound : ((params key matrices depths layout).shape s.castSucc).bound =
      productionGlobalParams.accumulatorBound (depths s) := by
    change bound depths s.castSucc = _
    simp only [bound, Fin.val_castSucc, s.isLt, dite_true, Fin.eta]
  have values := slot_table_le depth_le
  have small : 2 * productionGlobalParams.accumulatorBound (depths s) < fieldModulus := by
    have : (20737 : ℕ) < fieldModulus := by decide
    omega
  refine ⟨fun y cell => ?_, ?_⟩
  · obtain ⟨v, v_lt, equal⟩ := (holds s.castSucc).2.2
      (show decide ((s.castSucc : ℕ) < 2) = true from
        decide_eq_true (by rw [Fin.val_castSucc]; exact s.isLt)) y cell
    rw [shape_bound] at v_lt equal
    change W ((layout.witness s.castSucc).offset + y) = _ at equal
    rw [equal]
    exact Nat.lt_succ_of_le (centeredMagnitude_tableValue small v_lt)
  · exact GlobalParams.production_accumulatorBound_lt_bigB (by
      simp only [productionLightFoldCap] at depth_le
      omega)

/-- The witness of slot `s` as an assignment of `Spec.CE.Holds` (the cells of its slice from
`offset`) has the norm `accumulatorBound(depth_s) + 1`. -/
theorem slot_normBounded {st : Statement (params key matrices depths layout)} {W : ℕ → F}
    (holds : Relation (params key matrices depths layout) st W) (s : Fin 2)
    (depth_le : depths s ≤ productionLightFoldCap) {shape : RingRelation.Shape}
    (fits : shape.carrierWidth ≤ 2 ^ (layout.witness s.castSucc).vars) :
    RingRelation.assignmentNormBounded (productionGlobalParams.accumulatorBound (depths s) + 1)
      (fun column : Fin shape.carrierWidth => W ((layout.witness s.castSucc).offset + column)) :=
  fun column => (slot_norm key matrices depths layout holds s depth_le).1 column
    (Nat.lt_of_lt_of_le column.isLt fits)

end Norm

end NightstreamFPrime.Spec.Compression.Production
