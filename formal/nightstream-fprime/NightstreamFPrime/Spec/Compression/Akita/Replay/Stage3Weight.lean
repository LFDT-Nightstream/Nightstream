import NightstreamFPrime.Spec.Compression.Akita.Replay.Range
import NightstreamFPrime.Spec.Compression.Akita.Replay.Relation
import NightstreamFPrime.Spec.Compression.Akita.Extract.SetupFactor

/-! The weight of Stage 3 of the pinned Akita verifier at its point (Akita revision `d50f700`:
`akita-types/src/setup_claims/query.rs`, `evaluate_weight`, branch `Contribution` with
`LiftedPower`; `akita-types/src/setup_contribution/setup_index_weight.rs`,
`SetupIndexWeightMle`; `akita-types/src/setup_contribution/plan/`; `akita-params/src/witness.rs`;
ePrint 2026/1983, Section 9 with (133) and (134)).

- `powerScaleExt b c` is `Π_i ((1 − c_i) + c_i·b^{2^i})` (`evaluate_power_sequence_mle`), the
  `powerScale` of the specification (`powerScaleExt_into`).
- `eqTable r` is `ẽq(r, i)` for each `i < 2^|r|` (`eqTable_getD`), and `EqSplit` the split
  table of the code (`SplitEqEvals`): `ẽq(r, i) = ẽq(r_low, i mod 2^h)·ẽq(r_high, i / 2^h)` for
  `h = ⌊|r|/2⌋`. `EqSplit.eval_into`: it is `eqAt` of the specification.
- `SetupGroup` has the numbers of one group of a fold and `SetupPlan` the groups in the order of
  the code (`root_group_order`, the last group first) and the numbers of `D`.
  `SetupPlan.placed` places each group: the start of its unit `[Z | E | T]` in the witness
  (`WitnessLayout::new`, one chunk), its first row (`C, A, B` of each group, then `D`), and its
  first column of `D`.
- `stage3Weight` is the weight of Stage 3 at the point `ρ`: `powerScale(α, ρ_low)` times the sum
  over the groups and the roles `A`, `B`, `D` of each entry of the view of the setup prefix,
  with the row weight `ẽq(τ_1, row)`, the column weight of the witness at the relation point
  `r_2[κ..]` (`materialize_role_tensor_weights`), and `ẽq` of `ρ_high` at the entry
  (`materialize_setup_index_weights`, `setup_index_weight_at`). A column weight of a role of
  degree `d_R` is the lane sum `Σ_{m < d_R/κ'} β^m·ẽq(r', ℓ + m)`, `β = α^{κ'}` for the
  relation block `κ' = 2^κ`; the weight of `A` has the factor `−g_f` of the fold digit `f`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita NightstreamFPrime.Spec.Compression.Lib

/-- `Π_i ((1 − c_i) + c_i·b^{2^i})`. -/
def powerScaleExt (b : Ext) : List Ext → Ext
  | [] => Ext.one
  | c :: rest => (Ext.one - c + c * b) * powerScaleExt (b * b) rest

/-- `Σ_{i<n} f i`. -/
def sumBelow (n : ℕ) (f : ℕ → Ext) : Ext := Nat.fold n (fun i _ acc => acc + f i) Ext.zero

/-- `ẽq(r, i)` for each `i < 2^|r|`, the coordinate `t` of `r` at the bit `t` of `i`. -/
def eqTable : List Ext → Array Ext
  | [] => #[Ext.one]
  | c :: rest =>
      let t := eqTable rest
      (Array.range (2 * t.size)).map fun i =>
        (if i % 2 = 1 then c else Ext.one - c) * t.getD (i / 2) Ext.zero

/-- The split table of `ẽq(r, ·)`: the `size = 2^h` entries of the first `h` coordinates and the
entries of the others. -/
structure EqSplit where
  size : ℕ
  lo : Array Ext
  hi : Array Ext

def EqSplit.of (r : List Ext) : EqSplit :=
  ⟨2 ^ (r.length / 2), eqTable (r.take (r.length / 2)), eqTable (r.drop (r.length / 2))⟩

def EqSplit.eval (s : EqSplit) (i : ℕ) : Ext :=
  s.lo.getD (i % s.size) Ext.zero * s.hi.getD (i / s.size % s.hi.size) Ext.zero

/-- One side of the sums for a role of `ratio` base rings: `powerScale(b, r_low)` of the first
`log_2 ratio` coordinates, and the split table of the others. -/
structure Side where
  ratio : ℕ
  proj : Ext
  high : EqSplit

def Side.of (b : Ext) (r : List Ext) (ratio : ℕ) : Side :=
  ⟨ratio, powerScaleExt b (r.take (Nat.log 2 ratio)), EqSplit.of (r.drop (Nat.log 2 ratio))⟩

/-- `Σ_{m < ratio} b^m·ẽq(r, ℓ + m)` (`ℓ` a multiple of `ratio`: the factored form of the code). -/
def Side.lane (s : Side) (b : Ext) (r : List Ext) (ℓ : ℕ) : Ext :=
  if ℓ % s.ratio = 0 then s.proj * s.high.eval (ℓ / s.ratio)
  else sumBelow s.ratio fun m => powExt b m * eqIndex r (ℓ + m)

namespace SetupGroup

variable (G : SetupGroup)

def zCols : ℕ := G.positions * G.witnessDigits
def zLen : ℕ := G.zCols * G.foldDigits * G.innerDegree
def eLen : ℕ := G.blocks * G.openDigits * G.openWidth
def tLen : ℕ := G.blocks * G.innerRank * G.commitDigits * G.innerDegree
def unitLen : ℕ := G.zLen + G.eLen + G.tLen
/-- The columns of `D` of the group: `blocks·(k·d_car/d_D)·δ_open`. -/
def dCols (openDegree : ℕ) : ℕ := G.blocks * (G.openWidth / openDegree) * G.openDigits
/-- The rows `C`, `A` and `B` of the group. -/
def rows : ℕ := 1 + G.innerRank + G.slices * G.outerRank
/-- The elements of `R_{q,d_B}` of `t̂` of one block. -/
def blockWidth : ℕ := G.innerRank * (G.innerDegree / G.outerDegree) * G.commitDigits
/-- The first block of slice `s` (`dyadic_block_ranges`). -/
def sliceStart (s : ℕ) : ℕ := G.blocks / G.slices * s + G.blocks % G.slices * s / G.slices
def sliceLen (s : ℕ) : ℕ := G.sliceStart (s + 1) - G.sliceStart s
def maxSlice : ℕ := (List.range G.slices).foldl (fun m s => max m (G.sliceLen s)) 0
def physWidth : ℕ := G.maxSlice * G.blockWidth

end SetupGroup

/-- A group with the start of its unit in the witness, its consistency row and its first column
of `D`. -/
structure Placed where
  G : SetupGroup
  z : ℕ
  row : ℕ
  col : ℕ

namespace SetupPlan

variable (P : SetupPlan)

def placed : List Placed :=
  (P.groups.foldl (fun (acc : Array Placed × ℕ × ℕ × ℕ) G =>
    (acc.1.push ⟨G, acc.2.1, acc.2.2.1, acc.2.2.2⟩, acc.2.1 + G.unitLen, acc.2.2.1 + G.rows,
      acc.2.2.2 + G.dCols P.openDegree)) (#[], 0, 0, 0)).1.toList

/-- The first row of `D`. -/
def openRow : ℕ := (P.groups.map SetupGroup.rows).sum

def dColsAll : ℕ := (P.groups.map (SetupGroup.dCols · P.openDegree)).sum

/-- The base ring of Stage 3: the least degree of `A`, `B` and `D` over the groups. -/
def base : ℕ := P.groups.foldl (fun m G => min m (min G.innerDegree G.outerDegree)) P.openDegree

end SetupPlan

/-- The part of the weight of Stage 3 of one group: the roles `A`, `B` and `D`, each without the
common factor `powerScale(α, ρ_low)`. `rel R` and `set R` are the sides of the relation point and
of the setup point for a role of `R` relation blocks and of `R` base rings. -/
def groupWeight (P : SetupPlan) (eqτ : ℕ → Ext) (rel set : ℕ → Side) (β : Ext) (r' : List Ext)
    (gadget : ℕ → Ext) (p : Placed) : Ext :=
  let G := p.G
  let aRel := rel (G.innerDegree / P.block)
  let bRel := rel (G.outerDegree / P.block)
  let dRel := rel (P.openDegree / P.block)
  let aSet := set (G.innerDegree / P.base)
  let bSet := set (G.outerDegree / P.base)
  let dSet := set (P.openDegree / P.base)
  let zBase := p.z / P.block
  let eBase := (p.z + G.zLen) / P.block
  let tBase := (p.z + G.zLen + G.eLen) / P.block
  let zw := (Array.range G.zCols).map fun c => sumBelow G.foldDigits fun f =>
    -(gadget f * aRel.lane β r' (zBase + (c * G.foldDigits + f) * aRel.ratio))
  let a := aSet.proj * sumBelow G.innerRank fun i => eqτ (p.row + 1 + i) *
    sumBelow G.zCols fun c => zw.getD c Ext.zero * aSet.high.eval (i * G.zCols + c)
  let tw := (Array.range (G.blocks * G.blockWidth)).map fun L =>
    bRel.lane β r' (tBase + L * bRel.ratio)
  let ends := (Array.range G.slices).map fun s => G.sliceLen s * G.blockWidth
  let starts := (Array.range G.slices).map fun s => G.sliceStart s * G.blockWidth
  let b := bSet.proj * sumBelow G.outerRank fun j =>
    let rows := (Array.range G.slices).map fun s =>
      eqτ (p.row + 1 + G.innerRank + s * G.outerRank + j)
    sumBelow G.physWidth fun x =>
      (sumBelow G.slices fun s => if x < ends.getD s 0 then
          rows.getD s Ext.zero * tw.getD (starts.getD s 0 + x) Ext.zero
        else Ext.zero) * bSet.high.eval (j * G.physWidth + x)
  let ew := (Array.range (G.dCols P.openDegree)).map fun L =>
    dRel.lane β r' (eBase + L * dRel.ratio)
  let d := dSet.proj * sumBelow P.openRank fun j => eqτ (P.openRow + j) *
    sumBelow (G.dCols P.openDegree) fun L =>
      ew.getD L Ext.zero * dSet.high.eval (j * P.dColsAll + p.col + L)
  a + b + d

/-- The weight of Stage 3 at `ρ` for the plan `P`, the point `α` of the ring reduction, the
batching point `τ_1` and the point `r_2` of the fused sum-check. -/
def stage3Weight (P : SetupPlan) (α : Ext) (τ r2 ρ : List Ext) : Ext :=
  let r' := r2.drop (Nat.log 2 P.block)
  let bits := Nat.log 2 P.base
  let ρidx := ρ.drop bits
  let β := powExt α P.block
  let β0 := powExt α P.base
  let gadget (f : ℕ) : Ext := powExt (Ext.ofBase (Fq.ofNat (2 ^ P.logBasis))) f
  let ratios := [1, 2, 4, 8, 16, 32, 64, 128]
  let rels := ratios.map (Side.of β r')
  let sets := ratios.map (Side.of β0 ρidx)
  let pick (sides : List Side) (b : Ext) (r : List Ext) (ratio : ℕ) : Side :=
    (sides.find? (·.ratio = ratio)).getD (Side.of b r ratio)
  let eqτ (i : ℕ) : Ext := eqIndex τ i
  powerScaleExt α (ρ.take bits) * (P.placed.map fun p =>
    groupWeight P eqτ (pick rels β r') (pick sets β0 ρidx) β r' gadget p).foldl (· + ·) Ext.zero

/-- The plan of a fold with groups `B` for the order `order` of the code, the relation block
`2^low` and the base `2^logBasis` of the opening digits. -/
def SetupPlan.ofBatch {n : ℕ} (B : Batch (Fin n)) (order : Equiv.Perm (Fin n)) (low logBasis : ℕ) :
    SetupPlan where
  groups := (List.finRange n).map fun k =>
    let g := order k
    let S := B.shape g
    { innerDegree := S.innerDegree, outerDegree := S.outerDegree, innerRank := S.innerRank
      positions := S.blockLength, witnessDigits := S.sourceDigits
      foldDigits := (B.fold g).responseDigits, commitDigits := S.openingDigits
      openDigits := S.openingDigits, blocks := S.blocks, slices := S.slices
      outerRank := S.outerRank, openWidth := B.planes g * B.carrierDegree g }
  openDegree := B.openDegree
  openRank := B.openRank
  block := 2 ^ low
  logBasis

theorem eqTable_size : ∀ r : List Ext, (eqTable r).size = 2 ^ r.length
  | [] => rfl
  | c :: rest => by
      simp only [eqTable, Array.size_map, Array.size_range, eqTable_size rest, List.length_cons,
        pow_succ]
      ring

theorem eqTable_getD : ∀ (r : List Ext) (i : ℕ), i < 2 ^ r.length →
    (eqTable r).getD i Ext.zero = eqIndex r i
  | [], i, small => by
      have zero : i = 0 := by simpa using small
      subst zero
      rfl
  | c :: rest, i, small => by
      have inside : i < (eqTable (c :: rest)).size := by rwa [eqTable_size]
      have half : i / 2 < 2 ^ rest.length := by
        rw [List.length_cons, pow_succ] at small
        omega
      rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem inside, Option.getD_some]
      simp only [eqTable, Array.getElem_map, Array.getElem_range, eqIndex,
        eqTable_getD rest (i / 2) half]

/-- `ẽq(r, i)` reads only the low `|r|` bits of `i`. -/
theorem eqIndex_mod : ∀ (r : List Ext) (i : ℕ), eqIndex r (i % 2 ^ r.length) = eqIndex r i
  | [], _ => rfl
  | c :: rest, i => by
      simp only [eqIndex, List.length_cons, pow_succ', Nat.mod_mul_right_mod,
        Nat.mod_mul_right_div_self, eqIndex_mod rest (i / 2)]

section Spec

variable {E : Type} [CommRing E] [Algebra Zq E] [Degree2 E]

theorem eqIndex_append_into : ∀ (a b : List Ext) (i : ℕ),
    ((eqIndex (a ++ b) i).into : E) =
      (eqIndex a (i % 2 ^ a.length)).into * (eqIndex b (i / 2 ^ a.length)).into
  | [], b, i => by simp [eqIndex, Ext.into_one]
  | c :: a, b, i => by
      simp only [List.cons_append, eqIndex, List.length_cons, pow_succ', Nat.mod_mul_right_mod,
        Nat.mod_mul_right_div_self, Ext.into_mul, eqIndex_append_into a b (i / 2),
        Nat.div_div_eq_div_mul]
      ring

/-- The split table of the code is `ẽq(r, ·)` of the specification. -/
theorem EqSplit.eval_into (r : List Ext) (i : ℕ) :
    (((EqSplit.of r).eval i).into : E) = eqAt (r.map Ext.into) i := by
  have short : (r.take (r.length / 2)).length = r.length / 2 := by
    rw [List.length_take]
    omega
  have low : (eqTable (r.take (r.length / 2))).getD (i % 2 ^ (r.length / 2)) Ext.zero =
      eqIndex (r.take (r.length / 2)) (i % 2 ^ (r.length / 2)) :=
    eqTable_getD _ _ (by rw [short]; exact Nat.mod_lt _ (by positivity))
  have high : (eqTable (r.drop (r.length / 2))).getD
      (i / 2 ^ (r.length / 2) % (eqTable (r.drop (r.length / 2))).size) Ext.zero =
      eqIndex (r.drop (r.length / 2)) (i / 2 ^ (r.length / 2)) := by
    rw [eqTable_size, eqTable_getD _ _ (Nat.mod_lt _ (by positivity)), eqIndex_mod]
  simp only [EqSplit.of, EqSplit.eval, low, high, Ext.into_mul]
  rw [← eqIndex_into, ← List.take_append_drop (r.length / 2) r, eqIndex_append_into, short,
    List.take_append_drop]

/-- `powerScaleExt` is `powerScale` of the specification (`Extract/SetupFactor.lean`). -/
theorem powerScaleExt_into (b : Ext) (c : List Ext) :
    ((powerScaleExt b c).into : E) = powerScale (b.into : E) (c.map Ext.into) := by
  induction c generalizing b with
  | nil => simp [powerScaleExt, powerScale, Ext.into_one]
  | cons x rest ih =>
      simp only [powerScaleExt, powerScale, List.map_cons, Ext.into_mul, Ext.into_add,
        Ext.into_sub, Ext.into_one, ih]

end Spec

end NightstreamFPrime.Spec.Compression.Akita.Replay
