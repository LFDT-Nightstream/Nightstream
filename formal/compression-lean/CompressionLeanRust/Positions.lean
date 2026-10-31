import CompressionLeanRust.Layout
import CompressionLean.Ports.Circuit.Block

/-! The translated positions of `neo_verified::layout` equal the copy in
`CompressionLean/Ports/Shrink/Layout.lean` and `Ports/Shrink/BlockPart.lean`, for the template
of the one block kind, `Circuit.Block.poseidonTemplate`. -/

namespace CompressionLeanRust

open Aeneas Aeneas.Std Result
open CompressionLean.Shrink (wide narrow slot part partsEnd variablesFor Counts Part)
open neo_verified
open CompressionLean.Circuit.Block (poseidonTemplate)
open CompressionLean.Shrink.Layout (blockRow blockCell glueRow rowVars cellVars one column entryColumn)

/-! ## Steps of the translated code -/

theorem bind_eq_ok {α β : Type} {m : Result α} {k : α → Result β} {y : β} (h : Std.bind m k = ok y) :
    ∃ a, m = ok a ∧ k a = ok y := by
  cases m with
  | ret a => exact ⟨a, rfl, by simpa only [bind_ok] using h⟩
  | vis e k' => simp at h
  | div => simp at h

theorem unwrap_eq_ok {T : Type} {o : Option T} {v : T} (h : core.option.Option.unwrap o = ok v) :
    o = some v := by
  cases o with
  | none => exact absurd h fail_not_ok
  | some a => rw [unwrap_some, Result.ok.injEq] at h; rw [h]

theorem add_eq_ok {a b c : Usize} (h : core.option.Option.unwrap (Usize.checked_add a b) = ok c) :
    c.val = a.val + b.val := by
  have := Usize.checked_add_bv_spec a b
  rw [unwrap_eq_ok h] at this
  exact this.2.1

theorem add_ok (a b : Usize) (h : a.val + b.val ≤ Usize.max) :
    ∃ c, core.option.Option.unwrap (Usize.checked_add a b) = ok c ∧ c.val = a.val + b.val := by
  have := Usize.checked_add_bv_spec a b
  split at this
  · rename_i c hc; exact ⟨c, by rw [hc, unwrap_some], this.2.1⟩
  · omega

theorem mul_eq_ok {a b c : Usize} (h : core.option.Option.unwrap (Usize.checked_mul a b) = ok c) :
    c.val = a.val * b.val := by
  have := Usize.checked_mul_bv_spec a b
  rw [unwrap_eq_ok h] at this
  exact this.2.1

theorem mul_ok (a b : Usize) (h : a.val * b.val ≤ Usize.max) :
    ∃ c, core.option.Option.unwrap (Usize.checked_mul a b) = ok c ∧ c.val = a.val * b.val := by
  have := Usize.checked_mul_bv_spec a b
  split at this
  · rename_i c hc; exact ⟨c, by rw [hc, unwrap_some], this.2.1⟩
  · omega

theorem sub_eq_ok {ty : UScalarTy} {a b c : UScalar ty}
    (h : core.option.Option.unwrap (core.num.checked_sub_UScalar a b) = ok c) :
    c.val = a.val - b.val ∧ b.val ≤ a.val := by
  have := core.num.checked_sub_UScalar_bv_spec a b
  rw [unwrap_eq_ok h] at this
  exact ⟨this.2.1, this.1⟩

theorem index_eq_ok {α : Type} {n : Usize} {v : Aeneas.Std.Array α n} {i : Usize} {a : α}
    (h : Array.index_usize v i = ok a) : v.val[i.val]? = some a := by
  simp only [Array.index_usize] at h
  split at h
  · exact absurd h fail_not_ok
  · rename_i x hx; rw [Result.ok.injEq] at h; subst h; simpa using hx

theorem index_ok {α : Type} {n : Usize} (v : Aeneas.Std.Array α n) (i : Usize) (a : α)
    (h : v.val[i.val]? = some a) : Array.index_usize v i = ok a := by
  simp only [Array.index_usize]
  split
  · rename_i hx; simp_all
  · rename_i x hx; simp_all

theorem lift_ok {α : Type} (x : α) : lift x = ok x := rfl

/-! ## `place` and `slot` -/

theorem split_parts {size : Usize} {a : Aeneas.Std.Array Usize 2#usize} (h : layout.split size = ok a) :
    ∃ w n, a.val = [w, n] ∧ w.val = wide size.val ∧ n.val = narrow size.val := by
  have hv := split_sound size a h
  rcases hl : a.val with _ | ⟨w, _ | ⟨n, _ | ⟨m, t⟩⟩⟩ <;>
    simp only [usizeValues, hl, List.map_nil, List.map_cons, List.cons.injEq, reduceCtorEq,
      and_false] at hv
  exact ⟨w, n, rfl, hv.1, hv.2.1⟩

theorem index_two {α : Type} {a : Aeneas.Std.Array α 2#usize} {x y : α} (h : a.val = [x, y]) :
    Array.index_usize a 0#usize = ok x ∧ Array.index_usize a 1#usize = ok y :=
  ⟨index_ok a 0#usize x (by simp [h]), index_ok a 1#usize y (by simp [h])⟩

theorem place_eq_ok {size count : Usize} {r : Aeneas.Std.Array layout.Part 2#usize × Usize}
    (h : layout.place size count = ok r) :
    placeValue r = ([part size.val count.val 0, part size.val count.val 1], partsEnd size.val count.val) := by
  unfold layout.place at h
  obtain ⟨a, ha, h⟩ := bind_eq_ok h
  obtain ⟨w, n, hl, hw, hn⟩ := split_parts ha
  simp only [(index_two hl).1, (index_two hl).2, bind_ok, lift_ok] at h
  obtain ⟨start, hs, h⟩ := bind_eq_ok h
  obtain ⟨i, hi, h⟩ := bind_eq_ok h
  obtain ⟨e, he, h⟩ := bind_eq_ok h
  rw [Result.ok.injEq] at h
  subst h
  have hs' := mul_eq_ok hs
  have hi' := mul_eq_ok hi
  have he' := add_eq_ok he
  simp [placeValue, Aeneas.Std.Array.make, partValue, part, partsEnd, hw, hn, hs', hi', he']

theorem place_sound :
    Sound (fun x : Usize × Usize => layout.place x.1 x.2) placeValue
      (fun x => ([part x.1.val x.2.val 0, part x.1.val x.2.val 1], partsEnd x.1.val x.2.val)) :=
  fun _ _ h => place_eq_ok h

theorem place_succeeds (size count : Usize) (h0 : 0 < size.val)
    (h : partsEnd size.val count.val ≤ Usize.max) : ∃ r, layout.place size count = ok r := by
  obtain ⟨a, ha, -⟩ := split_ok size h0
  obtain ⟨w, n, hl, hw, hn⟩ := split_parts ha
  unfold partsEnd at h
  obtain ⟨s, hs, hsv⟩ := mul_ok w count (by rw [hw]; omega)
  obtain ⟨i, hi, hiv⟩ := mul_ok n count (by rw [hn]; omega)
  obtain ⟨e, he, -⟩ := add_ok s i (by rw [hsv, hiv, hw, hn]; omega)
  exact ⟨_, by
    simp only [layout.place, ha, (index_two hl).1, (index_two hl).2, lift_ok, bind_ok, hs, hi, he]; rfl⟩

theorem place_complete :
    Complete (fun x : Usize × Usize => 0 < x.1.val ∧ partsEnd x.1.val x.2.val ≤ Usize.max)
      (fun x : Usize × Usize => layout.place x.1 x.2) placeValue
      (fun x => ([part x.1.val x.2.val 0, part x.1.val x.2.val 1], partsEnd x.1.val x.2.val)) :=
  complete_of place_sound ⟨(1#usize, 0#usize), by decide, by simp [partsEnd]⟩
    fun x hx => place_succeeds x.1 x.2 hx.1 hx.2

theorem place_parts {size count e : Usize} {parts : Aeneas.Std.Array layout.Part 2#usize}
    (h : layout.place size count = ok (parts, e)) :
    ∃ p q : layout.Part, parts.val = [p, q] ∧ p.start.val = 0 ∧ p.stride.val = wide size.val ∧
      q.start.val = wide size.val * count.val ∧ q.stride.val = narrow size.val ∧
      e.val = partsEnd size.val count.val := by
  have hv := place_eq_ok h
  simp only [placeValue, Prod.mk.injEq] at hv
  obtain ⟨hl, he⟩ := hv
  rcases hp : parts.val with _ | ⟨p, _ | ⟨q, _ | ⟨m, t⟩⟩⟩ <;>
    simp only [hp, List.map_nil, List.map_cons, List.cons.injEq, reduceCtorEq, and_false] at hl
  simp only [partValue, part, CompressionLean.Shrink.Part.mk.injEq, ↓reduceIte, one_ne_zero] at hl
  exact ⟨p, q, rfl, hl.1.1, hl.1.2, hl.2.1.1, hl.2.1.2, he⟩

theorem slot_eq_ok {size count b k y : Usize} (h : layout.slot size count b k = ok y) :
    y.val = slot size.val count.val b.val k.val := by
  unfold layout.slot at h
  obtain ⟨⟨parts, e⟩, hr, h⟩ := bind_eq_ok h
  obtain ⟨p, q, hl, hp0, hpw, hq0, hqn, -⟩ := place_parts hr
  simp only [uncurry_apply_pair, (index_two hl).1, (index_two hl).2, bind_ok, lift_ok] at h
  unfold slot
  by_cases hk : k < p.stride
  · have hk' : k.val < wide size.val := by rw [← hpw]; scalar_tac
    simp only [hk, ↓reduceIte, bind_ok, uncurry_apply_pair] at h
    obtain ⟨m, hm, h⟩ := bind_eq_ok h
    obtain ⟨a, ha, h⟩ := bind_eq_ok h
    simp only [hk', ↓reduceIte]
    rw [add_eq_ok h, add_eq_ok ha, mul_eq_ok hm, hp0, hpw]
    ring
  · have hk' : ¬ k.val < wide size.val := by rw [← hpw]; scalar_tac
    simp only [hk, ↓reduceIte] at h
    obtain ⟨⟨part', l⟩, hs, h⟩ := bind_eq_ok h
    obtain ⟨d, hd, hs⟩ := bind_eq_ok hs
    rw [Result.ok.injEq, Prod.mk.injEq] at hs
    obtain ⟨rfl, rfl⟩ := hs
    simp only [uncurry_apply_pair] at h
    obtain ⟨m, hm, h⟩ := bind_eq_ok h
    obtain ⟨a, ha, h⟩ := bind_eq_ok h
    simp only [hk', ↓reduceIte]
    rw [add_eq_ok h, add_eq_ok ha, mul_eq_ok hm, hq0, hqn, (sub_eq_ok hd).1, hpw]
    ring

theorem slot_sound :
    Sound (fun x : Usize × Usize × Usize × Usize => layout.slot x.1 x.2.1 x.2.2.1 x.2.2.2) usizeValue
      (fun x => slot x.1.val x.2.1.val x.2.2.1.val x.2.2.2.val) :=
  fun _ _ h => slot_eq_ok h

theorem slot_succeeds (size count b k : Usize) (h0 : 0 < size.val)
    (hend : partsEnd size.val count.val ≤ Usize.max)
    (h : slot size.val count.val b.val k.val ≤ Usize.max) : ∃ y, layout.slot size count b k = ok y := by
  obtain ⟨⟨parts, e⟩, hr⟩ := place_succeeds size count h0 hend
  obtain ⟨p, q, hl, hp0, hpw, hq0, hqn, -⟩ := place_parts hr
  unfold slot at h
  by_cases hk : k < p.stride
  · have hk' : k.val < wide size.val := by rw [← hpw]; scalar_tac
    simp only [hk', ↓reduceIte] at h
    obtain ⟨m, hm, hmv⟩ := mul_ok b p.stride (by rw [hpw]; nlinarith)
    obtain ⟨a, ha, hav⟩ := add_ok p.start m (by rw [hp0, hmv, hpw]; nlinarith)
    obtain ⟨c, hc, -⟩ := add_ok a k (by rw [hav, hp0, hmv, hpw]; nlinarith)
    exact ⟨c, by simp only [layout.slot, hr, uncurry_apply_pair, (index_two hl).1, (index_two hl).2,
      bind_ok, lift_ok, hk, ↓reduceIte, hm, ha, hc]⟩
  · have hk' : ¬ k.val < wide size.val := by rw [← hpw]; scalar_tac
    simp only [hk', ↓reduceIte] at h
    obtain ⟨d, hd, hdv⟩ := checked_sub_some k p.stride (by rw [hpw]; omega)
    obtain ⟨m, hm, hmv⟩ := mul_ok b q.stride (by rw [hqn]; nlinarith)
    obtain ⟨a, ha, hav⟩ := add_ok q.start m (by rw [hq0, hmv, hqn]; nlinarith)
    obtain ⟨c, hc, -⟩ := add_ok a d (by rw [hav, hq0, hmv, hqn, hdv, hpw]; nlinarith)
    exact ⟨c, by simp only [layout.slot, hr, uncurry_apply_pair, (index_two hl).1, (index_two hl).2,
      bind_ok, lift_ok, hk, ↓reduceIte, Usize.checked_sub, hd, unwrap_some, hm, ha, hc]⟩

theorem slot_complete :
    Complete
      (fun x : Usize × Usize × Usize × Usize => 0 < x.1.val ∧ partsEnd x.1.val x.2.1.val ≤ Usize.max ∧
        slot x.1.val x.2.1.val x.2.2.1.val x.2.2.2.val ≤ Usize.max)
      (fun x : Usize × Usize × Usize × Usize => layout.slot x.1 x.2.1 x.2.2.1 x.2.2.2) usizeValue
      (fun x => slot x.1.val x.2.1.val x.2.2.1.val x.2.2.2.val) :=
  complete_of slot_sound ⟨(1#usize, 0#usize, 0#usize, 0#usize), by decide, by simp [partsEnd],
      by simp [slot]⟩
    fun x hx => slot_succeeds x.1 x.2.1 x.2.2.1 x.2.2.2 hx.1 hx.2.1 hx.2.2

/-! ## The positions of a shape -/

/-- The block count of a translated shape. The copy has one block kind. -/
def blockCount (s : layout.Shape) : Usize := s.blocks.val.headD 0#usize

/-- The copy's counts of a translated shape. -/
def counts (s : layout.Shape) : Counts :=
  ⟨(blockCount s).val, s.glue_rows.val, s.glue_cells.val, s.publics.val⟩

/-- The copy's wire of a translated wire. The copy has one block kind, so `.block` drops it. -/
def copyWire : circuit.Wire → CompressionLean.Shrink.Wire
  | .Glue c => .glue c.val
  | .Block _ b c => .block b.val c.val
  | .Public i => .pub i.val

/-- The ends of a shape's row parts and glue rows fit in a `usize`. Production meets it: the prover
indexes tables of `2 ^ rowVars` rows with these positions. -/
def RowsFit (s : layout.Shape) : Prop :=
  partsEnd poseidonTemplate.rows (counts s).blocks + (counts s).glueRows ≤ Usize.max

/-- The ends of a shape's cell parts and glue cells, and the constant one and the statement words
after it, fit in a `usize`. Production meets it: the prover indexes `z`, of `2 ^ (cellVars + 1)`
cells, with these positions. -/
def CellsFit (s : layout.Shape) : Prop :=
  partsEnd poseidonTemplate.cells (counts s).blocks + (counts s).glueCells ≤ Usize.max ∧
    (counts s).publics + 1 ≤ Usize.max ∧ one poseidonTemplate (counts s) ≤ Usize.max

/-- A shape with no block, row, cell or statement word. -/
def emptyShape : layout.Shape := ⟨Aeneas.Std.Array.make 1#usize [0#usize], 0#usize, 0#usize, 0#usize⟩

theorem template_rows : poseidonTemplate.rows = 166 := rfl

theorem template_cells : poseidonTemplate.cells = 182 := rfl

theorem max_ge : U32.max ≤ Usize.max := by
  rcases Usize.bounds_eq with h | h <;> simp [h, U32.max_eq, U64.max_eq]

theorem add_eq (x y : Usize) (h : x.val + y.val ≤ Usize.max) : ∃ z, x + y = ok z ∧ z.val = x.val + y.val := by
  have := UScalar.add_equiv x y
  split at this
  · rename_i z hz; exact ⟨z, Result.match.isOk.mp hz, this.2.1⟩
  · have hpos : 0 < 2 ^ UScalarTy.Usize.numBits := Nat.two_pow_pos _
    rw [← UScalar.max_USize_eq, UScalar.max] at h
    simp only [UScalar.inBounds] at this
    omega
  · exact this.elim

theorem rows_eq (kind : circuit.Kind) : circuit.Kind.rows kind = ok 166#usize := by
  cases kind
  have := max_ge
  rw [U32.max_eq] at this
  obtain ⟨z, hz, hv⟩ := add_eq 150#usize 16#usize (by simp; omega)
  simp only [circuit.Kind.rows, circuit.ROWS, circuit.SBOXES, circuit.WIDTH, hz]
  rw [UScalar.val_eq_imp z 166#usize (by simp [hv])]

theorem cells_eq (kind : circuit.Kind) : circuit.Kind.cells kind = ok 182#usize := by
  cases kind
  have := max_ge
  rw [U32.max_eq] at this
  obtain ⟨z, hz, hv⟩ := add_eq 16#usize 150#usize (by simp; omega)
  obtain ⟨w, hw, hwv⟩ := add_eq z 16#usize (by simp [hv]; omega)
  simp only [circuit.Kind.cells, circuit.CELLS, circuit.SBOXES, circuit.WIDTH, hz, bind_ok, hw]
  rw [UScalar.val_eq_imp w 182#usize (by simp [hwv, hv])]

theorem blocks_index (s : layout.Shape) : Array.index_usize s.blocks 0#usize = ok (blockCount s) := by
  apply index_ok
  have hlen := s.blocks.property
  unfold blockCount
  rcases h : s.blocks.val with _ | ⟨c, t⟩
  · simp [h] at hlen
  · simp

theorem numBits_ge : 32 ≤ UScalarTy.Usize.numBits := by
  cases System.Platform.numBits_eq <;> simp_all

theorem cast_u32 (x : U32) : (UScalar.cast .Usize x).val = x.val := by
  rw [UScalar.cast_val_eq]
  apply Nat.mod_eq_of_lt
  have := x.hBounds
  have := Nat.pow_le_pow_right (show 0 < 2 by decide) numBits_ge
  simp only [UScalarTy.U32_numBits_eq] at *
  omega

theorem cast_u16 (x : U16) : (UScalar.cast .Usize x).val = x.val := by
  rw [UScalar.cast_val_eq]
  apply Nat.mod_eq_of_lt
  have := x.hBounds
  have := Nat.pow_le_pow_right (show 0 < 2 by decide) numBits_ge
  have := Nat.pow_le_pow_right (show 0 < 2 by decide) (show 16 ≤ 32 by decide)
  simp only [UScalarTy.U16_numBits_eq] at *
  omega

theorem empty_counts : counts emptyShape = ⟨0, 0, 0, 0⟩ := rfl

theorem empty_rows_fit : RowsFit emptyShape := by
  simp [RowsFit, empty_counts, partsEnd]

theorem empty_cells_fit : CellsFit emptyShape := by
  have := max_ge
  rw [U32.max_eq] at this
  have hv : cellVars poseidonTemplate (counts emptyShape) = 1 := by
    simp [cellVars, empty_counts, partsEnd, variablesFor]
  refine ⟨by simp [empty_counts, partsEnd], by simp [empty_counts]; omega, ?_⟩
  rw [one, hv]
  omega

theorem two_pow_eq_ok {k : U32} {p : Usize} (h : layout.two_pow k = ok p) : p.val = 2 ^ k.val := by
  have h0 := h
  unfold layout.two_pow at h
  simp only [lift_ok, bind_ok] at h
  obtain ⟨a, ha, h⟩ := bind_eq_ok h
  obtain ⟨c, hc, -⟩ := bind_eq_ok h
  have ha' := sub_eq_ok ha
  have hc' := sub_eq_ok hc
  have := bits_val
  have h1 : (1#u32).val = 1 := rfl
  obtain ⟨q, hq, hqv⟩ := two_pow_eq k (by omega)
  rw [hq, Result.ok.injEq] at h0
  rw [← h0, hqv]

theorem variables_le (m : Usize) : variablesFor m.val ≤ UScalarTy.Usize.numBits := by
  have := Nat.clog_le_of_le_pow m.hBounds.le
  have := numBits_pos
  unfold variablesFor
  omega

theorem block_row_eq (s : layout.Shape) (kind : circuit.Kind) (b k : Usize) :
    layout.block_row s kind b k = layout.slot 166#usize (blockCount s) b k := by
  simp only [layout.block_row, blocks_index, rows_eq, bind_ok]

theorem block_cell_eq (s : layout.Shape) (kind : circuit.Kind) (b k : Usize) :
    layout.block_cell s kind b k = layout.slot 182#usize (blockCount s) b k := by
  simp only [layout.block_cell, blocks_index, cells_eq, bind_ok]

theorem block_row_sound :
    Sound (fun x : layout.Shape × circuit.Kind × Usize × Usize => layout.block_row x.1 x.2.1 x.2.2.1 x.2.2.2)
      usizeValue (fun x => blockRow poseidonTemplate (counts x.1) x.2.2.1.val x.2.2.2.val) := by
  rintro ⟨s, kind, b, k⟩ y h
  dsimp only at h
  rw [block_row_eq] at h
  simp [usizeValue, slot_eq_ok h, blockRow, template_rows, counts]

theorem block_row_complete :
    Complete
      (fun x : layout.Shape × circuit.Kind × Usize × Usize =>
        RowsFit x.1 ∧ blockRow poseidonTemplate (counts x.1) x.2.2.1.val x.2.2.2.val ≤ Usize.max)
      (fun x : layout.Shape × circuit.Kind × Usize × Usize => layout.block_row x.1 x.2.1 x.2.2.1 x.2.2.2)
      usizeValue (fun x => blockRow poseidonTemplate (counts x.1) x.2.2.1.val x.2.2.2.val) := by
  refine complete_of block_row_sound ⟨(emptyShape, .Permutation, 0#usize, 0#usize), ?_⟩ ?_
  · simp [empty_rows_fit, blockRow, slot, template_rows, wide]
  · rintro ⟨s, kind, b, k⟩ ⟨hfit, h⟩
    dsimp only at hfit h ⊢
    rw [block_row_eq]
    simp only [RowsFit, blockRow, template_rows, counts] at hfit h
    exact slot_succeeds _ _ b k (by simp) (by simp; omega) (by simpa using h)

theorem block_cell_sound :
    Sound (fun x : layout.Shape × circuit.Kind × Usize × Usize => layout.block_cell x.1 x.2.1 x.2.2.1 x.2.2.2)
      usizeValue (fun x => blockCell poseidonTemplate (counts x.1) x.2.2.1.val x.2.2.2.val) := by
  rintro ⟨s, kind, b, k⟩ y h
  dsimp only at h
  rw [block_cell_eq] at h
  simp [usizeValue, slot_eq_ok h, blockCell, template_cells, counts]

theorem block_cell_succeeds (s : layout.Shape) (kind : circuit.Kind) (b k : Usize)
    (hend : partsEnd poseidonTemplate.cells (counts s).blocks ≤ Usize.max)
    (h : blockCell poseidonTemplate (counts s) b.val k.val ≤ Usize.max) :
    ∃ y, layout.block_cell s kind b k = ok y := by
  rw [block_cell_eq]
  simp only [blockCell, template_cells, counts] at hend h
  exact slot_succeeds _ _ b k (by simp) (by simpa using hend) (by simpa using h)

theorem block_cell_complete :
    Complete
      (fun x : layout.Shape × circuit.Kind × Usize × Usize =>
        CellsFit x.1 ∧ blockCell poseidonTemplate (counts x.1) x.2.2.1.val x.2.2.2.val ≤ Usize.max)
      (fun x : layout.Shape × circuit.Kind × Usize × Usize => layout.block_cell x.1 x.2.1 x.2.2.1 x.2.2.2)
      usizeValue (fun x => blockCell poseidonTemplate (counts x.1) x.2.2.1.val x.2.2.2.val) := by
  refine complete_of block_cell_sound ⟨(emptyShape, .Permutation, 0#usize, 0#usize), empty_cells_fit, ?_⟩ ?_
  · simp [blockCell, slot, template_cells, wide]
  · rintro ⟨s, kind, b, k⟩ ⟨hfit, h⟩
    dsimp only at hfit h
    exact block_cell_succeeds s kind b k (by have := hfit.1; omega) h

theorem rows_end_eq_ok {s : layout.Shape} {v : Usize} (h : layout.rows_end s = ok v) :
    v.val = partsEnd poseidonTemplate.rows (counts s).blocks := by
  simp only [layout.rows_end, blocks_index, rows_eq, bind_ok] at h
  obtain ⟨⟨parts, e⟩, hr, h⟩ := bind_eq_ok h
  simp only [uncurry_apply_pair, Result.ok.injEq] at h
  subst h
  obtain ⟨-, -, -, -, -, -, -, he⟩ := place_parts hr
  simp [he, template_rows, counts]

theorem rows_end_succeeds (s : layout.Shape) (h : partsEnd poseidonTemplate.rows (counts s).blocks ≤ Usize.max) :
    ∃ v, layout.rows_end s = ok v := by
  simp only [template_rows, counts] at h
  obtain ⟨⟨parts, e⟩, hr⟩ := place_succeeds 166#usize (blockCount s) (by simp) (by simpa using h)
  exact ⟨e, by simp only [layout.rows_end, blocks_index, rows_eq, bind_ok, hr, uncurry_apply_pair]⟩

theorem cells_end_eq_ok {s : layout.Shape} {v : Usize} (h : layout.cells_end s = ok v) :
    v.val = partsEnd poseidonTemplate.cells (counts s).blocks := by
  simp only [layout.cells_end, blocks_index, cells_eq, bind_ok] at h
  obtain ⟨⟨parts, e⟩, hr, h⟩ := bind_eq_ok h
  simp only [uncurry_apply_pair, Result.ok.injEq] at h
  subst h
  obtain ⟨-, -, -, -, -, -, -, he⟩ := place_parts hr
  simp [he, template_cells, counts]

theorem cells_end_succeeds (s : layout.Shape)
    (h : partsEnd poseidonTemplate.cells (counts s).blocks ≤ Usize.max) : ∃ v, layout.cells_end s = ok v := by
  simp only [template_cells, counts] at h
  obtain ⟨⟨parts, e⟩, hr⟩ := place_succeeds 182#usize (blockCount s) (by simp) (by simpa using h)
  exact ⟨e, by simp only [layout.cells_end, blocks_index, cells_eq, bind_ok, hr, uncurry_apply_pair]⟩

theorem glue_row_sound :
    Sound (fun x : layout.Shape × Usize => layout.glue_row x.1 x.2) usizeValue
      (fun x => glueRow poseidonTemplate (counts x.1) x.2.val) := by
  rintro ⟨s, g⟩ y h
  dsimp only at h
  unfold layout.glue_row at h
  obtain ⟨i, hi, h⟩ := bind_eq_ok h
  simp only [lift_ok, bind_ok] at h
  rw [usizeValue, add_eq_ok h, rows_end_eq_ok hi]
  rfl

theorem glue_row_complete :
    Complete
      (fun x : layout.Shape × Usize => RowsFit x.1 ∧ glueRow poseidonTemplate (counts x.1) x.2.val ≤ Usize.max)
      (fun x : layout.Shape × Usize => layout.glue_row x.1 x.2) usizeValue
      (fun x => glueRow poseidonTemplate (counts x.1) x.2.val) := by
  refine complete_of glue_row_sound ⟨(emptyShape, 0#usize), empty_rows_fit, ?_⟩ ?_
  · simp [glueRow, empty_counts, partsEnd]
  · rintro ⟨s, g⟩ ⟨-, h⟩
    dsimp only at h ⊢
    unfold glueRow at h
    obtain ⟨i, hi⟩ := rows_end_succeeds s (by omega)
    obtain ⟨c, hc, -⟩ := add_ok i g (by rw [rows_end_eq_ok hi]; omega)
    exact ⟨c, by simp only [layout.glue_row, hi, bind_ok, lift_ok, hc]⟩

theorem row_variables_sound :
    Sound layout.row_variables usizeValue (fun s => rowVars poseidonTemplate (counts s)) := by
  intro s y h
  unfold layout.row_variables at h
  obtain ⟨i, hi, h⟩ := bind_eq_ok h
  simp only [lift_ok, bind_ok] at h
  obtain ⟨j, hj, h⟩ := bind_eq_ok h
  have hv : y.val = variablesFor j.val := variables_sound j y h
  show y.val = _
  rw [hv, add_eq_ok hj, rows_end_eq_ok hi]
  rfl

theorem row_variables_complete :
    Complete RowsFit layout.row_variables usizeValue (fun s => rowVars poseidonTemplate (counts s)) := by
  refine complete_of row_variables_sound ⟨emptyShape, empty_rows_fit⟩ ?_
  intro s h
  unfold RowsFit at h
  obtain ⟨i, hi⟩ := rows_end_succeeds s (by omega)
  obtain ⟨c, hc, -⟩ := add_ok i s.glue_rows (by rw [rows_end_eq_ok hi]; exact h)
  obtain ⟨v, hv, -⟩ := variables_ok c
  exact ⟨v, by simp only [layout.row_variables, hi, bind_ok, lift_ok, hc, hv]⟩

theorem cell_variables_eq_ok {s : layout.Shape} {y : Usize} (h : layout.cell_variables s = ok y) :
    y.val = cellVars poseidonTemplate (counts s) ∧ y.val ≤ UScalarTy.Usize.numBits := by
  unfold layout.cell_variables at h
  obtain ⟨i, hi, h⟩ := bind_eq_ok h
  simp only [lift_ok, bind_ok] at h
  obtain ⟨c, hc, h⟩ := bind_eq_ok h
  obtain ⟨p, hp, h⟩ := bind_eq_ok h
  have hv : y.val = variablesFor (core.cmp.impls.OrdUsize.max c p).val := variables_sound _ y h
  refine ⟨?_, hv ▸ variables_le _⟩
  rw [hv, core.cmp.impls.OrdUsize.max_val, add_eq_ok hc, add_eq_ok hp, cells_end_eq_ok hi]
  rfl

theorem cell_variables_sound :
    Sound layout.cell_variables usizeValue (fun s => cellVars poseidonTemplate (counts s)) :=
  fun _ _ h => (cell_variables_eq_ok h).1

theorem cell_variables_succeeds (s : layout.Shape) (h : CellsFit s) : ∃ y, layout.cell_variables s = ok y := by
  obtain ⟨h1, h2, -⟩ := h
  obtain ⟨i, hi⟩ := cells_end_succeeds s (by omega)
  obtain ⟨c, hc, -⟩ := add_ok i s.glue_cells (by rw [cells_end_eq_ok hi]; exact h1)
  obtain ⟨p, hp, -⟩ := add_ok s.publics 1#usize (by simpa [counts] using h2)
  obtain ⟨v, hv, -⟩ := variables_ok (core.cmp.impls.OrdUsize.max c p)
  exact ⟨v, by simp only [layout.cell_variables, hi, bind_ok, lift_ok, hc, hp, hv]⟩

theorem cell_variables_complete :
    Complete CellsFit layout.cell_variables usizeValue (fun s => cellVars poseidonTemplate (counts s)) :=
  complete_of cell_variables_sound ⟨emptyShape, empty_cells_fit⟩ cell_variables_succeeds

theorem one_eq_ok {s : layout.Shape} {y : Usize} (h : layout.one s = ok y) :
    y.val = one poseidonTemplate (counts s) := by
  unfold layout.one at h
  obtain ⟨i, hi, h⟩ := bind_eq_ok h
  simp only [lift_ok, bind_ok] at h
  obtain ⟨hv, hle⟩ := cell_variables_eq_ok hi
  have := numBits_le
  rw [two_pow_eq_ok h, UScalar.cast_val_eq, Nat.mod_eq_of_lt (by simp only [UScalarTy.U32_numBits_eq]; omega),
    hv, one]

theorem one_sound : Sound layout.one usizeValue (fun s => one poseidonTemplate (counts s)) :=
  fun _ _ h => one_eq_ok h

theorem one_succeeds (s : layout.Shape) (h : CellsFit s) : ∃ y, layout.one s = ok y := by
  obtain ⟨i, hi⟩ := cell_variables_succeeds s h
  have hv := (cell_variables_eq_ok hi).1
  have hone := h.2.2
  rw [one, ← hv] at hone
  have hlt : i.val < UScalarTy.Usize.numBits := by
    have : 2 ^ i.val < 2 ^ UScalarTy.Usize.numBits := by
      rw [← UScalar.max_USize_eq, UScalar.max] at hone
      have := Nat.two_pow_pos UScalarTy.Usize.numBits
      omega
    exact (Nat.pow_lt_pow_iff_right (by decide)).mp this
  have hcast : (UScalar.cast .U32 i).val = i.val := by
    rw [UScalar.cast_val_eq]
    apply Nat.mod_eq_of_lt
    have := numBits_le
    simp only [UScalarTy.U32_numBits_eq]
    omega
  obtain ⟨p, hp, -⟩ := two_pow_eq (UScalar.cast .U32 i) (by rw [hcast]; exact hlt)
  exact ⟨p, by simp only [layout.one, hi, bind_ok, lift_ok, hp]⟩

theorem one_complete : Complete CellsFit layout.one usizeValue (fun s => one poseidonTemplate (counts s)) :=
  complete_of one_sound ⟨emptyShape, empty_cells_fit⟩ one_succeeds

theorem column_sound :
    Sound (fun x : layout.Shape × circuit.Wire => layout.column x.1 x.2) usizeValue
      (fun x => column poseidonTemplate (counts x.1) (copyWire x.2)) := by
  rintro ⟨s, w⟩ y h
  dsimp only at h ⊢
  cases w with
  | Glue c =>
    simp only [layout.column] at h
    obtain ⟨i, hi, h⟩ := bind_eq_ok h
    simp only [lift_ok, bind_ok] at h
    rw [usizeValue, add_eq_ok h, cells_end_eq_ok hi, cast_u32, copyWire, column]
  | Block kind b c =>
    simp only [layout.column, lift_ok, bind_ok] at h
    have := block_cell_sound (s, kind, UScalar.cast .Usize b, UScalar.cast .Usize c) y h
    simp only [cast_u32, cast_u16] at this
    rw [this, copyWire, column]
  | Public i =>
    simp only [layout.column] at h
    obtain ⟨o, ho, h⟩ := bind_eq_ok h
    simp only [lift_ok, bind_ok] at h
    rw [usizeValue, add_eq_ok h, one_eq_ok ho, cast_u32, copyWire, column]

theorem column_complete :
    Complete
      (fun x : layout.Shape × circuit.Wire =>
        CellsFit x.1 ∧ column poseidonTemplate (counts x.1) (copyWire x.2) ≤ Usize.max)
      (fun x : layout.Shape × circuit.Wire => layout.column x.1 x.2) usizeValue
      (fun x => column poseidonTemplate (counts x.1) (copyWire x.2)) := by
  refine complete_of column_sound ⟨(emptyShape, .Glue 0#u32), empty_cells_fit, ?_⟩ ?_
  · simp [column, copyWire, empty_counts, partsEnd]
  · rintro ⟨s, w⟩ ⟨hfit, h⟩
    dsimp only at hfit h ⊢
    cases w with
    | Glue c =>
      simp only [copyWire, column] at h
      obtain ⟨i, hi⟩ := cells_end_succeeds s (by omega)
      obtain ⟨d, hd, -⟩ := add_ok i (UScalar.cast .Usize c) (by rw [cells_end_eq_ok hi, cast_u32]; exact h)
      exact ⟨d, by simp only [layout.column, hi, bind_ok, lift_ok, hd]⟩
    | Block kind b c =>
      simp only [copyWire, column] at h
      obtain ⟨d, hd⟩ := block_cell_succeeds s kind (UScalar.cast .Usize b) (UScalar.cast .Usize c)
        (by have := hfit.1; omega) (by simpa only [cast_u32, cast_u16] using h)
      exact ⟨d, by simp only [layout.column, lift_ok, bind_ok, hd]⟩
    | Public i =>
      simp only [copyWire, column] at h
      obtain ⟨o, ho⟩ := one_succeeds s hfit
      obtain ⟨d, hd, -⟩ := add_ok o (UScalar.cast .Usize i) (by rw [one_eq_ok ho, cast_u32]; exact h)
      exact ⟨d, by simp only [layout.column, ho, bind_ok, lift_ok, hd]⟩

theorem entry_column_sound :
    Sound (fun x : layout.Shape × circuit.Kind × Usize × Option Usize => layout.entry_column x.1 x.2.1 x.2.2.1 x.2.2.2)
      usizeValue (fun x => entryColumn poseidonTemplate (counts x.1) x.2.2.1.val (x.2.2.2.map (·.val))) := by
  rintro ⟨s, kind, b, cell⟩ y h
  dsimp only at h ⊢
  cases cell with
  | none =>
    simp only [layout.entry_column] at h
    rw [usizeValue, one_eq_ok h, Option.map_none, entryColumn]
  | some c =>
    simp only [layout.entry_column] at h
    rw [block_cell_sound (s, kind, b, c) y h, Option.map_some, entryColumn]

theorem entry_column_complete :
    Complete
      (fun x : layout.Shape × circuit.Kind × Usize × Option Usize =>
        CellsFit x.1 ∧ entryColumn poseidonTemplate (counts x.1) x.2.2.1.val (x.2.2.2.map (·.val)) ≤ Usize.max)
      (fun x : layout.Shape × circuit.Kind × Usize × Option Usize => layout.entry_column x.1 x.2.1 x.2.2.1 x.2.2.2)
      usizeValue (fun x => entryColumn poseidonTemplate (counts x.1) x.2.2.1.val (x.2.2.2.map (·.val))) := by
  refine complete_of entry_column_sound ⟨(emptyShape, .Permutation, 0#usize, none), empty_cells_fit, ?_⟩ ?_
  · have := empty_cells_fit.2.2
    simpa [entryColumn] using this
  · rintro ⟨s, kind, b, cell⟩ ⟨hfit, h⟩
    dsimp only at hfit h ⊢
    cases cell with
    | none =>
      obtain ⟨o, ho⟩ := one_succeeds s hfit
      exact ⟨o, by simp only [layout.entry_column, ho]⟩
    | some c =>
      simp only [Option.map_some, entryColumn] at h
      obtain ⟨d, hd⟩ := block_cell_succeeds s kind b c (by have := hfit.1; omega) h
      exact ⟨d, by simp only [layout.entry_column, hd]⟩

end CompressionLeanRust
