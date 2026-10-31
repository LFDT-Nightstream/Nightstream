import CompressionLeanRust.Hash

/-! The translated `Duplex` of `neo_verified::hash`, run on `build`, equals the copy's `Duplex` in
`CompressionLean/Ports/Circuit/Hash.lean`. `duplexOf` reads a translated duplex as the copy's, with
its state, input and output as arrays; a copy runs as a transcript on that duplex and the builder.

Where the two differ, Rust fails and the copy returns, so `Sound` keeps no premise. Rust's `duplex`
writes the input into the state by index and fails on an input of more than 16 words, where the copy's
`set!` writes nothing (`duplex_loop0_returns`). `Vec::push` fails at `usize::MAX` words. The `Complete`
premises are these bounds: an input of at most 16 words for `duplex` and the samples, fewer than
`usize::MAX` words for `observe`, and fewer than `RATE` words for `observe_slice` and `observe_ext`,
which each `observe` keeps (`observe_fits`). Production meets all of them: `new` starts with an empty
input, `observe` duplexes when the input reaches `RATE` words, and `duplex` and `sample` empty it, so
between two calls the input holds fewer than `RATE` words. -/

namespace CompressionLeanRust

open Aeneas Aeneas.Std Result
open neo_verified
open CompressionLean
open CompressionLean.Circuit

abbrev RDuplex := hash.Duplex Builder Var ExtVar

/-- A translated duplex as the copy's: its state, input and output as arrays. -/
def duplexOf (d : RDuplex) : Duplex := ⟨d.state.val.toArray, d.input.val.toArray, d.output.val.toArray⟩

theorem duplexOf_injective : Function.Injective duplexOf := by
  rintro ⟨s1, i1, o1⟩ ⟨s2, i2, o2⟩ h
  simp only [duplexOf, Duplex.mk.injEq] at h
  obtain ⟨hs, hi, ho⟩ := h
  rw [Aeneas.Std.Array.ext s1 s2 (List.toArray_inj hs), vec_val_injective (List.toArray_inj hi),
    vec_val_injective (List.toArray_inj ho)]

/-- A run that returns the duplex, as the copy's run of `Duplex.new`. -/
def builtNew (x : RDuplex × Builder) : Duplex × Builder := (duplexOf x.1, x.2)

/-- A run that changes the duplex, as the copy's transcript run. -/
def builtStep (x : RDuplex × Builder) : (Unit × Duplex) × Builder := (((), duplexOf x.1), x.2)

/-- A run that returns a value and changes the duplex, as the copy's transcript run. -/
def builtSample {α : Type} (x : α × RDuplex × Builder) : (α × Duplex) × Builder := ((x.1, duplexOf x.2.1), x.2.2)

/-- A translated `Result<Vec<F>, Rejected>` as an `Except`. -/
def resultOf (r : core.result.Result (alloc.vec.Vec Var) backend.Rejected) : Except backend.Rejected (_root_.Array Var) :=
  match r with
  | .Ok v => .ok v.val.toArray
  | .Err e => .error e

/-- A run of `sample_bits`, as the copy's run with its bits as `Except.ok`. -/
def builtBits (x : core.result.Result (alloc.vec.Vec Var) backend.Rejected × RDuplex × Builder) :
    (Except backend.Rejected (_root_.Array Var) × Duplex) × Builder :=
  ((resultOf x.1, duplexOf x.2.1), x.2.2)

theorem builtNew_injective : Function.Injective builtNew := by
  rintro ⟨a, s⟩ ⟨b, t⟩ h
  simp only [builtNew, Prod.mk.injEq] at h
  rw [duplexOf_injective h.1, h.2]

theorem builtStep_injective : Function.Injective builtStep := by
  rintro ⟨a, s⟩ ⟨b, t⟩ h
  simp only [builtStep, Prod.mk.injEq] at h
  rw [duplexOf_injective h.1.2, h.2]

theorem builtSample_injective {α : Type} : Function.Injective (@builtSample α) := by
  rintro ⟨x, a, s⟩ ⟨y, b, t⟩ h
  simp only [builtSample, Prod.mk.injEq] at h
  rw [h.1.1, duplexOf_injective h.1.2, h.2]

theorem resultOf_injective : Function.Injective resultOf := by
  rintro (v | e) (w | f) h <;> simp only [resultOf, Except.ok.injEq, Except.error.injEq, reduceCtorEq] at h
  · rw [vec_val_injective (List.toArray_inj h)]
  · rw [h]

theorem builtBits_injective : Function.Injective builtBits := by
  rintro ⟨x, a, s⟩ ⟨y, b, t⟩ h
  simp only [builtBits, Prod.mk.injEq] at h
  rw [resultOf_injective h.1.1, duplexOf_injective h.1.2, h.2]

theorem vec_len_val {α : Type} (v : alloc.vec.Vec α) : (alloc.vec.Vec.len v).val = v.val.length :=
  alloc.vec.Vec.len_val v

theorem vec_lt (v : alloc.vec.Vec Var) (i : Usize) : i < alloc.vec.Vec.len v ↔ i.val < v.val.length := by
  rw [UScalar.lt_equiv, vec_len_val]

/-- Lanes `lo` to `11` of `st` set to `z`, as the copy's zero fill writes them. -/
def zeroFill (st : List Var) (lo : ℕ) (z : Var) : List Var :=
  (List.range' lo (12 - lo)).foldl (fun st lane => st.set lane z) st

theorem duplex_loop0_eq (state : Aeneas.Std.Array Var 16#usize) (input : alloc.vec.Vec Var)
    (h : input.val.length ≤ 16) :
    ∃ a, hash.Duplex.duplex_loop0 state input (alloc.vec.Vec.len input) 0#usize = ok a ∧
      a.val = fill state.val input.val := by
  unfold hash.Duplex.duplex_loop0
  have key := loop_total (fun (a : Aeneas.Std.Array Var 16#usize × Usize) =>
      hash.Duplex.duplex_loop0.body input (alloc.vec.Vec.len input) a.1 a.2)
    (fun n a => a.2.val ≤ input.val.length ∧ n = input.val.length - a.2.val ∧
      a.1.val = fill state.val (input.val.take a.2.val))
    (fun b => b.val = fill state.val input.val)
    (by
      rintro n ⟨st, lane⟩ ⟨hle, rfl, hst⟩
      dsimp only at hle hst ⊢
      by_cases hm : lane.val < input.val.length
      · right
        have hl1 := add_one_val lane (by scalar_tac)
        refine ⟨(setN st lane.val input.val[lane.val], core.num.Usize.wrapping_add lane 1#usize),
          input.val.length - (lane.val + 1), ?_, by omega, by dsimp only; omega, by dsimp only; omega, ?_⟩
        · simp only [hash.Duplex.duplex_loop0.body, (vec_lt input lane).mpr hm, ite_true, vec_index input lane hm,
            update_eq st lane _ (by scalar_tac), bind_ok, lift]
        · dsimp only
          rw [setN_val, hst, hl1, List.take_add_one, List.getElem?_eq_getElem hm, Option.toList_some, fill_snoc,
            List.length_take, Nat.min_eq_left (by omega)]
      · left
        refine ⟨st, ?_, ?_⟩
        · simp only [hash.Duplex.duplex_loop0.body, show ¬ lane < alloc.vec.Vec.len input by rw [vec_lt]; exact hm,
            ite_false]
        · rw [hst, List.take_of_length_le (by omega)])
    _ (state, 0#usize) ⟨by simp, rfl, by simp [fill]⟩
  exact key

/-- `duplex` returns only when the input fits the state. -/
theorem duplex_loop0_returns (state : Aeneas.Std.Array Var 16#usize) (input : alloc.vec.Vec Var)
    (a : Aeneas.Std.Array Var 16#usize)
    (h : hash.Duplex.duplex_loop0 state input (alloc.vec.Vec.len input) 0#usize = ok a) : input.val.length ≤ 16 := by
  unfold hash.Duplex.duplex_loop0 at h
  refine loop_sound (fun (a : Aeneas.Std.Array Var 16#usize × Usize) =>
      hash.Duplex.duplex_loop0.body input (alloc.vec.Vec.len input) a.1 a.2)
    (fun n a => a.2.val ≤ input.val.length ∧ a.2.val ≤ 16 ∧ n = input.val.length - a.2.val)
    (fun _ => input.val.length ≤ 16) ?_ _ (state, 0#usize) a ⟨by simp, by simp, rfl⟩ h
  rintro n ⟨st, lane⟩ c ⟨hle, h16, rfl⟩ hc
  dsimp only at hle h16 ⊢
  by_cases hm : lane.val < input.val.length
  · right
    by_cases hl : lane.val < 16
    · have hl1 := add_one_val lane (by scalar_tac)
      simp only [hash.Duplex.duplex_loop0.body, (vec_lt input lane).mpr hm, ite_true, vec_index input lane hm,
        update_eq st lane _ (by scalar_tac), bind_ok, lift, Result.ok.injEq] at hc
      exact ⟨_, input.val.length - (lane.val + 1), hc.symm, by omega, by dsimp only; omega, by dsimp only; omega,
        by dsimp only; omega⟩
    · have hu : Aeneas.Std.Array.update st lane input.val[lane.val] = fail .arrayOutOfBounds := by
        unfold Aeneas.Std.Array.update
        rw [Aeneas.Std.Array.getElem?_Usize_eq, List.getElem?_eq_none (by rw [st.property]; scalar_tac)]
      simp only [hash.Duplex.duplex_loop0.body, (vec_lt input lane).mpr hm, ite_true, vec_index input lane hm,
        hu, bind_ok] at hc
      simp at hc
  · left
    simp only [hash.Duplex.duplex_loop0.body, show ¬ lane < alloc.vec.Vec.len input by rw [vec_lt]; exact hm,
      ite_false, Result.ok.injEq] at hc
    exact ⟨st, hc.symm, by omega⟩

theorem duplex_loop1_eq (a : Aeneas.Std.Array Var 16#usize) (z : Var) (lo : Usize) :
    ∃ b, hash.Duplex.duplex_loop1 a z lo = ok b ∧ b.val = zeroFill a.val lo.val z := by
  unfold hash.Duplex.duplex_loop1
  exact loop_total (fun (a : Aeneas.Std.Array Var 16#usize × Usize) => hash.Duplex.duplex_loop1.body z a.1 a.2)
    (fun n b => lo.val ≤ b.2.val ∧ (b.2.val ≤ 12 ∨ b.2.val = lo.val) ∧ n = 12 - b.2.val ∧
      b.1.val = (List.range' lo.val (b.2.val - lo.val)).foldl (fun st lane => st.set lane z) a.val)
    (fun b => b.val = zeroFill a.val lo.val z)
    (by
      rintro n ⟨st, lane⟩ ⟨hlo, hle, rfl, hst⟩
      dsimp only at hlo hle hst ⊢
      by_cases hm : lane.val < 12
      · right
        have hl1 := add_one_val lane (by scalar_tac)
        refine ⟨(setN st lane.val z, core.num.Usize.wrapping_add lane 1#usize), 12 - (lane.val + 1), ?_, by omega,
          by dsimp only; omega, by dsimp only; omega, by dsimp only; omega, ?_⟩
        · simp only [hash.Duplex.duplex_loop1.body, rate_eq, show lane < 12#usize by scalar_tac, ite_true,
            update_eq st lane _ (by scalar_tac), bind_ok, lift]
        · dsimp only
          rw [setN_val, hst, hl1, show lane.val + 1 - lo.val = lane.val - lo.val + 1 by omega, List.range'_concat,
            List.foldl_append, show lo.val + 1 * (lane.val - lo.val) = lane.val by omega]
          rfl
      · left
        refine ⟨st, ?_, ?_⟩
        · simp only [hash.Duplex.duplex_loop1.body, rate_eq, show ¬ lane < 12#usize by scalar_tac, ite_false]
        · rw [hst, zeroFill, show 12 - lo.val = lane.val - lo.val by omega])
    _ (a, lo) ⟨le_refl _, Or.inr rfl, rfl, by simp⟩

theorem duplex_loop2_eq (a : Aeneas.Std.Array Var 16#usize) :
    ∃ v, hash.Duplex.duplex_loop2 a (alloc.vec.Vec.new Var) 0#usize = ok v ∧ v.val = a.val.take 12 := by
  unfold hash.Duplex.duplex_loop2
  exact loop_total (fun (b : alloc.vec.Vec Var × Usize) => hash.Duplex.duplex_loop2.body a b.1 b.2)
    (fun n b => b.2.val ≤ 12 ∧ n = 12 - b.2.val ∧ b.1.val = a.val.take b.2.val)
    (fun v => v.val = a.val.take 12)
    (by
      rintro n ⟨out, lane⟩ ⟨hle, rfl, hst⟩
      dsimp only at hle hst ⊢
      by_cases hm : lane.val < 12
      · right
        have hl1 := add_one_val lane (by scalar_tac)
        obtain ⟨w, hw, hwv⟩ := vec_push_ok out a.val[lane.val]! (by rw [hst]; simp; scalar_tac)
        refine ⟨(w, core.num.Usize.wrapping_add lane 1#usize), 12 - (lane.val + 1), ?_, by omega,
          by dsimp only; omega, by dsimp only; omega, ?_⟩
        · simp only [hash.Duplex.duplex_loop2.body, rate_eq, show lane < 12#usize by scalar_tac, ite_true,
            index_eq a lane (by scalar_tac), hw, bind_ok, lift]
        · dsimp only
          have h16 : lane.val < a.val.length := by rw [a.property]; scalar_tac
          rw [hwv, hst, hl1, List.take_add_one, List.getElem?_eq_getElem h16, Option.toList_some,
            getElem!_pos a.val lane.val h16]
      · left
        refine ⟨out, ?_, ?_⟩
        · simp only [hash.Duplex.duplex_loop2.body, rate_eq, show ¬ lane < 12#usize by scalar_tac, ite_false]
        · rw [hst, show lane.val = 12 by omega])
    _ (alloc.vec.Vec.new Var, 0#usize) ⟨by simp, rfl, by simp⟩

theorem duplex_loop3_eq (a : Aeneas.Std.Array Var 16#usize) (o : alloc.vec.Vec Var) (l : Usize) :
    hash.Duplex.duplex_loop3 a o l = hash.Duplex.duplex_loop2 a o l := rfl

theorem sample_loop_eq (v : alloc.vec.Vec Var) (last : Usize) (h : last.val ≤ v.val.length) :
    ∃ w, hash.Duplex.sample_loop v last (alloc.vec.Vec.new Var) 0#usize = ok w ∧ w.val = v.val.take last.val := by
  unfold hash.Duplex.sample_loop
  have hmax := v.property
  exact loop_total (fun (b : alloc.vec.Vec Var × Usize) => hash.Duplex.sample_loop.body v last b.1 b.2)
    (fun n b => b.2.val ≤ last.val ∧ n = last.val - b.2.val ∧ b.1.val = v.val.take b.2.val)
    (fun w => w.val = v.val.take last.val)
    (by
      rintro n ⟨out, i⟩ ⟨hle, rfl, hst⟩
      dsimp only at hle hst ⊢
      by_cases hm : i.val < last.val
      · right
        have hi : i.val < v.val.length := by omega
        have hl1 := add_one_val i (by omega)
        obtain ⟨w, hw, hwv⟩ := vec_push_ok out v.val[i.val] (by rw [hst]; simp; omega)
        refine ⟨(w, core.num.Usize.wrapping_add i 1#usize), last.val - (i.val + 1), ?_, by omega,
          by dsimp only; omega, by dsimp only; omega, ?_⟩
        · simp only [hash.Duplex.sample_loop.body, show i < last by scalar_tac, ite_true, vec_index v i hi, hw,
            bind_ok, lift]
        · dsimp only
          rw [hwv, hst, hl1, List.take_add_one, List.getElem?_eq_getElem hi, Option.toList_some]
      · left
        refine ⟨out, ?_, ?_⟩
        · simp only [hash.Duplex.sample_loop.body, show ¬ i < last by scalar_tac, ite_false]
        · rw [hst, show i.val = last.val by omega])
    _ (alloc.vec.Vec.new Var, 0#usize) ⟨by simp, rfl, by simp⟩

theorem sample_bits_loop_eq (bits : Usize) (all : alloc.vec.Vec Var) :
    ∃ w, hash.Duplex.sample_bits_loop bits all (alloc.vec.Vec.new Var) 0#usize = ok w ∧ w.val = all.val.take bits.val := by
  unfold hash.Duplex.sample_bits_loop
  have hmax := all.property
  exact loop_total (fun (b : alloc.vec.Vec Var × Usize) => hash.Duplex.sample_bits_loop.body bits all b.1 b.2)
    (fun n b => b.2.val ≤ min bits.val all.val.length ∧ n = min bits.val all.val.length - b.2.val ∧
      b.1.val = all.val.take b.2.val)
    (fun w => w.val = all.val.take bits.val)
    (by
      rintro n ⟨out, i⟩ ⟨hle, rfl, hst⟩
      dsimp only at hle hst ⊢
      by_cases hm : i.val < min bits.val all.val.length
      · right
        have hi : i.val < all.val.length := by omega
        have hl1 := add_one_val i (by omega)
        obtain ⟨w, hw, hwv⟩ := vec_push_ok out all.val[i.val] (by rw [hst]; simp; omega)
        refine ⟨(w, core.num.Usize.wrapping_add i 1#usize), min bits.val all.val.length - (i.val + 1), ?_, by omega,
          by dsimp only; omega, by dsimp only; omega, ?_⟩
        · simp only [hash.Duplex.sample_bits_loop.body, show i < bits by scalar_tac, ite_true,
            (vec_lt all i).mpr hi, vec_index all i hi, hw, bind_ok, lift]
        · dsimp only
          rw [hwv, hst, hl1, List.take_add_one, List.getElem?_eq_getElem hi, Option.toList_some]
      · left
        refine ⟨out, ?_, ?_⟩
        · by_cases hb : i.val < bits.val
          · simp only [hash.Duplex.sample_bits_loop.body, show i < bits by scalar_tac, ite_true,
              show ¬ i < alloc.vec.Vec.len all by rw [vec_lt]; omega, ite_false]
          · simp only [hash.Duplex.sample_bits_loop.body, show ¬ i < bits by scalar_tac, ite_false]
        · rw [hst, show i.val = min bits.val all.val.length by omega, ← take_min_length all.val bits.val])
    _ (alloc.vec.Vec.new Var, 0#usize) ⟨by simp, rfl, by simp⟩

theorem id_map {α β : Type} (f : α → β) (x : α) : (f <$> (x : Id α) : Id β) = f x := rfl

theorem mk_getElem! (l : List Var) (i : ℕ) : (_root_.Array.mk l)[i]! = l[i]! := by
  simp [getElem!_def]

theorem lanes16_val (out : Fin 16 → Var) : (lanes16 out).val = List.ofFn out := rfl

/-- The duplex that `duplex` leaves after the permutation `out`, as the copy's. -/
theorem permuted_eq (out : Fin 16 → Var) (w : alloc.vec.Vec Var) (s : Builder) (hw : w.val = (lanes16 out).val.take 12) :
    builtStep ({ state := lanes16 out, input := alloc.vec.Vec.new Var, output := w }, s) =
      StateT.run (pure (PUnit.unit,
        (⟨_root_.Array.ofFn out, #[], (_root_.Array.ofFn out).extract 0 12⟩ : Duplex)) : Build (Unit × Duplex)) s := by
  have he : ((List.ofFn out).take 12).toArray = (_root_.Array.ofFn out).extract 0 12 :=
    Array.toList_inj.mp (by simp [Array.toList_extract, List.extract_eq_take_drop])
  simp only [builtStep, duplexOf, hw, lanes16_val, List.toArray_ofFn, he]
  rfl

theorem fill_array (st inp : _root_.Array Var) :
    _root_.Array.foldl (fun b a => b.set! a.2 a.1) st inp.zipIdx = ⟨fill st.toList inp.toList⟩ :=
  Array.toList_inj.mp (by rw [← Array.foldl_toList, toList_foldl_set, Array.toList_zipIdx]; rfl)

theorem mk_set (l : List Var) (i : ℕ) (v : Var) : (_root_.Array.mk l).set! i v = ⟨l.set i v⟩ :=
  Array.toList_inj.mp (by rw [Array.set!_eq_setIfInBounds, Array.toList_setIfInBounds])

theorem zero_array (st : List Var) (l : List ℕ) (z : Var) :
    List.foldl (fun b a => b.set! a z) (_root_.Array.mk st) l = ⟨l.foldl (fun st lane => st.set lane z) st⟩ := by
  induction l generalizing st with
  | nil => rfl
  | cons x l ih => rw [List.foldl_cons, mk_set, ih, List.foldl_cons]

theorem duplex_eq (s : Builder) (d : RDuplex) (h : d.input.val.length ≤ 16) :
    ∃ y, hash.Duplex.duplex build d s = ok y ∧ builtStep y = (Duplex.duplexing.run (duplexOf d)).run s ∧
      y.1.input.val = [] ∧ y.1.output.val.length = 12 := by
  obtain ⟨a, ha, hav⟩ := duplex_loop0_eq d.state d.input h
  have hfill : _root_.Array.foldl (fun b a => b.set! a.2 a.1) (duplexOf d).state (duplexOf d).input.zipIdx =
      ⟨a.val⟩ := by
    rw [fill_array, hav]
    rfl
  by_cases hpos : 0 < d.input.val.length
  · obtain ⟨⟨z, s1⟩, hz⟩ : ∃ p, (Build.const (wordOf 0#u64)).run s = p := ⟨_, rfl⟩
    obtain ⟨a1, ha1, ha1v⟩ := duplex_loop1_eq a z (alloc.vec.Vec.len d.input)
    obtain ⟨⟨tag, s2⟩, htag⟩ : ∃ p, (Build.const (Replay.Word.ofNat d.input.val.length)).run s1 = p := ⟨_, rfl⟩
    obtain ⟨⟨x, s3⟩, hx⟩ : ∃ p, (Build.add a1.val[12]! tag).run s2 = p := ⟨_, rfl⟩
    obtain ⟨⟨out, s4⟩, hout⟩ : ∃ p, (Build.permute (lanesOf (setN a1 12 x))).run s3 = p := ⟨_, rfl⟩
    have hperm : build.permute s3 (setN a1 12 x) = ok (lanes16 out, s4) := by
      rw [build_permute, StateT.run_map, hout]
      rfl
    obtain ⟨w, hw, hwv⟩ := duplex_loop2_eq (lanes16 out)
    have hlen : 0#usize < alloc.vec.Vec.len d.input := by rw [UScalar.lt_equiv, vec_len_val]; simpa using hpos
    refine ⟨({ state := lanes16 out, input := alloc.vec.Vec.new Var, output := w }, s4), ?_, ?_, rfl,
      by simp [hwv, lanes16_val]⟩
    · unfold hash.Duplex.duplex
      have h12 : (12#usize : Usize).val = 12 := by simp
      simp only [ha, bind_ok, gt_iff_lt, hlen, ite_true, build_constant, hz, uncurry_eq, ha1, lift, wordOf_cast,
        vec_len_val, htag, rate_eq, index_eq a1 12#usize (by scalar_tac), h12, build_add, hx,
        update_eq a1 12#usize _ (by scalar_tac), hperm, hw]
    · have hsize : (duplexOf d).input.size = d.input.val.length := by simp [duplexOf]
      unfold Duplex.duplexing
      simp only [StateT.run_bind, StateT.run_get, pure_bind, Array.forIn_pure_yield_eq_foldl, hfill,
        Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl, hsize]
      simp only [gt_iff_lt, hpos, ite_true]
      simp only [StateT.run_bind, StateT.run_monadLift, monadLift_self, bind_pure_comp, id_bind, StateT.run_map,
        id_map, show (0 : Replay.Word) = wordOf 0#u64 from rfl, hz, htag, zero_array]
      have hfold : List.foldl (fun st lane => st.set lane z) a.val
          (List.range' d.input.val.length [d.input.val.length:12].size) = a1.val := by
        rw [ha1v, zeroFill, vec_len_val]
        simp [Std.Legacy.Range.size]
      have hl := lanesOf_eq (setN a1 12 x) (a1.val.set 12 x) (setN_val _ _ _)
      simp only [rate]
      simp only [hfold, mk_getElem!, hx, mk_set, ← hl, hout, StateT.run_set]
      exact permuted_eq out w s4 hwv
  · obtain ⟨⟨out, s1⟩, hout⟩ : ∃ p, (Build.permute (lanesOf a)).run s = p := ⟨_, rfl⟩
    have hperm : build.permute s a = ok (lanes16 out, s1) := by
      rw [build_permute, StateT.run_map, hout]
      rfl
    obtain ⟨w, hw, hwv⟩ := duplex_loop2_eq (lanes16 out)
    have hlen : ¬ 0#usize < alloc.vec.Vec.len d.input := by rw [UScalar.lt_equiv, vec_len_val]; simpa using hpos
    refine ⟨({ state := lanes16 out, input := alloc.vec.Vec.new Var, output := w }, s1), ?_, ?_, rfl,
      by simp [hwv, lanes16_val]⟩
    · unfold hash.Duplex.duplex
      simp only [ha, bind_ok, gt_iff_lt, hlen, ite_false, uncurry_eq, hperm, duplex_loop3_eq, hw]
    · have hsize : (duplexOf d).input.size = d.input.val.length := by simp [duplexOf]
      have hl := lanesOf_eq a a.val rfl
      unfold Duplex.duplexing
      simp only [StateT.run_bind, StateT.run_get, pure_bind, Array.forIn_pure_yield_eq_foldl, hfill, hsize, gt_iff_lt,
        hpos, ite_false, StateT.run_monadLift, monadLift_self, bind_pure_comp, StateT.run_map, id_map, ← hl,
        hout, StateT.run_set, rate]
      exact permuted_eq out w s1 hwv

theorem duplex_returns (s : Builder) (d : RDuplex) (y : RDuplex × Builder) (h : hash.Duplex.duplex build d s = ok y) :
    d.input.val.length ≤ 16 := by
  unfold hash.Duplex.duplex at h
  obtain ⟨a, ha, _⟩ := bind_eq_ok h
  exact duplex_loop0_returns _ _ a ha

/-- An empty duplex, a witness. -/
def emptyDuplex : RDuplex := ⟨Aeneas.Std.Array.repeat 16#usize 0, alloc.vec.Vec.new Var, alloc.vec.Vec.new Var⟩

theorem duplex_sound : Sound (fun x : RDuplex × Builder => hash.Duplex.duplex build x.1 x.2) builtStep
    (fun x => (Duplex.duplexing.run (duplexOf x.1)).run x.2) :=
  sound_of_eq_on (fun x => x.1.input.val.length ≤ 16)
    (fun x h => let ⟨y, hy, hv, _⟩ := duplex_eq x.2 x.1 h; ⟨y, hy, hv⟩) fun x y h => duplex_returns x.2 x.1 y h

theorem duplex_complete : Complete (fun x : RDuplex × Builder => x.1.input.val.length ≤ 16)
    (fun x : RDuplex × Builder => hash.Duplex.duplex build x.1 x.2) builtStep
    (fun x => (Duplex.duplexing.run (duplexOf x.1)).run x.2) :=
  complete_of_eq duplex_sound (emptyDuplex, {}) (by simp [emptyDuplex])
    fun x h => let ⟨y, hy, hv, _⟩ := duplex_eq x.2 x.1 h; ⟨y, hy, hv⟩

/-! ## `new` -/

theorem new_eq (s : Builder) : ∃ y, hash.Duplex.new build s = ok y ∧ builtNew y = Duplex.new.run s := by
  obtain ⟨⟨z, s1⟩, hz⟩ : ∃ p, (Build.const (wordOf 0#u64)).run s = p := ⟨_, rfl⟩
  refine ⟨(⟨Aeneas.Std.Array.repeat 16#usize z, alloc.vec.Vec.new Var, alloc.vec.Vec.new Var⟩, s1), ?_, ?_⟩
  · unfold hash.Duplex.new
    simp only [build_constant, hz, bind_ok, uncurry_eq]
  · simp only [Duplex.new, StateT.run_bind, show (0 : Replay.Word) = wordOf 0#u64 from rfl, hz, id_bind]
    simp [builtNew, duplexOf, Aeneas.Std.Array.repeat]
    rfl

theorem new_sound : Sound (hash.Duplex.new build) builtNew (fun s => Duplex.new.run s) :=
  sound_of_eq new_eq

theorem new_complete : Complete (fun _ => True) (hash.Duplex.new build) builtNew (fun s => Duplex.new.run s) :=
  complete_of_eq new_sound {} trivial fun s _ => new_eq s

/-! ## `observe` -/

theorem vec_push_returns {α : Type} {v w : alloc.vec.Vec α} {x : α} (h : alloc.vec.Vec.push v x = ok w) :
    v.val.length < Usize.max := by
  unfold alloc.vec.Vec.push at h
  dsimp only at h
  split at h
  · rename_i hc
    simp only [Bool.or_eq_true, decide_eq_true_eq] at hc
    have := max_ge
    omega
  · exact absurd h fail_not_ok

theorem observe_eq (s : Builder) (d : RDuplex) (v : Var) (h : d.input.val.length < Usize.max) :
    ∃ y, hash.Duplex.observe build d s v = ok y ∧ builtStep y = ((Duplex.observe v).run (duplexOf d)).run s ∧
      (d.input.val.length < 12 → y.1.input.val.length < 12) := by
  obtain ⟨w, hw, hwv⟩ := vec_push_ok d.input v h
  have hd : duplexOf { d with input := w, output := alloc.vec.Vec.new Var } =
      ⟨(duplexOf d).state, (duplexOf d).input.push v, #[]⟩ := by
    simp [duplexOf, hwv]
  have hlen : (alloc.vec.Vec.len w).val = d.input.val.length + 1 := by rw [vec_len_val, hwv]; simp
  have hsz : ((duplexOf d).input.push v).size = d.input.val.length + 1 := by simp [duplexOf]
  by_cases h12 : d.input.val.length + 1 = 12
  · obtain ⟨y, hy, hv, hin, _⟩ := duplex_eq s { d with input := w, output := alloc.vec.Vec.new Var }
      (by simp only [hwv, List.length_append, List.length_singleton]; omega)
    have hr : alloc.vec.Vec.len w = circuit.RATE := by
      rw [rate_eq]
      exact UScalar.val_eq_imp _ _ (by rw [hlen, h12]; rfl)
    refine ⟨y, ?_, ?_, fun _ => by rw [hin]; simp⟩
    · unfold hash.Duplex.observe
      simp only [hw, bind_ok, hr, ite_true, hy]
    · rw [hv, hd]
      unfold Duplex.observe
      simp only [StateT.run_bind, StateT.run_modify, StateT.run_get, pure_bind, hsz, h12, rate, ite_true]
  · have hr : ¬ alloc.vec.Vec.len w = circuit.RATE := by
      rw [rate_eq]
      intro he
      have := congrArg UScalar.val he
      rw [hlen] at this
      exact h12 this
    refine ⟨({ d with input := w, output := alloc.vec.Vec.new Var }, s), ?_, ?_, ?_⟩
    · unfold hash.Duplex.observe
      simp only [hw, bind_ok, hr, ite_false]
    · unfold Duplex.observe
      simp only [StateT.run_bind, StateT.run_modify, StateT.run_get, pure_bind, hsz, h12, rate, ite_false,
        builtStep, hd]
      rfl
    · intro hl
      simp only [hwv, List.length_append, List.length_singleton]
      omega

theorem observe_returns (s : Builder) (d : RDuplex) (v : Var) (y : RDuplex × Builder)
    (h : hash.Duplex.observe build d s v = ok y) : d.input.val.length < Usize.max := by
  unfold hash.Duplex.observe at h
  obtain ⟨w, hw, _⟩ := bind_eq_ok h
  exact vec_push_returns hw

/-- `observe` returns the copy's run wherever it returns. -/
theorem observe_run (s : Builder) (d : RDuplex) (v : Var) (y : RDuplex × Builder)
    (h : hash.Duplex.observe build d s v = ok y) :
    ((Duplex.observe v).run (duplexOf d)).run s = (((), duplexOf y.1), y.2) := by
  obtain ⟨y', hy, hv, _⟩ := observe_eq s d v (observe_returns s d v y h)
  rw [hy, Result.ok.injEq] at h
  rw [← hv, h]
  rfl

theorem observe_sound : Sound (fun x : RDuplex × Builder × Var => hash.Duplex.observe build x.1 x.2.1 x.2.2) builtStep
    (fun x => ((Duplex.observe x.2.2).run (duplexOf x.1)).run x.2.1) :=
  fun x y h => (observe_run x.2.1 x.1 x.2.2 y h).symm

theorem observe_complete : Complete (fun x : RDuplex × Builder × Var => x.1.input.val.length < Usize.max)
    (fun x : RDuplex × Builder × Var => hash.Duplex.observe build x.1 x.2.1 x.2.2) builtStep
    (fun x => ((Duplex.observe x.2.2).run (duplexOf x.1)).run x.2.1) :=
  complete_of_eq observe_sound (emptyDuplex, {}, 0) (by simp [emptyDuplex]; scalar_tac)
    fun x h => let ⟨y, hy, hv, _⟩ := observe_eq x.2.1 x.1 x.2.2 h; ⟨y, hy, hv⟩

/-- An observation that keeps the input below `RATE` words. -/
theorem observe_fits (s : Builder) (d : RDuplex) (v : Var) (h : d.input.val.length < 12) :
    ∃ y, hash.Duplex.observe build d s v = ok y ∧ y.1.input.val.length < 12 := by
  obtain ⟨y, hy, _, hin⟩ := observe_eq s d v (by scalar_tac)
  exact ⟨y, hy, hin h⟩

/-! ## `observe_ext` -/

theorem coordinates_index (a : ExtVar) (i : Usize) (h : i.val < 3) :
    Aeneas.Std.Array.index_usize (handles3 a) i = ok (a ⟨i.val, h⟩) := by
  rw [index_eq _ i (by scalar_tac)]
  have : i.val = 0 ∨ i.val = 1 ∨ i.val = 2 := by omega
  rcases this with h0 | h0 | h0 <;> simp [handles3, h0]

theorem observe_ext_unfold (s : Builder) (d : RDuplex) (a : ExtVar) :
    hash.Duplex.observe_ext build d s a = (do
      let p ← hash.Duplex.observe build d s (a 0)
      let q ← hash.Duplex.observe build p.1 p.2 (a 1)
      hash.Duplex.observe build q.1 q.2 (a 2)) := by
  unfold hash.Duplex.observe_ext
  simp only [build_coordinates, bind_ok, uncurry_eq, coordinates_index a 0#usize (by simp),
    coordinates_index a 1#usize (by simp), coordinates_index a 2#usize (by simp)]
  rfl

theorem observe_ext_sound : Sound (fun x : RDuplex × Builder × ExtVar => hash.Duplex.observe_ext build x.1 x.2.1 x.2.2)
    builtStep (fun x => ((Duplex.observeExt x.2.2).run (duplexOf x.1)).run x.2.1) := by
  rintro ⟨d, s, a⟩ y h
  dsimp only at h ⊢
  rw [observe_ext_unfold] at h
  obtain ⟨p, hp, h⟩ := bind_eq_ok h
  obtain ⟨q, hq, h⟩ := bind_eq_ok h
  unfold Duplex.observeExt
  simp only [StateT.run_bind, observe_run s d _ p hp, observe_run _ _ _ q hq, observe_run _ _ _ y h, id_bind]
  rfl

theorem observe_ext_complete : Complete (fun x : RDuplex × Builder × ExtVar => x.1.input.val.length < 12)
    (fun x : RDuplex × Builder × ExtVar => hash.Duplex.observe_ext build x.1 x.2.1 x.2.2)
    builtStep (fun x => ((Duplex.observeExt x.2.2).run (duplexOf x.1)).run x.2.1) := by
  refine complete_of observe_ext_sound ⟨(emptyDuplex, {}, fun _ => 0), by simp [emptyDuplex]⟩ ?_
  rintro ⟨d, s, a⟩ h
  dsimp only at h ⊢
  obtain ⟨p, hp, h1⟩ := observe_fits s d (a 0) h
  obtain ⟨q, hq, h2⟩ := observe_fits p.2 p.1 (a 1) h1
  obtain ⟨y, hy, _⟩ := observe_fits q.2 q.1 (a 2) h2
  exact ⟨y, by rw [observe_ext_unfold, hp, bind_ok, hq, bind_ok, hy]⟩

/-! ## `observe_slice` -/

/-- One observation of the copy as a builder step on the duplex. -/
def observeStep (v : Var) (D : Duplex) : Build Duplex := (fun p => p.2) <$> (Duplex.observe v).run D

theorem forM_run (l : List Var) (D : Duplex) :
    (forM l Duplex.observe).run D =
      (fun D' => ((), D')) <$> forIn l D (fun v D => ForInStep.yield <$> observeStep v D) := by
  induction l generalizing D with
  | nil => rfl
  | cons v l ih =>
    simp only [List.forM_cons, StateT.run_bind, ih, List.forIn_cons, observeStep, map_bind, bind_map_left]

theorem observe_slice_run (values : Slice Var) (D : Duplex) (s : Builder) :
    ((Duplex.observeSlice values.val.toArray).run D).run s =
      (let p := (forIn values.val D fun v D => ForInStep.yield <$> observeStep v D).run s; (((), p.1), p.2)) := by
  simp only [Duplex.observeSlice, List.forM_toArray' _ _ rfl, List.forM_eq_forM, forM_run, StateT.run_map]
  rfl

theorem observe_slice_sound : Sound
    (fun x : RDuplex × Builder × Slice Var => hash.Duplex.observe_slice build x.1 x.2.1 x.2.2) builtStep
    (fun x => ((Duplex.observeSlice x.2.2.val.toArray).run (duplexOf x.1)).run x.2.1) := by
  rintro ⟨d, s, values⟩ y h
  dsimp only at h ⊢
  unfold hash.Duplex.observe_slice hash.Duplex.observe_slice_loop at h
  have := loop_forIn_rel_sound (fun (a : RDuplex × Builder × Usize) =>
      hash.Duplex.observe_slice_loop.body build values a.1 a.2.1 a.2.2)
    (fun a s r k => duplexOf a.1 = r ∧ a.2.1 = s ∧ a.2.2.val = k)
    (fun y r s => duplexOf y.1 = r ∧ y.2 = s) values.val observeStep ?_ ?_ values.val.length (d, s, 0#usize) s
    (duplexOf d) 0 y ⟨rfl, rfl, rfl⟩ (by simp) h
  · rw [observe_slice_run]
    simp only [List.drop_zero] at this
    rw [builtStep, this.1, this.2]
  · rintro ⟨e, b, i⟩ s r k c hk ⟨rfl, rfl, hi⟩ hc
    dsimp only at hi
    simp only [hash.Duplex.observe_slice_loop.body, show i < Slice.len values by rw [slice_lt]; omega, ite_true,
      slice_index values i (by omega), hi, bind_ok] at hc
    obtain ⟨⟨e', b'⟩, hp, hc⟩ := bind_eq_ok hc
    simp only [uncurry_eq, lift, bind_ok, Result.ok.injEq] at hc
    refine ⟨_, hc.symm, ?_, ?_, ?_⟩
    · simp only [observeStep, StateT.run_map, observe_run b e _ _ hp]
      rfl
    · simp only [observeStep, StateT.run_map, observe_run b e _ _ hp]
      rfl
    · rw [add_one_val i (by have := values.property; omega), hi]
  · rintro ⟨e, b, i⟩ s r k c ⟨rfl, rfl, hi⟩ hk hc
    dsimp only at hi
    simp only [hash.Duplex.observe_slice_loop.body, show ¬ i < Slice.len values by rw [slice_lt]; omega, ite_false,
      Result.ok.injEq] at hc
    exact ⟨_, hc.symm, rfl, rfl⟩

theorem observe_slice_complete : Complete (fun x : RDuplex × Builder × Slice Var => x.1.input.val.length < 12)
    (fun x : RDuplex × Builder × Slice Var => hash.Duplex.observe_slice build x.1 x.2.1 x.2.2) builtStep
    (fun x => ((Duplex.observeSlice x.2.2.val.toArray).run (duplexOf x.1)).run x.2.1) := by
  refine complete_of observe_slice_sound ⟨(emptyDuplex, {}, emptySlice), by simp [emptyDuplex]⟩ ?_
  rintro ⟨d, s, values⟩ h
  dsimp only at h ⊢
  unfold hash.Duplex.observe_slice hash.Duplex.observe_slice_loop
  have key := loop_forIn_spec (fun (a : RDuplex × Builder × Usize) =>
      hash.Duplex.observe_slice_loop.body build values a.1 a.2.1 a.2.2)
    (fun a s r k => duplexOf a.1 = r ∧ a.2.1 = s ∧ a.2.2.val = k ∧ a.1.input.val.length < 12)
    (fun _ _ _ => True) values.val observeStep ?_ ?_ (d, s, 0#usize) s (duplexOf d) 0 ⟨rfl, rfl, rfl, h⟩
  · obtain ⟨y, hy, _⟩ := key
    exact ⟨y, hy⟩
  · rintro ⟨e, b, i⟩ s r k hk ⟨rfl, rfl, hi, he⟩
    dsimp only at hi he
    obtain ⟨p, hp, hp12⟩ := observe_fits b e values.val[k] he
    refine ⟨(p.1, p.2, core.num.Usize.wrapping_add i 1#usize), ?_, ?_, ?_, ?_, hp12⟩
    · simp only [hash.Duplex.observe_slice_loop.body, show i < Slice.len values by rw [slice_lt]; omega, ite_true,
        slice_index values i (by omega), hi, hp, bind_ok, uncurry_eq, lift]
    · simp only [observeStep, StateT.run_map, observe_run b e _ p hp]
      rfl
    · simp only [observeStep, StateT.run_map, observe_run b e _ p hp]
      rfl
    · rw [add_one_val i (by have := values.property; omega), hi]
  · rintro ⟨e, b, i⟩ s r k ⟨rfl, rfl, hi, _⟩ hk
    dsimp only at hi
    exact ⟨(e, b), by simp only [hash.Duplex.observe_slice_loop.body,
      show ¬ i < Slice.len values by rw [slice_lt]; omega, ite_false], trivial⟩

/-! ## `sample` -/

/-- What `sample` does after the duplexing: pop the last output word. -/
def sampleTail (e : RDuplex) (b : Builder) : Result (Var × RDuplex × Builder) := do
  let i := alloc.vec.Vec.len e.output
  let last ← lift (core.num.Usize.wrapping_sub i 1#usize)
  let value ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Var) e.output last
  let rest ← hash.Duplex.sample_loop e.output last (alloc.vec.Vec.new Var) 0#usize
  ok (value, { e with output := rest }, b)

/-- The copy's `sample` after the duplexing. -/
def copyTail : Transcript Var := do
  let d ← get
  set { state := d.state, input := d.input, output := d.output.pop : Duplex }
  pure d.output.back!

theorem sample_unfold (s : Builder) (d : RDuplex) :
    hash.Duplex.sample build d s = (do
      let p ← if d.input.val.length = 0 then
          (if d.output.val.length = 0 then hash.Duplex.duplex build d s else ok (d, s))
        else hash.Duplex.duplex build d s
      sampleTail p.1 p.2) := by
  unfold hash.Duplex.sample
  simp only [core.slice.Slice.is_empty, alloc.vec.Vec.deref, bind_ok, Slice.length, Slice.from_val,
    decide_eq_true_eq]
  rfl

theorem sample_copy_unfold :
    Duplex.sample = (do
      let d ← get
      if (!d.input.isEmpty || d.output.isEmpty) = true then (Duplex.duplexing >>= fun _ => copyTail) else copyTail) := by
  rfl

theorem sample_tail_eq (e : RDuplex) (b : Builder) (hne : 0 < e.output.val.length) :
    ∃ y, sampleTail e b = ok y ∧ builtSample y = (copyTail.run (duplexOf e)).run b ∧ y.2.1.input = e.input := by
  have hlast : (core.num.Usize.wrapping_sub (alloc.vec.Vec.len e.output) 1#usize).val = e.output.val.length - 1 := by
    rw [wrapping_sub_val _ _ (by rw [vec_len_val]; simp; omega), vec_len_val]
    simp
  obtain ⟨w, hw, hwv⟩ := sample_loop_eq e.output _ (by rw [hlast]; omega)
  refine ⟨(e.output.val[e.output.val.length - 1], { e with output := w }, b), ?_, ?_, rfl⟩
  · simp only [sampleTail, lift, bind_ok, vec_index e.output _ (by rw [hlast]; omega), hlast, hw]
  · have hback : e.output.val.toArray.back! = e.output.val[e.output.val.length - 1] := by
      simp [Array.back!_eq_back?, List.getLast?_eq_getElem?, List.getElem?_eq_getElem (show e.output.val.length - 1 <
        e.output.val.length by omega)]
    have hpop : e.output.val.toArray.pop = w.val.toArray := by
      rw [hwv, hlast]
      exact Array.toList_inj.mp (by simp [List.dropLast_eq_take])
    simp only [copyTail, StateT.run_bind, StateT.run_get, StateT.run_set, pure_bind, builtSample, duplexOf, hback,
      hpop]
    rfl

theorem sample_eq (s : Builder) (d : RDuplex) (h : d.input.val.length ≤ 16) :
    ∃ y, hash.Duplex.sample build d s = ok y ∧ builtSample y = (Duplex.sample.run (duplexOf d)).run s ∧
      y.2.1.input.val = [] := by
  rw [sample_unfold, sample_copy_unfold]
  have hin : (duplexOf d).input.isEmpty = decide (d.input.val.length = 0) := by
    simp only [duplexOf, List.isEmpty_toArray]
    cases d.input.val <;> rfl
  have hout : (duplexOf d).output.isEmpty = decide (d.output.val.length = 0) := by
    simp only [duplexOf, List.isEmpty_toArray]
    cases d.output.val <;> rfl
  by_cases hc : d.input.val.length = 0 ∧ d.output.val.length ≠ 0
  · obtain ⟨y, hy, hv, hyin⟩ := sample_tail_eq d s (by omega)
    refine ⟨y, ?_, ?_, by rw [hyin]; exact List.eq_nil_of_length_eq_zero hc.1⟩
    · simp only [hc.1, hc.2, ite_true, ite_false, bind_ok, hy]
    · simp only [StateT.run_bind, StateT.run_get, pure_bind, hin, hout, hc.1, hc.2, decide_true, decide_false,
        Bool.not_true, Bool.false_or, Bool.false_eq_true, ite_false, hv]
  · obtain ⟨p, hp, hpv, hpin, hpout⟩ := duplex_eq s d h
    obtain ⟨y, hy, hv, hyin⟩ := sample_tail_eq p.1 p.2 (by omega)
    have hfirst : (if d.input.val.length = 0 then
        (if d.output.val.length = 0 then hash.Duplex.duplex build d s else ok (d, s))
      else hash.Duplex.duplex build d s) = ok p := by
      by_cases h0 : d.input.val.length = 0
      · have h1 : d.output.val.length = 0 := by omega
        simp only [h0, h1, ite_true, hp]
      · simp only [h0, ite_false, hp]
    have hcond : (!(duplexOf d).input.isEmpty || (duplexOf d).output.isEmpty) = true := by
      rw [hin, hout]
      by_cases h0 : d.input.val.length = 0
      · have h1 : d.output.val.length = 0 := by omega
        simp [h0, h1]
      · simp [h0]
    refine ⟨y, by rw [hfirst, bind_ok, hy], ?_, by rw [hyin, hpin]⟩
    simp only [StateT.run_bind, StateT.run_get, pure_bind, hcond, ite_true]
    rw [hv, ← hpv]
    rfl

theorem sample_returns (s : Builder) (d : RDuplex) (y : Var × RDuplex × Builder)
    (h : hash.Duplex.sample build d s = ok y) : d.input.val.length ≤ 16 := by
  rw [sample_unfold] at h
  obtain ⟨p, hp, _⟩ := bind_eq_ok h
  by_cases h0 : d.input.val.length = 0
  · omega
  · simp only [h0, ite_false] at hp
    exact duplex_returns s d p hp

/-- `sample` returns the copy's run wherever it returns. -/
theorem sample_run (s : Builder) (d : RDuplex) (y : Var × RDuplex × Builder)
    (h : hash.Duplex.sample build d s = ok y) :
    (Duplex.sample.run (duplexOf d)).run s = ((y.1, duplexOf y.2.1), y.2.2) := by
  obtain ⟨y', hy, hv, _⟩ := sample_eq s d (sample_returns s d y h)
  rw [hy, Result.ok.injEq] at h
  rw [← hv, h]
  rfl

theorem sample_sound : Sound (fun x : RDuplex × Builder => hash.Duplex.sample build x.1 x.2) builtSample
    (fun x => (Duplex.sample.run (duplexOf x.1)).run x.2) :=
  fun x y h => (sample_run x.2 x.1 y h).symm

theorem sample_complete : Complete (fun x : RDuplex × Builder => x.1.input.val.length ≤ 16)
    (fun x : RDuplex × Builder => hash.Duplex.sample build x.1 x.2) builtSample
    (fun x => (Duplex.sample.run (duplexOf x.1)).run x.2) :=
  complete_of_eq sample_sound (emptyDuplex, {}) (by simp [emptyDuplex])
    fun x h => let ⟨y, hy, hv, _⟩ := sample_eq x.2 x.1 h; ⟨y, hy, hv⟩

/-! ## `sample_ext` -/

theorem sample_ext_unfold (s : Builder) (d : RDuplex) :
    hash.Duplex.sample_ext build d s = (do
      let p ← hash.Duplex.sample build d s
      let q ← hash.Duplex.sample build p.2.1 p.2.2
      let r ← hash.Duplex.sample build q.2.1 q.2.2
      ok (handle (Aeneas.Std.Array.make 3#usize [p.1, q.1, r.1]), r.2.1, r.2.2)) := by
  unfold hash.Duplex.sample_ext
  simp only [build_ext]
  refine congrArg (Std.bind _) (funext fun p => ?_)
  obtain ⟨x, e1, b1⟩ := p
  refine congrArg (Std.bind _) (funext fun q => ?_)
  obtain ⟨y, e2, b2⟩ := q
  refine congrArg (Std.bind _) (funext fun r => ?_)
  obtain ⟨z, e3, b3⟩ := r
  simp only [uncurry_eq, bind_ok]

theorem sample_ext_sound : Sound (fun x : RDuplex × Builder => hash.Duplex.sample_ext build x.1 x.2) builtSample
    (fun x => (Duplex.sampleExt.run (duplexOf x.1)).run x.2) := by
  rintro ⟨d, s⟩ y h
  dsimp only at h ⊢
  rw [sample_ext_unfold] at h
  obtain ⟨p, hp, h⟩ := bind_eq_ok h
  obtain ⟨q, hq, h⟩ := bind_eq_ok h
  obtain ⟨r, hr, h⟩ := bind_eq_ok h
  rw [Result.ok.injEq] at h
  subst h
  unfold Duplex.sampleExt
  simp only [StateT.run_bind, sample_run s d p hp, sample_run _ _ q hq, sample_run _ _ r hr, id_bind, handle_make]
  rfl

theorem sample_ext_complete : Complete (fun x : RDuplex × Builder => x.1.input.val.length ≤ 16)
    (fun x : RDuplex × Builder => hash.Duplex.sample_ext build x.1 x.2) builtSample
    (fun x => (Duplex.sampleExt.run (duplexOf x.1)).run x.2) := by
  refine complete_of sample_ext_sound ⟨(emptyDuplex, {}), by simp [emptyDuplex]⟩ ?_
  rintro ⟨d, s⟩ h
  dsimp only at h ⊢
  obtain ⟨p, hp, _, hp0⟩ := sample_eq s d h
  obtain ⟨q, hq, _, hq0⟩ := sample_eq p.2.2 p.2.1 (by rw [hp0]; simp)
  obtain ⟨r, hr, _, _⟩ := sample_eq q.2.2 q.2.1 (by rw [hq0]; simp)
  exact ⟨_, by rw [sample_ext_unfold, hp, bind_ok, hq, bind_ok, hr, bind_ok]⟩

/-! ## `sample_bits` -/

theorem sample_bits_unfold (s : Builder) (d : RDuplex) (bits : Usize) :
    hash.Duplex.sample_bits build d s bits = (do
      let p ← hash.Duplex.sample build d s
      let w ← hash.Duplex.sample_bits_loop bits
        (vecOfFn (val_le_max 64#usize) ((Build.bits p.1 (64#usize).val).run p.2.2).1) (alloc.vec.Vec.new Var) 0#usize
      ok (.Ok w, p.2.1, ((Build.bits p.1 (64#usize).val).run p.2.2).2)) := by
  unfold hash.Duplex.sample_bits
  refine congrArg (Std.bind _) (funext fun p => ?_)
  obtain ⟨x, e, b⟩ := p
  simp only [uncurry_eq, build_bits, checked, StateT.run_map, bind_ok, core.result.Result.Insts.CoreOpsTry.branch,
    id_map]

theorem sample_bits_sound : Sound
    (fun x : RDuplex × Builder × Usize => hash.Duplex.sample_bits build x.1 x.2.1 x.2.2) builtBits
    (fun x => let r := ((Duplex.sampleBits x.2.2.val).run (duplexOf x.1)).run x.2.1; ((.ok r.1.1, r.1.2), r.2)) := by
  rintro ⟨d, s, bits⟩ y h
  dsimp only at h ⊢
  rw [sample_bits_unfold] at h
  obtain ⟨p, hp, h⟩ := bind_eq_ok h
  obtain ⟨w, hw, h⟩ := bind_eq_ok h
  rw [Result.ok.injEq] at h
  subst h
  obtain ⟨⟨all, s2⟩, hb⟩ : ∃ q, (Build.bits p.1 (64#usize).val).run p.2.2 = q := ⟨_, rfl⟩
  obtain ⟨w', hw', hwv⟩ := sample_bits_loop_eq bits (vecOfFn (val_le_max 64#usize) all)
  rw [hb] at hw
  rw [hw, Result.ok.injEq] at hw'
  subst hw'
  unfold Duplex.sampleBits
  have hb' : (Build.bits p.1 64).run p.2.2 = ((all : Fin 64 → Var), s2) := hb
  have he : ((List.ofFn all).take bits.val).toArray = (_root_.Array.ofFn (all : Fin 64 → Var)).extract 0 bits.val :=
    Array.toList_inj.mp (by simp [Array.toList_extract, List.extract_eq_take_drop])
  simp only [StateT.run_bind, sample_run s d p hp, id_bind, StateT.run_monadLift, monadLift_self, hb, hb',
    builtBits, resultOf, hwv, vecOfFn, alloc.vec.Vec.from_val, StateT.run_pure, id_pure, he]

theorem sample_bits_complete : Complete (fun x : RDuplex × Builder × Usize => x.1.input.val.length ≤ 16)
    (fun x : RDuplex × Builder × Usize => hash.Duplex.sample_bits build x.1 x.2.1 x.2.2) builtBits
    (fun x => let r := ((Duplex.sampleBits x.2.2.val).run (duplexOf x.1)).run x.2.1; ((.ok r.1.1, r.1.2), r.2)) := by
  refine complete_of sample_bits_sound ⟨(emptyDuplex, {}, 0#usize), by simp [emptyDuplex]⟩ ?_
  rintro ⟨d, s, bits⟩ h
  dsimp only at h ⊢
  obtain ⟨p, hp, _, _⟩ := sample_eq s d h
  obtain ⟨w, hw, _⟩ := sample_bits_loop_eq bits
    (vecOfFn (val_le_max 64#usize) ((Build.bits p.1 (64#usize).val).run p.2.2).1)
  exact ⟨_, by rw [sample_bits_unfold, hp, bind_ok, hw, bind_ok]⟩

end CompressionLeanRust
