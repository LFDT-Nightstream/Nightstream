import NeoVerified.Funs
import CompressionLean.Ports.Circuit.Hash
import CompressionLeanRust.Field
import CompressionLeanRust.Positions

/-! The copies' `Build` as a translated `Backend` (`build`), and the facts that the proofs of the
generic helpers use.

A helper of `crates/neo-verified` that is generic over `B: Backend` translates to a function that
takes the trait as a structure of operations. Run on `build`, each operation is the `Build`
operation of `Ports/Circuit/Ir.lean` that the copies call: a value is a `Var`, an extension value
an `ExtVar`, and the state is the `Builder`. So "the translation on `build` equals the copy's run"
says that the helper makes the copy's calls, in the copy's order, with the copy's constants. That
`Recorder` lowers each call as `Record.lean` does is not part of it.

A `u64` that the trait takes is read mod `p` (`wordOf`), as `Recorder` reads it. `ext` and
`coordinates` change no state: the copies write `![x, y, z]` and `a 0`. -/

namespace CompressionLeanRust

open Aeneas Aeneas.Std Result
open neo_verified
open CompressionLean
open CompressionLean.Circuit

/-- A translated `[F; 3]` as the copies' extension handle. -/
def handle (a : Aeneas.Std.Array Var 3#usize) : ExtVar := fun i => a.val.getD i 0

/-- The copies' extension handle as a translated `[F; 3]`. -/
def handles3 (a : ExtVar) : Aeneas.Std.Array Var 3#usize := Aeneas.Std.Array.make 3#usize [a 0, a 1, a 2]

/-- A translated `[F; 16]` as the copies' lanes. -/
def lanesOf (a : Aeneas.Std.Array Var 16#usize) : Fin 16 → Var := fun i => a.val.getD i 0

def lanes16 (f : Fin 16 → Var) : Aeneas.Std.Array Var 16#usize := Aeneas.Std.Array.make 16#usize (List.ofFn f)

theorem val_le_max (x : Usize) : x.val ≤ Usize.max := by scalar_tac

/-- `n` handles as a translated `Vec`. -/
def vecOfFn {n : ℕ} (h : n ≤ Usize.max) (f : Fin n → Var) : alloc.vec.Vec Var :=
  alloc.vec.Vec.from (List.ofFn f) (by rw [List.length_ofFn]; exact h)

/-- A `Build` step that cannot fail, as a step that returns `Ok`. -/
def checked {α : Type} (b : Build α) (s : Builder) : core.result.Result α backend.Rejected × Builder :=
  ((b.run s).1 |> .Ok, (b.run s).2)

/-- Terms `from` to `to` of a translated R1CS system: column and coefficient. -/
def termsOf (system : r1cs.R1cs) (first last : ℕ) : List (ℕ × Replay.Word) :=
  (List.range (last - first)).map fun k =>
    ((system.column.val.getD (first + k) 0#u32).val, wordOf (system.coefficient.val.getD (first + k) 0#u64))

/-- The end of side `k` of row `r`. -/
def endOf (system : r1cs.R1cs) (r k : ℕ) : ℕ :=
  match system.ends.val[r]? with
  | some ends => (ends.val.getD k 0#u32).val
  | none => 0

/-- The rows of a translated R1CS system, as `R1cs::row` reads them. -/
def rowsOf (system : r1cs.R1cs) : List (CompressionLean.Shrink.R1csRow Replay.Word) :=
  (List.range system.ends.val.length).map fun r =>
    let start := if r = 0 then 0 else endOf system (r - 1) 2
    ⟨termsOf system start (endOf system r 0), termsOf system (endOf system r 0) (endOf system r 1),
      termsOf system (endOf system r 1) (endOf system r 2)⟩

/-- The copies' `Build`, as the translated `Backend` trait. -/
def build : backend.Backend Builder Var ExtVar where
  coremarkerCopyInst := BuiltinCopy Var
  ValueInst := { coremarkerCopyInst := BuiltinCopy ExtVar }
  constant s v := ok ((Build.const (wordOf v)).run s)
  «private» s _ := ok (Build.priv.run s)
  «public» s _ := ok (Build.pub.run s)
  add s a b := ok ((Build.add a b).run s)
  sub s a b := ok ((Build.sub a b).run s)
  scale s a by_ := ok ((Build.scale a (wordOf by_)).run s)
  mul s a b := ok ((Build.mul a b).run s)
  ext s c := ok (handle c, s)
  coordinates s a := ok (handles3 a, s)
  ext_constant s v := ok ((Build.extConstant fun i => wordOf (v.val.getD i 0#u64)).run s)
  ext_add s a b := ok ((Build.extAdd a b).run s)
  ext_sub s a b := ok ((Build.extSub a b).run s)
  ext_mul s a b := ok ((Build.extMul a b).run s)
  ext_scale s a by_ := ok ((Build.extScale a by_).run s)
  ext_inverse s a _ := ok (checked (Build.extInv a) s)
  assert_zero s a _ := ok (checked (Build.assertZero a) s)
  bits s a count _ := ok (checked ((vecOfFn (val_le_max count)) <$> Build.bits a count.val) s)
  select s bit z o := ok ((Build.select bit z o).run s)
  permute s state := ok (((lanes16 ·) <$> Build.permute (lanesOf state)).run s)
  hint s _ len _ := ok (((vecOfFn (val_le_max len)) <$> Build.hint len.val).run s)
  r1cs s system columns _ := ok (checked (Build.r1cs (rowsOf system) columns.val) s)
  assert_ext_zero s a _ := ok (checked (Build.assertExtZero a) s)
  assert_equal s a b _ := ok (checked (Build.assertEqual a b) s)
  assert_ext_equal s a b _ := ok (checked (Build.assertExtEqual a b) s)

/-! ## The operations as `simp` lemmas -/

section
variable (s : Builder)

@[simp] theorem build_constant (v : U64) : build.constant s v = ok ((Build.const (wordOf v)).run s) := rfl
@[simp] theorem build_private (v : U64) : build.«private» s v = ok (Build.priv.run s) := rfl
@[simp] theorem build_add (a b : Var) : build.add s a b = ok ((Build.add a b).run s) := rfl
@[simp] theorem build_sub (a b : Var) : build.sub s a b = ok ((Build.sub a b).run s) := rfl
@[simp] theorem build_scale (a : Var) (v : U64) : build.scale s a v = ok ((Build.scale a (wordOf v)).run s) := rfl
@[simp] theorem build_mul (a b : Var) : build.mul s a b = ok ((Build.mul a b).run s) := rfl
@[simp] theorem build_ext (c : Aeneas.Std.Array Var 3#usize) : build.ext s c = ok (handle c, s) := rfl
@[simp] theorem build_coordinates (a : ExtVar) : build.coordinates s a = ok (handles3 a, s) := rfl
@[simp] theorem build_ext_constant (v : Aeneas.Std.Array U64 3#usize) :
    build.ext_constant s v = ok ((Build.extConstant fun i => wordOf (v.val.getD i 0#u64)).run s) := rfl
@[simp] theorem build_ext_add (a b : ExtVar) : build.ext_add s a b = ok ((Build.extAdd a b).run s) := rfl
@[simp] theorem build_ext_sub (a b : ExtVar) : build.ext_sub s a b = ok ((Build.extSub a b).run s) := rfl
@[simp] theorem build_ext_mul (a b : ExtVar) : build.ext_mul s a b = ok ((Build.extMul a b).run s) := rfl
@[simp] theorem build_ext_scale (a : ExtVar) (b : Var) : build.ext_scale s a b = ok ((Build.extScale a b).run s) :=
  rfl
@[simp] theorem build_bits (a : Var) (count : Usize) (what : Str) :
    build.bits s a count what = ok (checked ((vecOfFn (val_le_max count)) <$> Build.bits a count.val) s) := rfl
@[simp] theorem build_select (bit z o : Var) : build.select s bit z o = ok ((Build.select bit z o).run s) := rfl
@[simp] theorem build_permute (state : Aeneas.Std.Array Var 16#usize) :
    build.permute s state = ok (((lanes16 ·) <$> Build.permute (lanesOf state)).run s) := rfl

end

/-! ## Views -/

/-- A run on `build` as the copy's run: the value and the builder. -/
def built {α : Type} (x : α × Builder) : α × Builder := x

theorem built_injective {α : Type} : Function.Injective (@built α) := fun _ _ h => h

/-- A run that returns a translated `Vec` of handles, as the copy's array. -/
def builtVec (x : alloc.vec.Vec ExtVar × Builder) : _root_.Array ExtVar × Builder := (x.1.val.toArray, x.2)

theorem builtVec_injective : Function.Injective builtVec := by
  rintro ⟨⟨⟨n, l, hb⟩⟩, s⟩ ⟨⟨⟨m, k, hc⟩⟩, t⟩ h
  simp only [builtVec, Prod.mk.injEq, alloc.vec.Vec.val, Slice.val] at h
  obtain ⟨hl, rfl⟩ := h
  replace hl := congrArg _root_.Array.toList hl
  have hn : n = m := by
    rw [← Aeneas.Data.ListN.ListN_length l, ← Aeneas.Data.ListN.ListN_length k]
    exact congrArg List.length hl
  subst hn
  rw [Aeneas.Data.ListN.ListN.toList_inj l k hl]

/-- A translated `Kx` run as the copy's `KxVar` run. -/
def builtKx (x : algebra.Kx Builder Var ExtVar × Builder) : KxVar × Builder := (⟨x.1.re, x.1.im⟩, x.2)

theorem builtKx_injective : Function.Injective builtKx := by
  rintro ⟨⟨a, b⟩, s⟩ ⟨⟨c, d⟩, t⟩ h
  simp only [builtKx, Prod.mk.injEq, KxVar.mk.injEq] at h
  obtain ⟨⟨rfl, rfl⟩, rfl⟩ := h
  rfl

/-! ## `Sound` and `Complete` from an equation -/

theorem sound_eq {α β γ : Type} {f : α → Result β} {view : β → γ} {copy : α → γ} (g : α → β)
    (eq : ∀ x, f x = ok (g x)) (hv : ∀ x, view (g x) = copy x) : Sound f view copy := by
  intro x y h
  rw [eq, Result.ok.injEq] at h
  subst h
  exact hv x

theorem complete_eq {α β γ : Type} {f : α → Result β} {view : β → γ} {copy : α → γ} (g : α → β)
    (eq : ∀ x, f x = ok (g x)) (hv : ∀ x, view (g x) = copy x) (w : α) : Complete (fun _ => True) f view copy :=
  complete_of (sound_eq g eq hv) ⟨w, trivial⟩ fun x _ => ⟨g x, eq x⟩

/-- The empty slice, a witness. -/
def emptySlice {α : Type} : Slice α := Slice.from [] (by simp)

/-! ## Facts the proofs share -/

theorem id_bind {α β : Type} (x : α) (f : α → Id β) : (x >>= f : Id β) = f x := rfl

theorem uncurry_eq {α β γ : Type} (f : α → β → γ) (p : α × β) : uncurry f p = f p.1 p.2 := rfl

theorem slice_index {α : Type} (d : Slice α) (t : Usize) (ht : t.val < d.val.length) :
    Slice.index_usize d t = ok d.val[t.val] := by
  unfold Slice.index_usize
  rw [Slice.getElem?_Usize_eq, List.getElem?_eq_getElem ht]

theorem add_one_val (t : Usize) (h : t.val < Usize.max) : (core.num.Usize.wrapping_add t 1#usize).val = t.val + 1 := by
  have : t.val + 1 < Usize.size := by scalar_tac
  simp [Nat.mod_eq_of_lt this]

theorem loop_cont {α β : Type} {body : α → Result (ControlFlow α β)} {a a' : α}
    (h : body a = ok (.cont a')) : loop body a = loop body a' := by
  rw [loop, h, bind_ok]

theorem loop_done {α β : Type} {body : α → Result (ControlFlow α β)} {a : α} {b : β}
    (h : body a = ok (.done b)) : loop body a = ok b := by
  rw [loop, h, bind_ok]

theorem gl_word_eq (a : field.Gl) : field.gl_word a = ok (redW a) := by
  simp only [field.gl_word, reduce_word_eq]

theorem wordOf_red (a : field.Gl) : wordOf (redW a) = wordOf a :=
  (show Exact (redW a) (wordOf a) from toU_red_word a).wordOf

/-- A translated loop over `xs` whose body, at index `i < xs.length`, runs `f xs[i]` on the
builder and the accumulator, as the copy's `for x in xs do acc ← f x acc`. -/
theorem loop_forIn {α γ ρ β : Type} (body : α → Result (ControlFlow α β)) (pack : Builder → ρ → Usize → α)
    (stop : ρ → Builder → β) (xs : List γ) (f : γ → ρ → Build ρ) (hn : xs.length ≤ Usize.max)
    (hstep : ∀ s r (i : Usize) (h : i.val < xs.length),
      body (pack s r i) = ok (.cont (pack ((f xs[i.val] r).run s).2 ((f xs[i.val] r).run s).1
        (core.num.Usize.wrapping_add i 1#usize))))
    (hstop : ∀ s r (i : Usize), xs.length ≤ i.val → body (pack s r i) = ok (.done (stop r s))) :
    ∀ n s r (i : Usize), xs.length - i.val = n →
      loop body (pack s r i) =
        ok (let p := (forIn (xs.drop i.val) r fun x r => ForInStep.yield <$> f x r).run s; stop p.1 p.2) := by
  intro n
  induction n with
  | zero =>
    intro s r i hi
    rw [loop_done (hstop s r i (by omega)), List.drop_eq_nil_of_le (by omega)]
    rfl
  | succ n ih =>
    intro s r i hi
    have hlt : i.val < xs.length := by omega
    have hmax : i.val < Usize.max := by omega
    rw [loop_cont (hstep s r i hlt), ih _ _ _ (by rw [add_one_val i hmax]; omega), add_one_val i hmax,
      List.drop_eq_getElem_cons hlt, List.forIn_cons]
    simp only [StateT.run_bind, StateT.run_map]
    rfl

/-- `loop_forIn` for a loop whose state `R` relates to the copy's accumulator and to the index
`k` of the copy's list, so that the state may hold more than the copy reads or count down. -/
theorem loop_forIn_rel {α γ ρ β : Type} (body : α → Result (ControlFlow α β)) (R : α → Builder → ρ → ℕ → Prop)
    (stop : ρ → Builder → β) (xs : List γ) (f : γ → ρ → Build ρ)
    (hstep : ∀ a s r k (h : k < xs.length), R a s r k →
      ∃ a', body a = ok (.cont a') ∧ R a' ((f xs[k] r).run s).2 ((f xs[k] r).run s).1 (k + 1))
    (hstop : ∀ a s r k, R a s r k → xs.length ≤ k → body a = ok (.done (stop r s))) :
    ∀ n a s r k, R a s r k → xs.length - k = n →
      loop body a = ok (let p := (forIn (xs.drop k) r fun x r => ForInStep.yield <$> f x r).run s; stop p.1 p.2) := by
  intro n
  induction n with
  | zero =>
    intro a s r k hR hk
    rw [loop_done (hstop a s r k hR (by omega)), List.drop_eq_nil_of_le (by omega)]
    rfl
  | succ n ih =>
    intro a s r k hR hk
    have hlt : k < xs.length := by omega
    obtain ⟨a', hbody, hR'⟩ := hstep a s r k hlt hR
    rw [loop_cont hbody, ih _ _ _ _ hR' (by omega), List.drop_eq_getElem_cons hlt, List.forIn_cons]
    simp only [StateT.run_bind, StateT.run_map]
    rfl

/-- `loop_forIn_rel` for a loop whose body can fail: if the loop returns `y`, then `Q` holds of `y`
and the copy's run. -/
theorem loop_forIn_rel_sound {α γ ρ β : Type} (body : α → Result (ControlFlow α β))
    (R : α → Builder → ρ → ℕ → Prop) (Q : β → ρ → Builder → Prop) (xs : List γ) (f : γ → ρ → Build ρ)
    (hstep : ∀ a s r k c (h : k < xs.length), R a s r k → body a = ok c →
      ∃ a', c = .cont a' ∧ R a' ((f xs[k] r).run s).2 ((f xs[k] r).run s).1 (k + 1))
    (hstop : ∀ a s r k c, R a s r k → xs.length ≤ k → body a = ok c → ∃ y, c = .done y ∧ Q y r s) :
    ∀ n a s r k y, R a s r k → xs.length - k = n → loop body a = ok y →
      (let p := (forIn (xs.drop k) r fun x r => ForInStep.yield <$> f x r).run s; Q y p.1 p.2) := by
  intro n
  induction n with
  | zero =>
    intro a s r k y hR hk h
    rw [loop] at h
    obtain ⟨c, hc, h⟩ := bind_eq_ok h
    obtain ⟨y', rfl, hQ⟩ := hstop a s r k c hR (by omega) hc
    simp only [Result.ok.injEq] at h
    rw [← h, List.drop_eq_nil_of_le (by omega)]
    exact hQ
  | succ n ih =>
    intro a s r k y hR hk h
    have hlt : k < xs.length := by omega
    rw [loop] at h
    obtain ⟨c, hc, h⟩ := bind_eq_ok h
    obtain ⟨a', rfl, hR'⟩ := hstep a s r k c hlt hR hc
    have := ih _ _ _ _ y hR' (by omega) h
    rw [List.drop_eq_getElem_cons hlt, List.forIn_cons]
    simp only [StateT.run_bind, StateT.run_map]
    exact this

end CompressionLeanRust
