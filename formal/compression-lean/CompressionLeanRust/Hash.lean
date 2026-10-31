import CompressionLeanRust.Algebra

/-! The translated `compress` of `neo_verified::hash`, run on `build`, equals its copy in
`CompressionLean/Ports/Circuit/Hash.lean`. A copy reads a translated `[F; 4]` as `digestOf`
reads it. `compress` always returns, so its `Complete` premise is `True`. `hash_leaf`,
`merkle_root` and the duplex keep token hashes in `ports.json`. -/

namespace CompressionLeanRust

open Aeneas Aeneas.Std Result
open neo_verified
open CompressionLean
open CompressionLean.Circuit

/-- A translated `[F; 4]` as the copies' digest. -/
def digestOf (a : Aeneas.Std.Array Var 4#usize) : DigestVar := fun i => a.val.getD i 0

/-- A run that returns a translated digest, as the copy's run. -/
def builtDigest (x : Aeneas.Std.Array Var 4#usize × Builder) : DigestVar × Builder := (digestOf x.1, x.2)

theorem builtDigest_injective : Function.Injective builtDigest := by
  rintro ⟨a, s⟩ ⟨c, t⟩ h
  simp only [builtDigest, Prod.mk.injEq] at h
  obtain ⟨h, rfl⟩ := h
  congr 1
  apply Aeneas.Std.Array.ext
  apply List.ext_getElem (by rw [a.property, c.property])
  intro i hi _
  have := congrFun h ⟨i, by rw [a.property] at hi; exact hi⟩
  simp only [digestOf] at this
  rwa [List.getD_eq_getElem _ _ hi, List.getD_eq_getElem _ _ (by rw [c.property]; rw [a.property] at hi; exact hi)]
    at this

theorem digest_len : circuit.DIGEST_LEN = 4#usize := by
  unfold circuit.DIGEST_LEN
  rfl

/-- One step of `compress`'s lane loop. -/
def compressStep (left right : Aeneas.Std.Array Var 4#usize) (k : ℕ) (state : Aeneas.Std.Array Var 16#usize) :
    Aeneas.Std.Array Var 16#usize :=
  setN (setN state k left.val[k]!) (4 + k) right.val[k]!

theorem compress_loop_eq (left right : Aeneas.Std.Array Var 4#usize) (state : Aeneas.Std.Array Var 16#usize) :
    hash.compress_loop left right state 0#usize = ok (upto 4 1 (compressStep left right) 4 0 state) := by
  unfold hash.compress_loop
  refine loop_upto _ (fun (a : Aeneas.Std.Array Var 16#usize) (i : Usize) => (a, i)) 4 1 1#usize rfl (by omega)
    (by omega) (compressStep left right) id ?_ ?_ 4 state 0#usize 0 rfl rfl (by omega)
  · intro a i h _
    have h4 : 4 + i.val < 16 := by omega
    have hadd : (core.num.Usize.wrapping_add 4#usize i).val = 4 + i.val := by
      rw [wrapping_add_val _ _ (by scalar_tac)]
      rfl
    simp only [hash.compress_loop.body, digest_len, show i < 4#usize by scalar_tac, ite_true,
      index_eq left i (by scalar_tac), index_eq right i (by scalar_tac), update_eq a i _ (by scalar_tac), bind_ok,
      lift]
    rw [update_eq _ _ _ (by rw [hadd]; scalar_tac), hadd, bind_ok]
    rfl
  · intro a i h
    simp only [hash.compress_loop.body, digest_len, show ¬ i < 4#usize by scalar_tac, ite_false]
    rfl

theorem compress_lanes (left right : Aeneas.Std.Array Var 4#usize) (z : Var) :
    lanesOf (upto 4 1 (compressStep left right) 4 0 (Aeneas.Std.Array.repeat 16#usize z)) = fun (lane : Fin 16) =>
      if h : lane.val < 4 then digestOf left ⟨lane, h⟩
      else if h' : lane.val < 8 then digestOf right ⟨lane - 4, by omega⟩ else z := by
  have hl := left.property
  have hr := right.property
  obtain ⟨l, hl'⟩ := left
  obtain ⟨r, hr'⟩ := right
  funext lane
  fin_cases lane <;> simp [lanesOf, upto, compressStep, digestOf, Aeneas.Std.Array.repeat]

theorem compress_eq (s : Builder) (left right : Aeneas.Std.Array Var 4#usize) :
    ∃ y, hash.compress build s left right = ok y ∧
      builtDigest y = (compress (digestOf left) (digestOf right)).run s := by
  obtain ⟨⟨z, s1⟩, hz⟩ : ∃ p, (Build.const (wordOf 0#u64)).run s = p := ⟨_, rfl⟩
  obtain ⟨⟨out, s2⟩, hp⟩ : ∃ p, (Build.permute (lanesOf (upto 4 1 (compressStep left right) 4 0
      (Aeneas.Std.Array.repeat 16#usize z)))).run s1 = p := ⟨_, rfl⟩
  have hm : ((fun x => lanes16 x) <$> Build.permute (lanesOf (upto 4 1 (compressStep left right) 4 0
      (Aeneas.Std.Array.repeat 16#usize z)))).run s1 = (lanes16 out, s2) := by
    rw [StateT.run_map, hp]
    rfl
  unfold hash.compress
  simp only [build_constant, hz, bind_ok, uncurry_eq, compress_loop_eq, build_permute, hm,
    index_eq (lanes16 out) 0#usize (by scalar_tac), index_eq (lanes16 out) 1#usize (by scalar_tac),
    index_eq (lanes16 out) 2#usize (by scalar_tac), index_eq (lanes16 out) 3#usize (by scalar_tac)]
  refine ⟨_, rfl, ?_⟩
  simp only [compress, StateT.run_bind, show (0 : Replay.Word) = wordOf 0#u64 from rfl, hz, id_bind]
  rw [← compress_lanes, hp]
  simp only [builtDigest]
  refine Prod.ext ?_ rfl
  funext i
  fin_cases i <;> rfl

theorem compress_sound : Sound (fun x : Builder × Aeneas.Std.Array Var 4#usize × Aeneas.Std.Array Var 4#usize =>
    hash.compress build x.1 x.2.1 x.2.2) builtDigest (fun x => (compress (digestOf x.2.1) (digestOf x.2.2)).run x.1) := by
  rintro ⟨s, l, r⟩ y h
  obtain ⟨y', h', hv⟩ := compress_eq s l r
  simp only at h
  rw [h', Result.ok.injEq] at h
  subst h
  exact hv

theorem compress_complete : Complete (fun _ => True)
    (fun x : Builder × Aeneas.Std.Array Var 4#usize × Aeneas.Std.Array Var 4#usize => hash.compress build x.1 x.2.1 x.2.2)
    builtDigest (fun x => (compress (digestOf x.2.1) (digestOf x.2.2)).run x.1) :=
  complete_of compress_sound
    ⟨({}, Aeneas.Std.Array.make 4#usize [0, 0, 0, 0], Aeneas.Std.Array.make 4#usize [0, 0, 0, 0]), trivial⟩
    fun x _ => let ⟨y, h, _⟩ := compress_eq x.1 x.2.1 x.2.2; ⟨y, h⟩

end CompressionLeanRust
