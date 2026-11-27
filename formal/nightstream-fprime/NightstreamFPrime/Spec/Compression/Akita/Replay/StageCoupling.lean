import NightstreamFPrime.Spec.Compression.Akita.Extract.Present
import NightstreamFPrime.Spec.Compression.Akita.Key20Tower
import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinBits
import NightstreamFPrime.Spec.Compression.Akita.Replay.SamplerFairFold

/-! The stages of the pinned Akita code (revision `d50f700`) as images of the answers of the
model (Akita, ePrint 2026/1983, revision of 2026-11-23, Section 8.2 `[KECCAK]`, (146) and
Appendix E.1): the slot reading of a squeeze as a theorem.

- `bits20_le_word`, `bits20_max`: the largest work bit count of the opening of `key20.aks` is
  11, so the count `card_worked` of the work check is exact for each stage.
- `exists_stage_coupling`: a stage with a work check and a draw. When the strings that pass the
  work check are not more probable than the work slots, and each reading of the draw of the code
  is not more probable than the same reading of the model, a map `φ` from an answer and the
  squeezed strings to the squeezed strings exists whose fibres have the size `|Reply|`
  (`exists_reply_coupling`), under which each reading of the code that does not stop is the
  reading of the model. With `expect_fibres`, the strings `φ (reply, u)` of a uniform answer
  and independent uniform strings are uniform.
- The laws of the readers of the model: `card_workZeros_mul`, `card_work_sample` (a work slot
  and a fair challenge), `card_restrict`, `card_view_decode` (the fold decoders at different
  coordinates).
- `stage_field`: the stage of a work check (`worked` on 8 bytes) and of `decodePool`.
  `stage_field_one` is the instance for one challenge of `E`.
- `stage_fold`: the stage of a work check and of the draw `sample` of one fold group
  (`FoldCoupling`), for a decoder of the model that is uniform on the challenges of the code.
  `uniform_open`, `uniform_kept64`, `uniform_kept128` state this for the decoders of
  `CodeSlot`, and `stage_fold_open`, `stage_fold_kept64`, `stage_fold_kept128`,
  `stage_fold_root` are the instances.
- `stage_folds`: the stage of a work check and of the draws of a family of fold groups, one
  squeeze for each group (`FoldsCoupling`). `stage_folds_root` is the instance for the three
  groups of the root of `key20.aks`.
- `bits20_pos`: each stage of the opening of `key20.aks` has a work bit or more, so each stage
  has a work squeeze. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.RandomOracle (mass)
open Bytes

/-- Each work bit count of the opening of `key20.aks` is at most 11. -/
theorem bits20_le_word : ∀ bits ∈ bits20.flatten, bits ≤ 11 := by decide +kernel

/-- Each stage of the opening of `key20.aks` has a work bit or more. -/
theorem bits20_pos : ∀ bits ∈ bits20.flatten, 1 ≤ bits := by decide +kernel

/-- A stage of the opening of `key20.aks` has 11 work bits. -/
theorem bits20_max : 11 ∈ bits20.flatten := by decide +kernel

section Core

variable {Coord Slot Y₁ Y₂ T : Type} [Fintype Coord] [DecidableEq Coord] [Fintype Slot]
  [Fintype Y₁] [Fintype Y₂]

/-- The count of the events of the form `mass {r | p r} = 1/n`. -/
theorem card_of_mass {R : Type} [Fintype R] [Nonempty R] (p : R → Prop) {n : ℕ}
    (positive : 0 < n) (law : mass {r | p r} = 1 / n) :
    Nat.card {r // p r} * n = Fintype.card R := by
  rw [Extract.mass_eq_card, div_eq_div_iff (Nat.cast_ne_zero.mpr Fintype.card_ne_zero)
    (Nat.cast_ne_zero.mpr positive.ne'), one_mul] at law
  exact_mod_cast law

omit [Fintype Coord] [DecidableEq Coord] [Fintype Slot] in
/-- The work slot and the view of an answer are independent, as a count. -/
theorem card_reply_pair (Z : Finset Slot) (Q : (Coord → Slot) → Prop) :
    Nat.card {reply : Option Coord → Slot // reply none ∈ Z ∧ Q (Extract.view reply)} =
      Z.card * Nat.card {v : Coord → Slot // Q v} :=
  (Nat.card_congr ((Equiv.subtypeEquiv
    (q := fun pair : Slot × (Coord → Slot) => pair.1 ∈ Z ∧ Q pair.2)
    (Equiv.piOptionEquivProd (β := fun _ => Slot)) fun _ => Iff.rfl).trans
      Equiv.subtypeProdEquivProd)).trans
    (by rw [Nat.card_prod, Nat.card_eq_fintype_card (α := {s // s ∈ Z}), Fintype.card_coe])

/-- A stage with a work check and a draw: when the strings that pass the work check are not more
probable than the work slots of `Z`, and each reading of the draw of the code is not more
probable than the same reading of the model, a map from the pairs of an answer and two strings
to the two strings exists whose fibres have one size, under which each reading of the code that
does not stop is the reading of the model. -/
theorem exists_stage_coupling (Z : Finset Slot) (W : Y₁ → Prop)
    (work : Nat.card {y // W y} * Fintype.card Slot ≤ Z.card * Fintype.card Y₁)
    (m : (Coord → Slot) → T) (r : Y₂ → Option T)
    (draw : ∀ t, Nat.card {y // r y = some t} * Fintype.card (Coord → Slot) ≤
      Nat.card {v // m v = t} * Fintype.card Y₂) :
    ∃ φ : (Option Coord → Slot) × (Y₁ × Y₂) → Y₁ × Y₂,
      (∀ y, Nat.card {x // φ x = y} = Fintype.card (Option Coord → Slot)) ∧
      ∀ reply u t, W (φ (reply, u)).1 → r (φ (reply, u)).2 = some t →
        reply none ∈ Z ∧ m (Extract.view reply) = t := by
  classical
  suffices dominated : ∀ t' : Option T,
      Nat.card {y : Y₁ × Y₂ // (if W y.1 then (r y.2).map some else none) = some t'} *
          Fintype.card (Option Coord → Slot) ≤
        Nat.card {reply : Option Coord → Slot //
          (if reply none ∈ Z then some (m (Extract.view reply)) else none) = t'} *
          Fintype.card (Y₁ × Y₂) by
    obtain ⟨φ, fibres, reads⟩ := exists_reply_coupling _ _ dominated
    refine ⟨φ, fibres, fun reply u t passed read => ?_⟩
    have decoded := reads reply u (some t) (by rw [ite_eq_left passed, read]; rfl)
    by_cases zero : reply none ∈ Z
    · rw [ite_eq_left zero] at decoded
      exact ⟨zero, Option.some.inj decoded⟩
    · rw [ite_eq_right zero] at decoded
      cases decoded
  intro t'
  cases t' with
  | none =>
      have empty : IsEmpty {y : Y₁ × Y₂ //
          (if W y.1 then (r y.2).map some else none) = some none} :=
        ⟨fun y => by
          have read := y.2
          by_cases passed : W y.1.1
          · rw [ite_eq_left passed] at read
            cases found : r y.1.2 <;> simp [found] at read
          · rw [ite_eq_right passed] at read
            cases read⟩
      rw [Nat.card_of_isEmpty, Nat.zero_mul]
      exact Nat.zero_le _
  | some t =>
      have code : Nat.card {y : Y₁ × Y₂ //
          (if W y.1 then (r y.2).map some else none) = some (some t)} =
          Nat.card {y // W y} * Nat.card {y // r y = some t} :=
        (Nat.card_congr ((Equiv.subtypeEquivRight fun y : Y₁ × Y₂ =>
          (by by_cases passed : W y.1 <;> simp [passed] :
            (if W y.1 then (r y.2).map some else none) = some (some t) ↔
              W y.1 ∧ r y.2 = some t)).trans
            (Equiv.subtypeProdEquivProd (p := W) (q := fun y => r y = some t)))).trans
          (Nat.card_prod _ _)
      have model : Nat.card {reply : Option Coord → Slot //
          (if reply none ∈ Z then some (m (Extract.view reply)) else none) = some t} =
          Z.card * Nat.card {v // m v = t} :=
        (Nat.card_congr (Equiv.subtypeEquivRight fun reply : Option Coord → Slot =>
          (by by_cases zero : reply none ∈ Z <;> simp [zero] :
            (if reply none ∈ Z then some (m (Extract.view reply)) else none) = some t ↔
              reply none ∈ Z ∧ m (Extract.view reply) = t))).trans
          (card_reply_pair Z fun v => m v = t)
      have replies : Fintype.card (Option Coord → Slot) =
          Fintype.card Slot * Fintype.card (Coord → Slot) :=
        (Fintype.card_congr (Equiv.piOptionEquivProd (β := fun _ => Slot))).trans
          (Fintype.card_prod _ _)
      rw [code, model, replies, Fintype.card_prod]
      exact (Nat.mul_mul_mul_comm _ _ _ _).trans_le ((Nat.mul_le_mul work (draw t)).trans_eq
        (Nat.mul_mul_mul_comm _ _ _ _))

end Core

section Model

variable {E : Type} [Fintype E] {Coord : Type} [Fintype Coord] [DecidableEq Coord]

/-- For `g ≤ 131` the work slots with `g` zero bits are exactly `|CodeSlot E| / 2^g`. -/
theorem card_workZeros_mul {g : ℕ} (le : g ≤ 131) :
    (Extract.workZeros (CodeSlot E) g).card * 2 ^ g = Fintype.card (CodeSlot E) := by
  rw [Extract.workZeros_card]
  refine Nat.div_mul_cancel ((pow_dvd_pow 2 le).trans ?_)
  simp only [CodeSlot, Fintype.card_prod, Fintype.card_fin]
  exact (Nat.dvd_mul_right _ _).mul_left _

/-- For `bits ≤ 64` the strings that pass `worked` are as probable as the work slots with `bits`
zero bits. -/
theorem card_worked_le {bits : ℕ} (le : bits ≤ 64) :
    Nat.card {y : Bytes 8 // worked (array y) bits = true} * Fintype.card (CodeSlot E) ≤
      (Extract.workZeros (CodeSlot E) bits).card * Fintype.card (Bytes 8) := by
  have whole : 2 ^ (64 - bits) * 2 ^ bits = 256 ^ 8 := by
    rw [← pow_add, Nat.sub_add_cancel le]
    norm_num
  rw [card_worked le, ← card_workZeros_mul (g := bits) (by omega), card_bytes, ← whole]
  exact (Nat.mul_left_comm _ _ _).le

variable [Nonempty E]

/-- The law of the readers of the model at a stage with a work and a draw of a challenge of
`C`: `|Reply| / (2^g·|C|)` answers have a work slot with `g` zero bits and the challenge `c`. -/
theorem card_work_sample {C : Type} [Fintype C] [Nonempty C] {g : ℕ} (le : g ≤ 131)
    (fair : ∃ decode : (Coord → CodeSlot E) → C, Extract.Fair decode) (c : C) :
    Nat.card {reply : Option Coord → CodeSlot E //
        reply none ∈ Extract.workZeros (CodeSlot E) g ∧
          Extract.sample (Coord → CodeSlot E) C (Extract.view reply) = c} *
        (2 ^ g * Fintype.card C) = Fintype.card (Option Coord → CodeSlot E) := by
  have view := card_of_mass (fun v : Coord → CodeSlot E =>
    Extract.sample (Coord → CodeSlot E) C v = c) Fintype.card_pos
    (Extract.Fair.mass_eq (Extract.sample_fair fair) c)
  rw [card_reply_pair (Extract.workZeros (CodeSlot E) g) fun v : Coord → CodeSlot E =>
    Extract.sample (Coord → CodeSlot E) C v = c, Nat.mul_mul_mul_comm, card_workZeros_mul le, view]
  exact ((Fintype.card_congr (Equiv.piOptionEquivProd (β := fun _ => CodeSlot E))).trans
    (Fintype.card_prod _ _)).symm

/-- A stage of the code with a work check and a draw of `count` field challenges from a pool of
`m` bytes is the image of one answer of the model and of independent strings: a map `φ` whose
fibres have the size `|Reply|` exists, under which a work check that passes and an output `out`
of `decodePool` are a work slot with `bits` zero bits and the challenge `δ out` of the model. -/
theorem stage_field_bytes {C : Type} [Fintype C] [Nonempty C] {bits : ℕ} (le : bits ≤ 64)
    (count m : ℕ)
    (size : Fintype.card C = fieldModulus ^ (2 * count))
    (fair : ∃ decode : (Coord → CodeSlot E) → C, Extract.Fair decode) (δ : Array Ext → C)
    (injective : ∀ bytes bytes' out out', decodePool bytes count = some out →
      decodePool bytes' count = some out' → δ out = δ out' → out = out') :
    ∃ φ : (Option Coord → CodeSlot E) × (Bytes 8 × Bytes m) →
        Bytes 8 × Bytes m,
      (∀ y, Nat.card {x // φ x = y} = Fintype.card (Option Coord → CodeSlot E)) ∧
      ∀ reply u out, worked (array (φ (reply, u)).1) bits = true →
        decodePool (array (φ (reply, u)).2) count = some out →
        reply none ∈ Extract.workZeros (CodeSlot E) bits ∧
          Extract.sample (Coord → CodeSlot E) C (Extract.view reply) = δ out := by
  have draw : ∀ c : C,
      Nat.card {y : Bytes m // (decodePool (array y) count).map δ = some c} *
          Fintype.card (Coord → CodeSlot E) ≤
        Nat.card {v : Coord → CodeSlot E // Extract.sample (Coord → CodeSlot E) C v = c} *
          Fintype.card (Bytes m) := by
    intro c
    have view := card_of_mass (fun v : Coord → CodeSlot E =>
      Extract.sample (Coord → CodeSlot E) C v = c) Fintype.card_pos
      (Extract.Fair.mass_eq (Extract.sample_fair fair) c)
    have code : Nat.card {y : Bytes m //
        (decodePool (array y) count).map δ = some c} * fieldModulus ^ (2 * count) ≤
        256 ^ m := by
      rcases isEmpty_or_nonempty {y : Bytes m //
          (decodePool (array y) count).map δ = some c} with empty | ⟨⟨y₀, read₀⟩⟩
      · rw [Nat.card_of_isEmpty, Nat.zero_mul]
        exact Nat.zero_le _
      · obtain ⟨out₀, decoded₀, rfl⟩ := Option.map_eq_some_iff.mp read₀
        have same (y : {y : Bytes m //
            (decodePool (array y) count).map δ = some (δ out₀)}) :
            decodePool (array y.1) count = some out₀ := by
          obtain ⟨out, decoded, equal⟩ := Option.map_eq_some_iff.mp y.2
          rw [decoded, injective _ _ _ _ decoded decoded₀ equal]
        exact (Nat.mul_le_mul_right _ (Nat.card_le_card_of_injective
          (fun y => (⟨y.1, same y⟩ : {y : Bytes m //
            decodePool (array y) count = some out₀}))
          fun _ _ equal => Subtype.ext (congrArg (fun z : {y : Bytes m //
            decodePool (array y) count = some out₀} => z.1) equal))).trans
          (card_decodePool_le _ _ out₀)
    rw [← view, size, card_bytes]
    exact (Nat.mul_left_comm _ _ _).trans_le (Nat.mul_le_mul_left _ code)
  obtain ⟨φ, fibres, reads⟩ := exists_stage_coupling (Coord := Coord)
    (Extract.workZeros (CodeSlot E) bits) (fun y : Bytes 8 => worked (array y) bits = true)
    (card_worked_le le) (Extract.sample (Coord → CodeSlot E) C)
    (fun y : Bytes m => (decodePool (array y) count).map δ) draw
  exact ⟨φ, fibres, fun reply u out passed read =>
    reads reply u (δ out) passed (congrArg (Option.map δ) read)⟩

/-- `stage_field_bytes` for the pool of the code, `poolBytes (2·count)` bytes. -/
theorem stage_field {C : Type} [Fintype C] [Nonempty C] {bits : ℕ} (le : bits ≤ 64) (count : ℕ)
    (size : Fintype.card C = fieldModulus ^ (2 * count))
    (fair : ∃ decode : (Coord → CodeSlot E) → C, Extract.Fair decode) (δ : Array Ext → C)
    (injective : ∀ bytes bytes' out out', decodePool bytes count = some out →
      decodePool bytes' count = some out' → δ out = δ out' → out = out') :
    ∃ φ : (Option Coord → CodeSlot E) × (Bytes 8 × Bytes (poolBytes (2 * count))) →
        Bytes 8 × Bytes (poolBytes (2 * count)),
      (∀ y, Nat.card {x // φ x = y} = Fintype.card (Option Coord → CodeSlot E)) ∧
      ∀ reply u out, worked (array (φ (reply, u)).1) bits = true →
        decodePool (array (φ (reply, u)).2) count = some out →
        reply none ∈ Extract.workZeros (CodeSlot E) bits ∧
          Extract.sample (Coord → CodeSlot E) C (Extract.view reply) = δ out :=
  stage_field_bytes le count _ size fair δ injective

end Model

section Fold

variable {Coord Slot : Type} [Fintype Coord] [DecidableEq Coord] [Fintype Slot] [Nonempty Slot]

/-- Each value of the slots at `W` different coordinates has `|Slot|^(|Coord| − W)` views. -/
theorem card_restrict_fin {W : ℕ} (block : Fin W → Coord) (injective : Function.Injective block)
    (p : Fin W → Slot) :
    Nat.card {v : Coord → Slot // (fun b => v (block b)) = p} * Fintype.card Slot ^ W =
      Fintype.card (Coord → Slot) := by
  have law := Extract.Fair.mass_eq (Extract.fair_slots block injective) p
  rw [Fintype.card_fun, Fintype.card_fin] at law
  exact card_of_mass (fun v : Coord → Slot => (fun b => v (block b)) = p)
    (pow_pos Fintype.card_pos W) law

variable {ι : Type} [Fintype ι] [DecidableEq ι]

omit [DecidableEq ι] in
/-- `card_restrict_fin` for the coordinates of a finite index type. -/
theorem card_restrict (place : ι → Coord) (injective : Function.Injective place) (p : ι → Slot) :
    Nat.card {v : Coord → Slot // (fun x => v (place x)) = p} *
      Fintype.card Slot ^ Fintype.card ι = Fintype.card (Coord → Slot) := by
  rw [← card_restrict_fin (fun k => place ((Fintype.equivFin ι).symm k))
    (injective.comp (Fintype.equivFin ι).symm.injective)
    fun k => p ((Fintype.equivFin ι).symm k)]
  congr 1
  exact Nat.card_congr (Equiv.subtypeEquivRight fun v =>
    ⟨fun same => funext fun k => congrFun same _, fun same => funext fun x => by
      have entry := congrFun same (Fintype.equivFin ι x)
      simpa using entry⟩)

/-- The law of the fold decoders of the model: the views whose slots at different coordinates
decode to `t` are the product of the slot counts of each coordinate, over `|Slot|^|ι|`. -/
theorem card_view_decode {X : ι → Type} (place : ι → Coord)
    (injective : Function.Injective place) (decode : ∀ x, Slot → Option (X x))
    (t : ∀ x, Option (X x)) :
    Nat.card {v : Coord → Slot // ∀ x, decode x (v (place x)) = t x} *
        Fintype.card Slot ^ Fintype.card ι =
      (∏ x, Nat.card {slot // decode x slot = t x}) * Fintype.card (Coord → Slot) := by
  have fibres (p : ι → Slot) :
      Nat.card {v : Coord → Slot // (fun x => v (place x)) = p} =
        Nat.card {v : Coord → Slot // (fun x => v (place x)) = fun _ => Classical.arbitrary Slot} :=
    Nat.eq_of_mul_eq_mul_right (pow_pos Fintype.card_pos (Fintype.card ι))
      ((card_restrict place injective p).trans (card_restrict place injective _).symm)
  have slots : Nat.card {p : ι → Slot // ∀ x, decode x (p x) = t x} =
      ∏ x, Nat.card {slot // decode x slot = t x} :=
    (Nat.card_congr (Equiv.subtypePiEquivPi (p := fun x slot => decode x slot = t x))).trans
      Nat.card_pi
  rw [card_preimage (fun (v : Coord → Slot) x => v (place x)) fibres
    fun p : ι → Slot => ∀ x, decode x (p x) = t x, slots, Nat.mul_right_comm,
    card_restrict place injective, Nat.mul_comm]

/-- `N·V ≤ M·Y` from the two counts over the slots `S`. -/
theorem le_of_slots {N V M Y S P : ℕ} (positive : 0 < S) (code : N * S ≤ P * Y)
    (model : M * S = P * V) : N * V ≤ M * Y := by
  refine Nat.le_of_mul_le_mul_right ?_ positive
  calc N * V * S = N * S * V := Nat.mul_right_comm _ _ _
    _ ≤ P * Y * V := Nat.mul_le_mul_right _ code
    _ = P * V * Y := Nat.mul_right_comm _ _ _
    _ = M * Y * S := by rw [← model, Nat.mul_right_comm]

end Fold

section FoldStage

variable {E : Type} [Fintype E]

/-- The challenges of a list at the coordinates below `W`, in the ring of degree `d`. -/
noncomputable def ringsAt (d : ℕ) [NeZero d] (W : ℕ) (cs : List Sparse) (b : Fin W) :
    Option (Rq d) :=
  (cs[(b : ℕ)]?).map (Sparse.ring d)

/-- Two lists of `W` challenges with one `ringsAt` have the same elements of the ring. -/
theorem map_ring_of_ringsAt {d : ℕ} [NeZero d] {W : ℕ} {l l' : List Sparse}
    (length : l.length = W) (length' : l'.length = W) (same : ringsAt d W l = ringsAt d W l') :
    l.map (Sparse.ring d) = l'.map (Sparse.ring d) := by
  refine List.ext_getElem? fun i => ?_
  rw [List.getElem?_map, List.getElem?_map]
  by_cases small : i < W
  · exact congrFun same ⟨i, small⟩
  · rw [List.getElem?_eq_none (by omega), List.getElem?_eq_none (by omega)]

/-- The draw of one fold group of the code against the slots of a decoder of the model that
gives each challenge of the code for `|CodeSlot E| / |Kept G|` slots: each reading `t` has at most
`256^messageBytes · ∏_b |{slot | decode slot = t b}| / |CodeSlot E|^coordinates` messages. -/
theorem draw_group {G : FoldGroup} [NeZero G.degree] (decode : CodeSlot E → Option (Rq G.degree))
    (uniform : ∀ x, (∃ w, slotDecode G w = some x) →
      Nat.card {slot // decode slot = some x} * Nat.card (Kept G) = Fintype.card (CodeSlot E))
    (t : Fin G.coordinates → Option (Rq G.degree)) :
    Nat.card {y : Bytes G.messageBytes //
        (sample G (array y)).map (ringsAt G.degree G.coordinates) = some t} *
        Fintype.card (CodeSlot E) ^ G.coordinates ≤
      (∏ b, Nat.card {slot : CodeSlot E // decode slot = t b}) * 256 ^ G.messageBytes := by
  rcases isEmpty_or_nonempty {y : Bytes G.messageBytes //
      (sample G (array y)).map (ringsAt G.degree G.coordinates) = some t} with
    empty | ⟨⟨y₀, read₀⟩⟩
  · rw [Nat.card_of_isEmpty, Nat.zero_mul]
    exact Nat.zero_le _
  · obtain ⟨cs₀, sampled₀, rfl⟩ := Option.map_eq_some_iff.mp read₀
    have length₀ := (sample_mem sampled₀).1
    have same (y : {y : Bytes G.messageBytes //
        (sample G (array y)).map (ringsAt G.degree G.coordinates) =
          some (ringsAt G.degree G.coordinates cs₀)}) :
        (sample G (array y.1)).map (List.map (Sparse.ring G.degree)) =
          some (cs₀.map (Sparse.ring G.degree)) := by
      obtain ⟨cs, sampled, equal⟩ := Option.map_eq_some_iff.mp y.2
      rw [sampled]
      exact congrArg some (map_ring_of_ringsAt (sample_mem sampled).1 length₀ equal)
    have code := (Nat.mul_le_mul_right (Nat.card (Kept G) ^ G.coordinates)
      (Nat.card_le_card_of_injective
        (fun y => (⟨y.1, same y⟩ : {y : Bytes G.messageBytes //
          (sample G (array y)).map (List.map (Sparse.ring G.degree)) =
            some (cs₀.map (Sparse.ring G.degree))}))
        fun _ _ equal => Subtype.ext (congrArg (fun z : {y : Bytes G.messageBytes //
          (sample G (array y)).map (List.map (Sparse.ring G.degree)) =
            some (cs₀.map (Sparse.ring G.degree))} => z.1) equal))).trans
      (card_sample_ring_le G _)
    have each (b : Fin G.coordinates) :
        Nat.card {slot : CodeSlot E // decode slot = ringsAt G.degree G.coordinates cs₀ b} *
          Nat.card (Kept G) = Fintype.card (CodeSlot E) := by
      have inside : (b : ℕ) < cs₀.length := b.isLt.trans_eq length₀.symm
      have member : Sparse.ring G.degree cs₀[(b : ℕ)] ∈ cs₀.map (Sparse.ring G.degree) :=
        List.mem_map_of_mem (List.getElem_mem inside)
      rw [← sample_first sampled₀] at member
      obtain ⟨w, -, found⟩ := List.mem_filterMap.mp (List.mem_of_mem_take member)
      have value : ringsAt G.degree G.coordinates cs₀ b =
          some (Sparse.ring G.degree cs₀[(b : ℕ)]) := by
        rw [ringsAt, List.getElem?_eq_getElem inside, Option.map_some]
      rw [value]
      exact uniform _ ⟨w, found⟩
    have product : (∏ b : Fin G.coordinates, Nat.card {slot : CodeSlot E //
          decode slot = ringsAt G.degree G.coordinates cs₀ b}) *
          Nat.card (Kept G) ^ G.coordinates = Fintype.card (CodeSlot E) ^ G.coordinates :=
      calc _ = ∏ b : Fin G.coordinates, (Nat.card {slot : CodeSlot E //
            decode slot = ringsAt G.degree G.coordinates cs₀ b} * Nat.card (Kept G)) := by
            rw [Finset.prod_mul_distrib, Finset.prod_const, Finset.card_univ, Fintype.card_fin]
        _ = ∏ _b : Fin G.coordinates, Fintype.card (CodeSlot E) :=
            Finset.prod_congr rfl fun b _ => each b
        _ = _ := by rw [Finset.prod_const, Finset.card_univ, Fintype.card_fin]
    rw [← product]
    exact (Nat.mul_left_comm _ _ _).trans_le (Nat.mul_le_mul_left _ code)

variable [Nonempty E] {Coord : Type} [Fintype Coord] [DecidableEq Coord]

/-- The statement of a stage of the code with a work check and the draw of one fold group as
the image of one answer of the model and of independent strings: a map `φ` whose fibres have the
size `|Reply|` exists, under which a work check that passes and the challenges `cs` of the
sampler are a work slot with `bits` zero bits and the challenges of the decoder `decode` of the
model at the coordinates `block`. -/
def FoldCoupling (E : Type) [Fintype E] {Coord : Type} [Fintype Coord] [DecidableEq Coord]
    (G : FoldGroup) [NeZero G.degree] (bits : ℕ) (block : Fin G.coordinates → Coord)
    (decode : CodeSlot E → Option (Rq G.degree)) : Prop :=
  ∃ φ : (Option Coord → CodeSlot E) × (Bytes 8 × Bytes G.messageBytes) →
      Bytes 8 × Bytes G.messageBytes,
    (∀ y, Nat.card {x // φ x = y} = Fintype.card (Option Coord → CodeSlot E)) ∧
    ∀ reply u cs, worked (array (φ (reply, u)).1) bits = true →
      sample G (array (φ (reply, u)).2) = some cs →
      reply none ∈ Extract.workZeros (CodeSlot E) bits ∧
        ∀ b : Fin G.coordinates, decode (reply (some (block b))) =
          (cs[(b : ℕ)]?).map (Sparse.ring G.degree)

/-- `FoldCoupling` holds for a decoder of the model that gives each challenge of the code for
`|CodeSlot E| / |Kept G|` slots (`uniform`). -/
theorem stage_fold {G : FoldGroup} [NeZero G.degree] {bits : ℕ} (le : bits ≤ 64)
    (block : Fin G.coordinates → Coord) (injective : Function.Injective block)
    (decode : CodeSlot E → Option (Rq G.degree))
    (uniform : ∀ x, (∃ w, slotDecode G w = some x) →
      Nat.card {slot // decode slot = some x} * Nat.card (Kept G) = Fintype.card (CodeSlot E)) :
    FoldCoupling E G bits block decode := by
  have draw : ∀ t : Fin G.coordinates → Option (Rq G.degree),
      Nat.card {y : Bytes G.messageBytes //
          (sample G (array y)).map (ringsAt G.degree G.coordinates) = some t} *
          Fintype.card (Coord → CodeSlot E) ≤
        Nat.card {v : Coord → CodeSlot E // (fun b => decode (v (block b))) = t} *
          Fintype.card (Bytes G.messageBytes) := by
    intro t
    have model := card_view_decode block injective (fun _ => decode) t
    rw [Fintype.card_fin] at model
    rw [card_bytes, Nat.card_congr (Equiv.subtypeEquivRight fun v : Coord → CodeSlot E =>
      (funext_iff : (fun b => decode (v (block b))) = t ↔ ∀ b, decode (v (block b)) = t b))]
    exact le_of_slots (pow_pos Fintype.card_pos G.coordinates) (draw_group decode uniform t) model
  obtain ⟨φ, fibres, reads⟩ := exists_stage_coupling (Coord := Coord)
    (Extract.workZeros (CodeSlot E) bits) (fun y : Bytes 8 => worked (array y) bits = true)
    (card_worked_le le) (fun (v : Coord → CodeSlot E) b => decode (v (block b)))
    (fun y : Bytes G.messageBytes =>
      (sample G (array y)).map (ringsAt G.degree G.coordinates)) draw
  refine ⟨φ, fibres, fun reply u cs passed sampled => ?_⟩
  obtain ⟨zero, decoded⟩ := reads reply u _ passed
    (congrArg (Option.map (ringsAt G.degree G.coordinates)) sampled)
  exact ⟨zero, fun b => congrFun decoded b⟩

/-- The statement of a stage of the code with a work check and the draws of the fold groups
`G i`, each one squeeze, as the image of one answer of the model and of independent strings: a
map `φ` whose fibres have the size `|Reply|` exists, under which a work check that passes and
the challenges `cs i` of the sampler on each message are a work slot with `bits` zero bits and
the challenges of the decoders `decode i` of the model at the coordinates `block i`. -/
def FoldsCoupling (E : Type) [Fintype E] {Coord : Type} [Fintype Coord] [DecidableEq Coord]
    {n : ℕ} (G : Fin n → FoldGroup) [∀ i, NeZero (G i).degree] (bits : ℕ)
    (block : ∀ i, Fin (G i).coordinates → Coord)
    (decode : ∀ i, CodeSlot E → Option (Rq (G i).degree)) : Prop :=
  ∃ φ : (Option Coord → CodeSlot E) × (Bytes 8 × ∀ i, Bytes (G i).messageBytes) →
      Bytes 8 × ∀ i, Bytes (G i).messageBytes,
    (∀ y, Nat.card {x // φ x = y} = Fintype.card (Option Coord → CodeSlot E)) ∧
    ∀ reply u (cs : Fin n → List Sparse), worked (array (φ (reply, u)).1) bits = true →
      (∀ i, sample (G i) (array ((φ (reply, u)).2 i)) = some (cs i)) →
      reply none ∈ Extract.workZeros (CodeSlot E) bits ∧
        ∀ i (b : Fin (G i).coordinates), decode i (reply (some (block i b))) =
          ((cs i)[(b : ℕ)]?).map (Sparse.ring (G i).degree)

/-- `FoldsCoupling` holds for decoders of the model at different coordinates that give each
challenge of the code for `|CodeSlot E| / |Kept (G i)|` slots (`uniform`). -/
theorem stage_folds {n : ℕ} (G : Fin n → FoldGroup) [∀ i, NeZero (G i).degree] {bits : ℕ}
    (le : bits ≤ 64) (block : ∀ i, Fin (G i).coordinates → Coord)
    (injective : Function.Injective fun x : Σ i, Fin (G i).coordinates => block x.1 x.2)
    (decode : ∀ i, CodeSlot E → Option (Rq (G i).degree))
    (uniform : ∀ i x, (∃ w, slotDecode (G i) w = some x) →
      Nat.card {slot // decode i slot = some x} * Nat.card (Kept (G i)) =
        Fintype.card (CodeSlot E)) :
    FoldsCoupling E G bits block decode := by
  classical
  have draw : ∀ t : ∀ i, Option (Fin (G i).coordinates → Option (Rq (G i).degree)),
      Nat.card {y : ∀ i, Bytes (G i).messageBytes //
          (if ∀ i, (sample (G i) (array (y i))).isSome then
            some fun i => (sample (G i) (array (y i))).map
              (ringsAt (G i).degree (G i).coordinates) else none) = some t} *
          Fintype.card (Coord → CodeSlot E) ≤
        Nat.card {v : Coord → CodeSlot E //
          (fun i => some fun b => decode i (v (block i b))) = t} *
          Fintype.card (∀ i, Bytes (G i).messageBytes) := by
    intro t
    by_cases all : ∀ i, (t i).isSome
    · choose s value using fun i => Option.isSome_iff_exists.mp (all i)
      obtain rfl : t = fun i => some (s i) := funext value
      have code : Nat.card {y : ∀ i, Bytes (G i).messageBytes //
          (if ∀ i, (sample (G i) (array (y i))).isSome then
            some fun i => (sample (G i) (array (y i))).map
              (ringsAt (G i).degree (G i).coordinates) else none) = some fun i => some (s i)} =
          ∏ i, Nat.card {y : Bytes (G i).messageBytes //
            (sample (G i) (array y)).map (ringsAt (G i).degree (G i).coordinates) =
              some (s i)} :=
        (Nat.card_congr ((Equiv.subtypeEquivRight fun y : ∀ i, Bytes (G i).messageBytes =>
          (⟨fun read i => (by
              by_cases present : ∀ i, (sample (G i) (array (y i))).isSome
              · rw [ite_eq_left present] at read
                exact congrFun (Option.some.inj read) i
              · rw [ite_eq_right present] at read
                cases read),
            fun each => (by
              rw [ite_eq_left fun i => by rw [← Option.isSome_map, each i]; rfl]
              exact congrArg some (funext each))⟩ :
            (if ∀ i, (sample (G i) (array (y i))).isSome then
              some fun i => (sample (G i) (array (y i))).map
                (ringsAt (G i).degree (G i).coordinates) else none) = (some fun i => some (s i)) ↔
              ∀ i, (sample (G i) (array (y i))).map
                (ringsAt (G i).degree (G i).coordinates) = some (s i))).trans
          (Equiv.subtypePiEquivPi (p := fun i (y : Bytes (G i).messageBytes) =>
            (sample (G i) (array y)).map (ringsAt (G i).degree (G i).coordinates) =
              some (s i))))).trans Nat.card_pi
      have model := card_view_decode (ι := Σ i, Fin (G i).coordinates)
        (X := fun x => Rq (G x.1).degree) (fun x => block x.1 x.2) injective
        (fun x => decode x.1) fun x => s x.1 x.2
      rw [Fintype.prod_sigma, Fintype.card_sigma] at model
      simp only [Fintype.card_fin] at model
      rw [← Finset.prod_pow_eq_pow_sum] at model
      have groups : (∏ i, Nat.card {y : Bytes (G i).messageBytes //
            (sample (G i) (array y)).map (ringsAt (G i).degree (G i).coordinates) =
              some (s i)}) * ∏ i, Fintype.card (CodeSlot E) ^ (G i).coordinates ≤
          (∏ i, ∏ b, Nat.card {slot : CodeSlot E // decode i slot = s i b}) *
            ∏ i, 256 ^ (G i).messageBytes := by
        rw [← Finset.prod_mul_distrib, ← Finset.prod_mul_distrib]
        exact Finset.prod_le_prod fun i _ => draw_group (decode i) (uniform i) (s i)
      have strings : Fintype.card (∀ i, Bytes (G i).messageBytes) =
          ∏ i, 256 ^ (G i).messageBytes := by
        rw [Fintype.card_pi]
        exact Finset.prod_congr rfl fun i _ => card_bytes _
      rw [code, strings, Nat.card_congr (Equiv.subtypeEquivRight fun v : Coord → CodeSlot E =>
        (⟨fun same x => congrFun (Option.some.inj (congrFun same x.1)) x.2,
          fun each => funext fun i => congrArg some (funext fun b => each ⟨i, b⟩)⟩ :
          (fun i => some fun b => decode i (v (block i b))) = (fun i => some (s i)) ↔
            ∀ x : Σ i, Fin (G i).coordinates, decode x.1 (v (block x.1 x.2)) = s x.1 x.2))]
      exact le_of_slots (Finset.prod_pos fun i _ => pow_pos Fintype.card_pos _) groups model
    · have empty : IsEmpty {y : ∀ i, Bytes (G i).messageBytes //
          (if ∀ i, (sample (G i) (array (y i))).isSome then
            some fun i => (sample (G i) (array (y i))).map
              (ringsAt (G i).degree (G i).coordinates) else none) = some t} :=
        ⟨fun y => all fun i => by
          have read := y.2
          by_cases present : ∀ i, (sample (G i) (array (y.1 i))).isSome
          · rw [ite_eq_left present] at read
            rw [← congrFun (Option.some.inj read) i, Option.isSome_map]
            exact present i
          · rw [ite_eq_right present] at read
            cases read⟩
      rw [Nat.card_of_isEmpty, Nat.zero_mul]
      exact Nat.zero_le _
  obtain ⟨φ, fibres, reads⟩ := exists_stage_coupling (Coord := Coord)
    (Extract.workZeros (CodeSlot E) bits) (fun y : Bytes 8 => worked (array y) bits = true)
    (card_worked_le le)
    (fun (v : Coord → CodeSlot E) i => some fun b => decode i (v (block i b)))
    (fun y : ∀ i, Bytes (G i).messageBytes =>
      if ∀ i, (sample (G i) (array (y i))).isSome then
        some fun i => (sample (G i) (array (y i))).map
          (ringsAt (G i).degree (G i).coordinates) else none) draw
  refine ⟨φ, fibres, fun reply u cs passed sampled => ?_⟩
  obtain ⟨zero, decoded⟩ := reads reply u
    (fun i => some (ringsAt (G i).degree (G i).coordinates (cs i))) passed (by
      rw [ite_eq_left fun i => by rw [sampled i]; rfl]
      exact congrArg some (funext fun i => by rw [sampled i]; rfl))
  exact ⟨zero, fun i b => congrFun (Option.some.inj (congrFun decoded i)) b⟩

end FoldStage

section Decoders

variable {E : Type} [Fintype E]

/-- `CodeSlot.open G` gives each challenge of the code for `|CodeSlot E| / 2^K` slots. -/
theorem uniform_open {G : FoldGroup} [NeZero G.degree] (fits : G.candidateBits ≤ 131)
    (unfiltered : G.filtered = false) (positive : 0 < G.rankSpace) (x : Rq G.degree)
    (found : ∃ w, slotDecode G w = some x) :
    Nat.card {slot : CodeSlot E // CodeSlot.open G fits slot = some x} * Nat.card (Kept G) =
      Fintype.card (CodeSlot E) := by
  obtain ⟨w, decoded⟩ := found
  have fibre : Nat.card {slot : CodeSlot E // CodeSlot.open G fits slot = some x} =
      Nat.card {slot : CodeSlot E // (splitWord fits slot.2.1).2 = w} :=
    Nat.card_congr (Equiv.subtypeEquivRight fun slot =>
      ⟨fun read => slotDecode_injective G _ _ x read decoded,
        fun same => (congrArg (slotDecode G) same).trans decoded⟩)
  rw [fibre, card_open_fibre fits w, card_kept_open unfiltered positive,
    ← Nat.card_eq_fintype_card (α := CodeSlot E), ← Fintype.card_fin (2 ^ G.candidateBits),
    ← Nat.card_eq_fintype_card (α := Fin (2 ^ G.candidateBits)), ← Nat.card_prod]
  exact Nat.card_congr
    { toFun := fun y => (y.1.1, (splitWord fits).symm (y.1.2.1, y.2), y.1.2.2)
      invFun := fun slot =>
        ((slot.1, (splitWord fits slot.2.1).1, slot.2.2), (splitWord fits slot.2.1).2)
      left_inv := fun y => by simp
      right_inv := fun slot => by simp }

/-- A challenge of the code of a group is the challenge of exactly one element of `Fill`, and
`Fill` has the accepted words. -/
theorem fill_decode_unique {G : FoldGroup} [NeZero G.degree] {x : Rq G.degree}
    (found : ∃ w, slotDecode G w = some x) :
    Nat.card (Fill G) = Nat.card (Kept G) ∧
      ∃ f : Fill G, ∀ f', Fill.decode G f' = some x ↔ f' = f := by
  obtain ⟨w, decoded⟩ := found
  have kept : (slotDecode G w).isSome := Option.isSome_iff_exists.mpr ⟨x, decoded⟩
  have : Nonempty (Kept G) := ⟨⟨w, kept⟩⟩
  refine ⟨card_fill, Sum.inl ⟨w, kept⟩, fun f' => ⟨fun read => ?_, fun same => ?_⟩⟩
  · rcases f' with w' | extra
    · exact congrArg Sum.inl (Subtype.ext (slotDecode_injective G w'.1 w x read decoded))
    · exact absurd read (by simp [Fill.decode])
  · rw [same]
    exact decoded

/-- `CodeSlot.kept64` gives each challenge of the code of the family `(64, 31, 11)` with the
filter for `|CodeSlot E| / |Kept|` slots. -/
theorem uniform_kept64 (coordinates ring : ℕ) (x : Rq 64)
    (found : ∃ w, slotDecode (⟨coordinates, ring, 64, 31, 11, true⟩ : FoldGroup) w = some x) :
    Nat.card {slot : CodeSlot E // CodeSlot.kept64 slot = some x} *
        Nat.card (Kept ⟨coordinates, ring, 64, 31, 11, true⟩) = Fintype.card (CodeSlot E) := by
  obtain ⟨fill, f, unique⟩ := fill_decode_unique (G := filtered64) found
  have fibre : Nat.card {slot : CodeSlot E // CodeSlot.kept64 slot = some x} =
      Nat.card {slot : CodeSlot E // slot.2.2.1 = f} :=
    Nat.card_congr (Equiv.subtypeEquivRight fun slot => unique slot.2.2.1)
  rw [fibre, card_kept64_fibre f]
  show _ * Nat.card (Kept filtered64) = _
  rw [← fill, ← Nat.card_prod, ← Nat.card_eq_fintype_card (α := CodeSlot E)]
  exact Nat.card_congr
    { toFun := fun y => (y.1.1, y.1.2.1, y.2, y.1.2.2)
      invFun := fun slot => ((slot.1, slot.2.1, slot.2.2.2), slot.2.2.1)
      left_inv := fun _ => rfl
      right_inv := fun _ => rfl }

/-- `CodeSlot.kept128` gives each challenge of the code of the family `(128, 31, 0)` with the
filter for `|CodeSlot E| / |Kept|` slots. -/
theorem uniform_kept128 (coordinates ring : ℕ) (x : Rq 128)
    (found : ∃ w, slotDecode (⟨coordinates, ring, 128, 31, 0, true⟩ : FoldGroup) w = some x) :
    Nat.card {slot : CodeSlot E // CodeSlot.kept128 slot = some x} *
        Nat.card (Kept ⟨coordinates, ring, 128, 31, 0, true⟩) = Fintype.card (CodeSlot E) := by
  obtain ⟨fill, f, unique⟩ := fill_decode_unique (G := filtered128) found
  have fibre : Nat.card {slot : CodeSlot E // CodeSlot.kept128 slot = some x} =
      Nat.card {slot : CodeSlot E // slot.2.2.2 = f} :=
    Nat.card_congr (Equiv.subtypeEquivRight fun slot => unique slot.2.2.2)
  rw [fibre, card_kept128_fibre f]
  show _ * Nat.card (Kept filtered128) = _
  rw [← fill, ← Nat.card_prod, ← Nat.card_eq_fintype_card (α := CodeSlot E)]
  exact Nat.card_congr
    { toFun := fun y => (y.1.1, y.1.2.1, y.1.2.2, y.2)
      invFun := fun slot => ((slot.1, slot.2.1, slot.2.2.1), slot.2.2.2)
      left_inv := fun _ => rfl
      right_inv := fun _ => rfl }

variable [Nonempty E] {Coord : Type} [Fintype Coord] [DecidableEq Coord] {bits : ℕ}

/-- `FoldCoupling` for a group of the code without a filter, with `coordinates` challenges, and
the decoder `CodeSlot.open` of the model for its family. -/
theorem stage_fold_open (coordinates ring : ℕ) {G : FoldGroup} [NeZero G.degree]
    (fits : G.candidateBits ≤ 131) (unfiltered : G.filtered = false) (positive : 0 < G.rankSpace)
    (le : bits ≤ 64) (block : Fin coordinates → Coord) (injective : Function.Injective block) :
    FoldCoupling E ⟨coordinates, ring, G.degree, G.pm1, G.pm2, G.filtered⟩ bits block
      (CodeSlot.open G fits) :=
  stage_fold (G := ⟨coordinates, ring, G.degree, G.pm1, G.pm2, G.filtered⟩) le block injective
    (CodeSlot.open G fits)
    (uniform_open (G := ⟨coordinates, ring, G.degree, G.pm1, G.pm2, G.filtered⟩) fits unfiltered
      positive)

/-- `FoldCoupling` for a group of the code of the family `(64, 31, 11)` with the filter and the
decoder `CodeSlot.kept64` of the model. -/
theorem stage_fold_kept64 (coordinates ring : ℕ) (le : bits ≤ 64)
    (block : Fin coordinates → Coord) (injective : Function.Injective block) :
    FoldCoupling E ⟨coordinates, ring, 64, 31, 11, true⟩ bits block CodeSlot.kept64 :=
  stage_fold (G := ⟨coordinates, ring, 64, 31, 11, true⟩) le block injective CodeSlot.kept64
    (uniform_kept64 coordinates ring)

/-- `FoldCoupling` for a group of the code of the family `(128, 31, 0)` with the filter and the
decoder `CodeSlot.kept128` of the model. -/
theorem stage_fold_kept128 (coordinates ring : ℕ) (le : bits ≤ 64)
    (block : Fin coordinates → Coord) (injective : Function.Injective block) :
    FoldCoupling E ⟨coordinates, ring, 128, 31, 0, true⟩ bits block CodeSlot.kept128 :=
  stage_fold (G := ⟨coordinates, ring, 128, 31, 0, true⟩) le block injective CodeSlot.kept128
    (uniform_kept128 coordinates ring)

/-- `FoldCoupling` for the groups of the code of the root of `key20.aks` and the decoder
`Key20.rootDecode` of the model. -/
theorem stage_fold_root (g : Fin 3) (coordinates ring : ℕ) (le : bits ≤ 64)
    (block : Fin coordinates → Coord) (injective : Function.Injective block) :
    FoldCoupling E ⟨coordinates, ring, Key20.root.carrierDegree g, 31, Key20.rootTwos g, false⟩
      bits block (Key20.rootDecode E g) :=
  stage_fold_open coordinates ring (Key20.rootGroup_fits g) rfl (Key20.rootGroup_positive g) le
    block injective

/-- `FoldsCoupling` for groups of the code without a filter, with `coordinates i` challenges,
and the decoders `CodeSlot.open` of the model for their families. -/
theorem stage_folds_open {n : ℕ} (coordinates ring : Fin n → ℕ) (G : Fin n → FoldGroup)
    [∀ i, NeZero (G i).degree] (fits : ∀ i, (G i).candidateBits ≤ 131)
    (unfiltered : ∀ i, (G i).filtered = false) (positive : ∀ i, 0 < (G i).rankSpace)
    (le : bits ≤ 64) (block : ∀ i, Fin (coordinates i) → Coord)
    (injective : Function.Injective fun x : Σ i, Fin (coordinates i) => block x.1 x.2) :
    FoldsCoupling E (fun i => ⟨coordinates i, ring i, (G i).degree, (G i).pm1, (G i).pm2,
      (G i).filtered⟩) bits block fun i => CodeSlot.open (G i) (fits i) :=
  stage_folds (fun i => ⟨coordinates i, ring i, (G i).degree, (G i).pm1, (G i).pm2,
    (G i).filtered⟩) le block injective (fun i => CodeSlot.open (G i) (fits i)) fun i =>
    uniform_open (G := ⟨coordinates i, ring i, (G i).degree, (G i).pm1, (G i).pm2,
      (G i).filtered⟩) (fits i) (unfiltered i) (positive i)

/-- `FoldsCoupling` for the three groups of the code of the root of `key20.aks`, one squeeze
for each group, and the decoders `Key20.rootDecode` of the model. -/
theorem stage_folds_root (coordinates ring : Fin 3 → ℕ) (le : bits ≤ 64)
    (block : ∀ g, Fin (coordinates g) → Coord)
    (injective : Function.Injective fun x : Σ g, Fin (coordinates g) => block x.1 x.2) :
    FoldsCoupling E (fun g => ⟨coordinates g, ring g, Key20.root.carrierDegree g, 31,
      Key20.rootTwos g, false⟩) bits block (Key20.rootDecode E) :=
  stage_folds_open coordinates ring Key20.rootGroup Key20.rootGroup_fits (fun _ => rfl)
    Key20.rootGroup_positive le block injective

end Decoders

section FieldGuard

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E]

/-- Two elements of the code with canonical words and one element of `E` are equal. -/
theorem into_injective {z z' : Ext} (low : z.c0.word.toNat < fieldModulus)
    (high : z.c1.word.toNat < fieldModulus) (low' : z'.c0.word.toNat < fieldModulus)
    (high' : z'.c1.word.toNat < fieldModulus) (same : (z.into : E) = z'.into) : z = z' := by
  have word : ∀ a b : Fq, a.word.toNat < fieldModulus → b.word.toNat < fieldModulus →
      a.toZq = b.toZq → a = b := by
    intro a b small small' equal
    have value := (ZMod.natCast_eq_natCast_iff' _ _ _).mp equal
    rw [Nat.mod_eq_of_lt small, Nat.mod_eq_of_lt small'] at value
    show (⟨a.word⟩ : Fq) = ⟨b.word⟩
    rw [UInt64.toNat_inj.mp value]
  have coords : z.toE = z'.toE := (Degree2.coords (E := E)).symm.injective same
  show (⟨z.c0, z.c1⟩ : Ext) = ⟨z'.c0, z'.c1⟩
  rw [word _ _ low low' (congrArg QuadraticAlgebra.re coords),
    word _ _ high high' (congrArg QuadraticAlgebra.im coords)]

variable [Fintype E] {Coord : Type} [Fintype Coord] [DecidableEq Coord] [Nonempty Coord]

/-- `stage_field` for one challenge of `E`: the hypotheses of `stage_field` hold for `C = E`,
`count = 1` and the map of the one element of an output of `decodePool` to `E`. -/
theorem stage_field_one {bits : ℕ} (le : bits ≤ 64) :
    ∃ φ : (Option Coord → CodeSlot E) × (Bytes 8 × Bytes (poolBytes (2 * 1))) →
        Bytes 8 × Bytes (poolBytes (2 * 1)),
      (∀ y, Nat.card {x // φ x = y} = Fintype.card (Option Coord → CodeSlot E)) ∧
      ∀ reply u out, worked (array (φ (reply, u)).1) bits = true →
        decodePool (array (φ (reply, u)).2) 1 = some out →
        reply none ∈ Extract.workZeros (CodeSlot E) bits ∧
          Extract.sample (Coord → CodeSlot E) E (Extract.view reply) = (out[0]!).into :=
  stage_field le 1 Key20.card_field
    (Extract.exists_fair_field (Coord := Coord) (Slot := CodeSlot E) (E := E))
    (fun out => (out[0]!).into) fun _ _ out out' read read' same => by
      obtain ⟨size, canonical⟩ := decodePool_canonical read
      obtain ⟨size', canonical'⟩ := decodePool_canonical read'
      have inside : 0 < out.size := by omega
      have inside' : 0 < out'.size := by omega
      rw [getElem!_pos out 0 inside, getElem!_pos out' 0 inside'] at same
      have first := into_injective (canonical _ (Array.getElem_mem inside)).1
        (canonical _ (Array.getElem_mem inside)).2 (canonical' _ (Array.getElem_mem inside')).1
        (canonical' _ (Array.getElem_mem inside')).2 same
      refine Array.ext (size.trans size'.symm) fun i small small' => ?_
      obtain rfl : i = 0 := by omega
      exact first

end FieldGuard

end NightstreamFPrime.Spec.Compression.Akita.Replay
