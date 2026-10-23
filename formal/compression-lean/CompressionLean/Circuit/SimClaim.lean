import CompressionLean.Circuit.SimRing
import CompressionLean.Circuit.SimViews
import CompressionLean.Circuit.SimTerminal
import CompressionLean.Bridge.Relation

/-!
Owns the claim that layer 1 reads from the terminal package (`verify_with`,
`terminal/mod.rs:73-79`): its decoding to the replay's statement, and that
statement as the bridge reads it.

- `ClaimVars.view`, `ClaimView.decode`: the claim handles as the verifier's
  statement view (`claimView`, `TerminalCheck.lean`), and that view read
  through `w`.
- `isClaim_decode`: handles below `m` hold the decoded statement (`IsClaim`).
- `ClaimHolds` keeps only the flat handle lists, so the last two lemmas also
  take the row shape (`ClaimVars.Shaped`, `SimTerminal.lean`), which
  `Handoff` gives.
- `fits_decode`, `statementOf_decode`: under `ClaimHolds`, the decoded
  statement has the production sizes, and `statementOf` reads it as
  `Final.claimOf env`.
-/

namespace CompressionLean.Circuit

open CompressionLean.Replay (Gl D)
open CompressionLean.Circuit.Terminal (ClaimVars ClaimHolds HoldsValues pairHandles)
open NightstreamFPrime.Circuit (Env Expr)
open Fin.CommRing

/-- The terminal claim as the verifier's statement (`claimView`, `TerminalCheck.lean`). -/
def Terminal.ClaimVars.view (claim : ClaimVars) : ClaimView :=
  { commitment := claim.commitment, publicInput := claim.publicInput, point := claim.point,
    evalK := claim.evalK, evalA := claim.evalA }

/-- The statement of a claim view: each handle reads as its word under `w`. -/
def ClaimView.decode (w : Var → Gl) (v : ClaimView) : Replay.Statement :=
  { commitment := v.commitment.map (·.map w), publicInput := v.publicInput.map (·.map w),
    point := v.point.map (Prod.map w w), evalK := v.evalK.map (Prod.map w w),
    evalA := v.evalA.map (·.map (Prod.map w w)) }

section Decode

variable {F : Type} [CommRing F] {c : Ctx F} {w : Var → Gl} {m : ℕ}

private theorem isKs_map (hw : Decodes c w) {xs : Array KVar} (h : ∀ p ∈ xs, p.1 < m ∧ p.2 < m) :
    IsKs c m xs (xs.map (Prod.map w w)) := by
  refine ⟨(Array.size_map ..).symm, fun i hi => ?_⟩
  rw [getElem!_pos xs i hi, getElem!_pos _ i (by simpa using hi), Array.getElem_map]
  have := h _ (Array.getElem_mem hi)
  exact ⟨⟨this.1, hw _⟩, ⟨this.2, hw _⟩⟩

/-- Decoded handles below `m` hold the decoded statement. -/
theorem isClaim_decode (hw : Decodes c w) {v : ClaimView} (below : ∀ h ∈ v.words, h < m) :
    IsClaim c m v (v.decode w) := by
  have pair : ∀ p ∈ v.point ++ v.evalK ++ v.evalA.flatten, p.1 < m ∧ p.2 < m := fun p hp =>
    ⟨below _ (Array.mem_append_right _ (Array.mem_flatMap.mpr ⟨p, hp, by simp⟩)),
      below _ (Array.mem_append_right _ (Array.mem_flatMap.mpr ⟨p, hp, by simp⟩))⟩
  have row : ∀ {xss : Array (Array Var)}, (∀ h ∈ xss.flatten, h < m) →
      xss.size = (xss.map (·.map w)).size ∧ ∀ i < xss.size, IsWords c m xss[i]! (xss.map (·.map w))[i]! :=
    fun h => ⟨(Array.size_map ..).symm, fun i hi => by
      rw [getElem!_pos _ i hi, getElem!_pos _ i (by simpa using hi), Array.getElem_map]
      exact isWords_map hw fun j hj => h _ (Array.mem_flatten.mpr ⟨_, Array.getElem_mem hi, Array.getElem_mem hj⟩)⟩
  refine ⟨row fun h hh => below h (Array.mem_append_left _ (Array.mem_append_left _ hh)),
    row fun h hh => below h (Array.mem_append_left _ (Array.mem_append_right _ hh)),
    isKs_map hw fun p hp => pair p (by simp [hp]), isKs_map hw fun p hp => pair p (by simp [hp]),
    ⟨(Array.size_map ..).symm, fun j hj => ?_⟩⟩
  rw [getElem!_pos _ j hj, getElem!_pos _ j (by simpa [ClaimView.decode] using hj)]
  simp only [ClaimView.decode, Array.getElem_map]
  exact isKs_map hw fun p hp =>
    pair p (Array.mem_append_right _ (Array.mem_flatten.mpr ⟨_, Array.getElem_mem hj, hp⟩))

end Decode

section Holds

variable {c : Ctx NightstreamFPrime.Spec.F} {w : Var → Gl} {m : ℕ} {env : Env}

private theorem toF_word (g : Gl) : g.toF = ((word g : NightstreamFPrime.Spec.F).val : Bridge.Fp) := by
  rw [word, Fin.val_natCast, Nat.mod_eq_of_lt (show g.toNat < _ from g.canonical)]
  rfl

private theorem word_eq (hw : Decodes c w) {hs : List Var} {list : List Expr} (holds : HoldsValues c m env hs list)
    (k : ℕ) : Final.word (F := Bridge.Fp) list env k = ((hs[k]?.map w).getD 0).toF := by
  unfold HoldsValues at holds
  induction holds generalizing k with
  | nil => simp [Final.word]
  | cons hold rest ih =>
      cases k with
      | zero =>
          simp [Final.word]
          rw [toF_word, ← hw, hold.2]
      | succ k => simpa [Final.word] using ih k

private theorem holds_lt {hs : List Var} {list : List Expr} (holds : HoldsValues c m env hs list) :
    ∀ h ∈ hs, h < m := by
  unfold HoldsValues at holds
  induction holds with
  | nil => simp
  | cons hold rest ih =>
      intro h mem
      rcases List.mem_cons.mp mem with rfl | mem
      exacts [hold.1, ih h mem]

private theorem flatMap_getElem? {α β : Type} {f : α → List β} {n : ℕ} :
    ∀ {l : List α}, (∀ x ∈ l, (f x).length = n) → ∀ (i : ℕ) {j : ℕ}, j < n →
      (l.flatMap f)[n * i + j]? = l[i]?.bind fun x => (f x)[j]?
  | [], _, i, j, _ => by simp
  | x :: l, h, 0, j, hj => by
      rw [List.flatMap_cons, List.getElem?_append_left (by rw [h x (by simp)]; omega)]
      simp
  | x :: l, h, i + 1, j, hj => by
      rw [show n * (i + 1) + j = n + (n * i + j) by ring, List.flatMap_cons,
        List.getElem?_append_right (by rw [h x (by simp)]; omega), h x (by simp), Nat.add_sub_cancel_left]
      exact flatMap_getElem? (fun y hy => h y (by simp [hy])) i hj

private theorem length_flatMap_eq {α β : Type} {f : α → List β} {n : ℕ} :
    ∀ {l : List α}, (∀ x ∈ l, (f x).length = n) → (l.flatMap f).length = n * l.length
  | [], _ => by simp
  | x :: l, h => by
      rw [List.flatMap_cons, List.length_append, h x (by simp), length_flatMap_eq fun y hy => h y (by simp [hy]),
        List.length_cons]
      ring

private theorem outputs_length :
    CompressionLean.Terminal.outputs.commitment.length = 22 * D ∧
    CompressionLean.Terminal.outputs.publicInput.length = 5 * D ∧
    CompressionLean.Terminal.outputs.point.length = 2 * 28 ∧
    CompressionLean.Terminal.outputs.eval_K.length = 2 * D ∧
    CompressionLean.Terminal.outputs.eval_A.length = 4 * (2 * D) := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;>
    simp [CompressionLean.Terminal.outputs, CompressionLean.Terminal.extension, List.length_flatten, List.map_ofFn,
      Function.comp_def] <;> decide

private theorem pair_fst (xs : Array KVar) (k : ℕ) : (pairHandles xs)[2 * k]? = xs[k]?.map (·.1) := by
  have := flatMap_getElem? (f := fun p : KVar => [p.1, p.2]) (n := 2) (l := xs.toList) (fun _ _ => rfl) k
    (j := 0) (by decide)
  simp at this
  rw [pairHandles, this]
  cases xs[k]? <;> rfl

private theorem pair_snd (xs : Array KVar) (k : ℕ) : (pairHandles xs)[2 * k + 1]? = xs[k]?.map (·.2) := by
  have := flatMap_getElem? (f := fun p : KVar => [p.1, p.2]) (n := 2) (l := xs.toList) (fun _ _ => rfl) k
    (j := 1) (by decide)
  simp at this
  rw [pairHandles, this]
  cases xs[k]? <;> rfl

private theorem length_pairHandles (xs : Array KVar) : (pairHandles xs).length = 2 * xs.size := by
  rw [pairHandles, length_flatMap_eq (f := fun p : KVar => [p.1, p.2]) (n := 2) fun _ _ => rfl,
    Array.length_toList]

private theorem claim_sizes {claim : ClaimVars} (holds : ClaimHolds c m claim env) (shaped : claim.Shaped) :
    claim.commitment.size = 22 ∧ claim.publicInput.size = 5 ∧ claim.point.size = 28 ∧ claim.evalK.size = D ∧
      claim.evalA.size = 4 := by
  obtain ⟨hcom, hpub, hpt, hk, ha⟩ := holds
  obtain ⟨scom, spub, sa⟩ := shaped
  obtain ⟨lcom, lpub, lpt, lk, la⟩ := outputs_length
  have com := hcom.length_eq
  have pub := hpub.length_eq
  have pt := hpt.length_eq
  have k := hk.length_eq
  have a := ha.length_eq
  rw [length_flatMap_eq fun r hr => by rw [Array.length_toList, scom r (Array.mem_toList_iff.mp hr)],
    Array.length_toList, lcom] at com
  rw [length_flatMap_eq fun r hr => by rw [Array.length_toList, spub r (Array.mem_toList_iff.mp hr)],
    Array.length_toList, lpub] at pub
  rw [length_pairHandles, lpt] at pt
  rw [length_pairHandles, lk] at k
  rw [length_flatMap_eq fun r hr => by rw [length_pairHandles, sa r (Array.mem_toList_iff.mp hr)],
    Array.length_toList, la] at a
  simp only [D] at com pub k a ⊢
  omega

/-- The handles of the claim view are below `m`. -/
theorem claimHolds_words_lt {claim : ClaimVars} (holds : ClaimHolds c m claim env) :
    ∀ h ∈ claim.view.words, h < m := by
  obtain ⟨hcom, hpub, hpt, hk, ha⟩ := holds
  intro h mem
  simp [ClaimView.words, Terminal.ClaimVars.view] at mem
  rcases mem with mem | mem | ⟨a, b, mem, e⟩ | ⟨a, b, mem, e⟩ | ⟨a, b, mem, e⟩
  · obtain ⟨row, hrow, hh⟩ := Array.mem_flatten.mp mem
    exact holds_lt hcom h (List.mem_flatMap.mpr ⟨row, Array.mem_toList_iff.mpr hrow, Array.mem_toList_iff.mpr hh⟩)
  · obtain ⟨row, hrow, hh⟩ := Array.mem_flatten.mp mem
    exact holds_lt hpub h (List.mem_flatMap.mpr ⟨row, Array.mem_toList_iff.mpr hrow, Array.mem_toList_iff.mpr hh⟩)
  · exact holds_lt hpt h
      (List.mem_flatMap.mpr ⟨(a, b), Array.mem_toList_iff.mpr mem, by rcases e with rfl | rfl <;> simp⟩)
  · exact holds_lt hk h
      (List.mem_flatMap.mpr ⟨(a, b), Array.mem_toList_iff.mpr mem, by rcases e with rfl | rfl <;> simp⟩)
  · obtain ⟨row, hrow, hp⟩ := Array.mem_flatten.mp mem
    exact holds_lt ha h (List.mem_flatMap.mpr ⟨row, Array.mem_toList_iff.mpr hrow,
      List.mem_flatMap.mpr ⟨(a, b), Array.mem_toList_iff.mpr hp, by rcases e with rfl | rfl <;> simp⟩⟩)

/-- With the production sizes, the decoded statement fits the relation. -/
theorem fits_decode {claim : ClaimVars} (w : Var → Gl) {rel : Replay.Relation} (holds : ClaimHolds c m claim env)
    (shaped : claim.Shaped) (kappa : rel.kappa = 22) (blocks : rel.publicBlocks = 5) (point : rel.pointVariables = 28)
    (matrices : rel.matrices = 4) : (claim.view.decode w).fits rel = true := by
  obtain ⟨scom, spub, spt, sk, sa⟩ := claim_sizes holds shaped
  simp only [Replay.Statement.fits, ClaimView.decode, Terminal.ClaimVars.view, Bool.and_eq_true,
    Array.all_eq_true_iff_forall_mem]
  simp [scom, spub, spt, sk, sa, kappa, blocks, point, matrices]
  exact ⟨⟨shaped.1, shaped.2.1⟩, shaped.2.2⟩

private theorem get_map {α β : Type} [Inhabited β] (f : α → β) (xs : Array α) (i : ℕ) :
    (xs.map f)[i]! = (xs[i]?.map f).getD default := by
  rw [getElem!_def, Array.getElem?_map]
  cases xs[i]? <;> rfl

private theorem get_map_map {α β : Type} [Inhabited β] (f : α → β) (rows : Array (Array α)) (i j : ℕ) :
    ((rows.map (·.map f))[i]!)[j]! = ((rows[i]?.bind (·[j]?)).map f).getD default := by
  rw [get_map]
  cases rows[i]? with
  | none => exact getElem!_neg _ _ (Nat.not_lt_zero j)
  | some r => simp [get_map]

private theorem toF_default : (default : Gl).toF = 0 := Replay.Gl.toF_zero

private theorem row_toF {rows : Array (Array Var)} (shaped : ∀ row ∈ rows, row.size = D) (ρ l : ℕ) :
    (((rows.map (·.map w))[ρ]!)[l]!).toF =
      if l < D then (((rows.toList.flatMap Array.toList)[D * ρ + l]?.map w).getD 0).toF else 0 := by
  rw [get_map_map]
  split_ifs with hl
  · rw [flatMap_getElem? (fun r hr => by rw [Array.length_toList, shaped r (Array.mem_toList_iff.mp hr)]) ρ hl]
    simp
    rfl
  · cases h : rows[ρ]? with
    | none => exact toF_default
    | some r =>
        rw [Option.bind_some, Array.getElem?_eq_none (by rw [shaped r (Array.mem_of_getElem? h)]; omega)]
        exact toF_default

/-- The bridge reads the decoded statement as the claim of `env`. -/
theorem statementOf_decode (hw : Decodes c w) {claim : ClaimVars} (holds : ClaimHolds c m claim env)
    (shaped : claim.Shaped) : Bridge.statementOf (claim.view.decode w) = Final.claimOf (F := Bridge.Fp) env := by
  obtain ⟨-, -, spt, sk, -⟩ := claim_sizes holds shaped
  obtain ⟨hcom, hpub, hpt, hk, ha⟩ := holds
  obtain ⟨scom, spub, sa⟩ := shaped
  simp only [Bridge.statementOf, Final.claimOf, word_eq hw hcom, word_eq hw hpub, word_eq hw hpt, word_eq hw hk,
    word_eq hw ha, ← hpt.length_eq, length_pairHandles, ClaimView.decode, Terminal.ClaimVars.view,
    show NightstreamFPrime.Spec.ringDegree = D from rfl]
  congr 1
  · funext ρ l
    exact row_toF scom ρ l
  · funext ρ l
    exact row_toF spub ρ l
  · apply List.ext_getElem (by simp)
    intro k h1 h2
    simp at h1
    simp [pair_fst, pair_snd, h1]
  · funext l
    rw [get_map]
    split_ifs with hl
    · simp [pair_fst, pair_snd, sk, hl]
    · rw [Array.getElem?_eq_none (by omega)]
      exact Prod.ext toF_default toF_default
  · funext j l
    rw [get_map_map]
    split_ifs with hl
    · have r0 : (claim.evalA.toList.flatMap pairHandles)[2 * D * j + 2 * l]? =
          claim.evalA[j]?.bind fun r => r[l]?.map (·.1) := by
        have := flatMap_getElem? (f := pairHandles) (n := 2 * D)
          (fun r hr => by rw [length_pairHandles, sa r (Array.mem_toList_iff.mp hr)]) j (j := 2 * l) (by omega)
        simpa [pair_fst] using this
      have r1 : (claim.evalA.toList.flatMap pairHandles)[2 * D * j + 2 * l + 1]? =
          claim.evalA[j]?.bind fun r => r[l]?.map (·.2) := by
        have := flatMap_getElem? (f := pairHandles) (n := 2 * D)
          (fun r hr => by rw [length_pairHandles, sa r (Array.mem_toList_iff.mp hr)]) j (j := 2 * l + 1) (by omega)
        simpa [pair_snd, Nat.add_assoc] using this
      rw [r0, r1]
      cases claim.evalA[j]? with
      | none => rfl
      | some r =>
          simp only [Option.bind_some]
          cases r[l]? <;> rfl
    · cases h : claim.evalA[j]? with
      | none => exact Prod.ext toF_default toF_default
      | some r =>
          rw [Option.bind_some, Array.getElem?_eq_none (by rw [sa r (Array.mem_of_getElem? h)]; omega)]
          exact Prod.ext toF_default toF_default

end Holds

end CompressionLean.Circuit
