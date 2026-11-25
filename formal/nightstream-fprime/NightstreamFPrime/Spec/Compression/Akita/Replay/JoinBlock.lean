import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinRuns
import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinSlots

/-! The block of the stages of each level of `tower20` for an accepted proof of `key20.aks`, for
theorem A of the join (Akita revision `d50f700`; ePrint 2026/1983, revision of 2026-11-23,
Section 10.3; `docs/akita-work/w16/design.md` section 2).

- `block20_zeroth` .. `block20_fifth`: for each work slot function `work` and each wire message
  `wire₀` with the data `note20`, a block of the stages of the level exists (50, 69, 83, 81, 94
  and 94 stages from the stages 1, 51, 120, 203, 284 and 378): its messages have the data
  `note20` and the nonce 0, its answer `i` has the work slot `work (start + i)`, and each
  transcript with `wire₀` before the start and the block from there shows the fold challenges and
  the path of the code (`Tier.Shows`). The fold answer is `foldAnswer` of the slots of
  `slots20₀` .. `slots20₅`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Extract (Tier Presents view Prefix)

open Replayer

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]
  {Coord : Type} [Fintype Coord] [DecidableEq Coord] {total : ℕ}
  (o : Key20.Tower E (Key20.Wire E) Coord total)

local notation "code₁" =>
  (CodeSlot.open Key20.open64 Key20.open64_fits :
    CodeSlot E → Option (Rq Key20.level₁.opening.carrierDegree))

local notation "code₂" =>
  (CodeSlot.open Key20.open128 Key20.open128_fits :
    CodeSlot E → Option (Rq Key20.level₂.shape.innerDegree))

local notation "code₃" =>
  (CodeSlot.kept128 : CodeSlot E → Option (Rq Key20.level₃.shape.innerDegree))

local notation "code₄" =>
  (CodeSlot.kept64 : CodeSlot E → Option (Rq Key20.level₄.shape.innerDegree))

local notation "code₅" =>
  (CodeSlot.kept64 : CodeSlot E → Option (Rq Key20.level₅.shape.innerDegree))

local notation "zeroth" =>
  Extract.liftBatch o.β o.key o.openKey o.claim Key20.rootVars (And.left Key20.root_sized) 8
    Key20.quotientDigits Key20.root_room Key20.rootBatch (And.left (And.right Key20.root_sized))
    o.block (Key20.rootDecode E) o.read Key20.rootMass (0 + 1)

local notation "first" =>
  Tier.tensored (Extract.liftFold o.β₁ o.key₁ o.openKey₁ o.claim₁ Key20.level₁.vars
    (And.left Key20.level₁_sized) Key20.quotientDigits (And.left (And.right Key20.level₁_sized))
    Key20.level₁.batch (And.left (And.right (And.right Key20.level₁_sized))) o.block₁ code₁ o.read
    51 (Tier.finish zeroth)) Key20.level₁.tail

local notation "second" =>
  Tier.tensored (Extract.committedTrace (fold := Key20.level₂.fold) o.key₂ o.openKey₂ o.claim₂
    Key20.level₂.vars (And.left Key20.sized₂) Key20.level₂.batch (And.left (And.right Key20.sized₂))
    o.block₂ code₂ o.read 31 (Tier.finish first)) Key20.level₂.tail

local notation "third" =>
  Tier.tensored (Extract.committedTraceSq (fold := Key20.level₃.fold) o.key₃ o.openKey₃ o.claim₃
    Key20.level₃.vars (And.left Key20.sized₃) Key20.level₃.batch
    (And.left (And.right Key20.sized₃)) o.block₃ code₃ o.read 31 Key20.norm₃.operator
    Key20.norm₃.limit (Tier.finish second)) Key20.level₃.tail

local notation "fourth" =>
  Tier.tensored (Extract.committedTraceSq (fold := Key20.level₄.fold) o.key₄ o.openKey₄ o.claim₄
    Key20.level₄.vars (And.left Key20.sized₄) Key20.level₄.batch
    (And.left (And.right Key20.sized₄)) o.block₄ code₄ o.read 53 Key20.norm₄.operator
    Key20.norm₄.limit (Tier.finish third)) Key20.level₄.tail

local notation "fifth" =>
  Tier.tensored (Extract.committedTraceSq (fold := Key20.level₅.fold) o.key₅ o.openKey₅ o.claim₅
    Key20.level₅.vars (And.left Key20.sized₅) Key20.level₅.batch
    (And.left (And.right Key20.sized₅)) o.block₅ code₅ o.read 53 Key20.norm₅.operator
    Key20.norm₅.limit (Tier.finish fourth)) Key20.level₅.tail

variable (pub : Public) (proof : ByteArray) (messages : Array Message)
  (tLast : ℕ → Rq Key20.terminal.innerDegree) (eLast : Rq Key20.terminal.innerDegree)
  {block : Key20.root.Block → Coord} {block₁ : Fin Key20.level₁.shape.blocks → Coord}
  {block₂ : Fin Key20.level₂.shape.blocks → Coord} {block₃ : Fin Key20.level₃.shape.blocks → Coord}
  {block₄ : Fin Key20.level₄.shape.blocks → Coord} {block₅ : Fin Key20.level₅.shape.blocks → Coord}
  {blockL : Fin Key20.terminal.blocks → Coord}

/-- The block of the stages of the root fold of `tower20` for an accepted proof of `layout20`:
50 stages from the stage 1. -/
theorem block20_zeroth {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : verifyProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (points : ∀ g : Fin 3, ((pub.groups[g.val]?.map (·.point)).getD #[]).size = ![20, 16, 16] g)
    (values : (pub.groups.flatMap (·.value)).size = 3)
    (expanded : (pub.setupFor (fold20 0).setupCount).isSome)
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    (injective : Function.Injective block) (slots : 22 ≤ Fintype.card Coord) (work : ℕ → CodeSlot E)
    (wire₀ : Key20.Wire E) (data₀ : wire₀.data = note20 pub proof messages tLast eLast) :
    ∃ b : List ((Option Coord → CodeSlot E) × Key20.Wire E),
      b.length = 50 ∧ (∀ x ∈ b, x.2.data = note20 pub proof messages tLast eLast ∧ x.2.nonce = 0) ∧
      (∀ i (inside : i < b.length), b[i].1 none = work (1 + i)) ∧
      ∀ point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total, point.wire 1 = wire₀ →
        (point.rest.drop 1).take 50 = b →
        Tier.Shows (zeroth) (challenge20₀ pub proof messages) (path20 pub proof messages E 0) 0 point := by
  obtain ⟨started, length, -, -, -⟩ := zeroth_numbers o
  have placed : o.block = block := by subst tower; rfl
  obtain ⟨s, decodes⟩ := slots20₀ (E := E) layout accepted terminalSetup decoded
  obtain ⟨b, sized, fields, worked, shows⟩ := Tier.exists_block20 (zeroth) (by subst tower; rfl)
    (note20 pub proof messages tLast eLast) work
    (fun wire data => runs20_zeroth _ pub proof messages tLast eLast layout accepted terminalSetup
      decoded points values expanded tower wire data)
    (fun wire => (checks20_zeroth _ slots wire _).1) (fun wire => (checks20_zeroth _ slots wire _).2)
    (fits_path20 layout accepted terminalSetup decoded 0).1
    (Extract.foldAnswer o.block s (work (zeroth).start) fun _ => work 0) rfl
    (fun b => Extract.foldAnswer_decode _ (fun _ => rfl) (placed ▸ injective) decodes _ _ b) wire₀ data₀
  rw [length] at sized
  rw [started] at worked
  rw [started, length] at shows
  exact ⟨b, sized, fields, worked, shows⟩

/-- The block of the stages of the level 1 of `tower20` for an accepted proof of `layout20`:
69 stages from the stage 51. -/
theorem block20_first {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : verifyProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (expanded : (pub.setupFor (fold20 1).setupCount).isSome)
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    (injective₁ : Function.Injective block₁) (slots : 22 ≤ Fintype.card Coord) (work : ℕ → CodeSlot E)
    (wire₀ : Key20.Wire E) (data₀ : wire₀.data = note20 pub proof messages tLast eLast) :
    ∃ b : List ((Option Coord → CodeSlot E) × Key20.Wire E),
      b.length = 69 ∧ (∀ x ∈ b, x.2.data = note20 pub proof messages tLast eLast ∧ x.2.nonce = 0) ∧
      (∀ i (inside : i < b.length), b[i].1 none = work (51 + i)) ∧
      ∀ point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total, point.wire 51 = wire₀ →
        (point.rest.drop 51).take 69 = b →
        Tier.Shows (first) (challenge20₁ pub proof messages) (path20 pub proof messages E 1) 0 point := by
  obtain ⟨started, length, -, -, -⟩ := first20_numbers o
  have placed : o.block₁ = block₁ := by subst tower; rfl
  obtain ⟨s, decodes⟩ := slots20₁ (E := E) layout accepted terminalSetup decoded
  obtain ⟨b, sized, fields, worked, shows⟩ := Tier.exists_block20 (first) (by subst tower; rfl)
    (note20 pub proof messages tLast eLast) work
    (fun wire data => runs20_first _ pub proof messages tLast eLast layout accepted terminalSetup
      decoded expanded tower wire data)
    (fun wire => (checks20_first _ slots wire _).1) (fun wire => (checks20_first _ slots wire _).2)
    (fits_path20 layout accepted terminalSetup decoded 1).1
    (Extract.foldAnswer o.block₁ s (work (first).start) fun _ => work 0) rfl
    (fun b => Extract.foldAnswer_decode _ (fun _ => rfl) (placed ▸ injective₁) decodes _ _ b) wire₀ data₀
  rw [length] at sized
  rw [started] at worked
  rw [started, length] at shows
  exact ⟨b, sized, fields, worked, shows⟩

/-- The block of the stages of the level 2 of `tower20` for an accepted proof of `layout20`:
83 stages from the stage 120. -/
theorem block20_second {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : verifyProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (expanded : (pub.setupFor (fold20 2).setupCount).isSome)
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    (injective₂ : Function.Injective block₂) (slots : 22 ≤ Fintype.card Coord) (work : ℕ → CodeSlot E)
    (wire₀ : Key20.Wire E) (data₀ : wire₀.data = note20 pub proof messages tLast eLast) :
    ∃ b : List ((Option Coord → CodeSlot E) × Key20.Wire E),
      b.length = 83 ∧ (∀ x ∈ b, x.2.data = note20 pub proof messages tLast eLast ∧ x.2.nonce = 0) ∧
      (∀ i (inside : i < b.length), b[i].1 none = work (120 + i)) ∧
      ∀ point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total, point.wire 120 = wire₀ →
        (point.rest.drop 120).take 83 = b →
        Tier.Shows (second) (challengeTrace20 pub proof messages 2 Key20.level₂) (path20 pub proof messages E 2) 0 point := by
  obtain ⟨started, length, -, -, -⟩ := second20_numbers o
  have placed : o.block₂ = block₂ := by subst tower; rfl
  obtain ⟨s, decodes⟩ := slots20₂ (E := E) layout accepted terminalSetup decoded
  obtain ⟨b, sized, fields, worked, shows⟩ := Tier.exists_block20 (second) (by subst tower; rfl)
    (note20 pub proof messages tLast eLast) work
    (fun wire data => runs20_second _ pub proof messages tLast eLast layout accepted terminalSetup
      decoded expanded tower wire data)
    (fun wire => (checks20_second _ slots wire _).1) (fun wire => (checks20_second _ slots wire _).2)
    (fits_path20 layout accepted terminalSetup decoded 2).1
    (Extract.foldAnswer o.block₂ s (work (second).start) fun _ => work 0) rfl
    (fun b => Extract.foldAnswer_decode _ (fun _ => rfl) (placed ▸ injective₂) decodes _ _ b) wire₀ data₀
  rw [length] at sized
  rw [started] at worked
  rw [started, length] at shows
  exact ⟨b, sized, fields, worked, shows⟩

/-- The block of the stages of the level 3 of `tower20` for an accepted proof of `layout20`:
81 stages from the stage 203. -/
theorem block20_third {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : verifyProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (expanded : (pub.setupFor (fold20 3).setupCount).isSome)
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    (injective₃ : Function.Injective block₃) (slots : 22 ≤ Fintype.card Coord) (work : ℕ → CodeSlot E)
    (wire₀ : Key20.Wire E) (data₀ : wire₀.data = note20 pub proof messages tLast eLast) :
    ∃ b : List ((Option Coord → CodeSlot E) × Key20.Wire E),
      b.length = 81 ∧ (∀ x ∈ b, x.2.data = note20 pub proof messages tLast eLast ∧ x.2.nonce = 0) ∧
      (∀ i (inside : i < b.length), b[i].1 none = work (203 + i)) ∧
      ∀ point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total, point.wire 203 = wire₀ →
        (point.rest.drop 203).take 81 = b →
        Tier.Shows (third) (challengeTrace20 pub proof messages 3 Key20.level₃) (path20 pub proof messages E 3) 0 point := by
  obtain ⟨started, length, -, -, -⟩ := third20_numbers o
  have placed : o.block₃ = block₃ := by subst tower; rfl
  obtain ⟨s, decodes⟩ := slots20₃ (E := E) layout accepted terminalSetup decoded
  obtain ⟨b, sized, fields, worked, shows⟩ := Tier.exists_block20 (third) (by subst tower; rfl)
    (note20 pub proof messages tLast eLast) work
    (fun wire data => runs20_third _ pub proof messages tLast eLast layout accepted terminalSetup
      decoded expanded tower wire data)
    (fun wire => (checks20_third _ slots wire _).1) (fun wire => (checks20_third _ slots wire _).2)
    (fits_path20 layout accepted terminalSetup decoded 3).1
    (Extract.foldAnswer o.block₃ s (work (third).start) fun _ => work 0) rfl
    (fun b => Extract.foldAnswer_decode _ (fun _ => rfl) (placed ▸ injective₃) decodes _ _ b) wire₀ data₀
  rw [length] at sized
  rw [started] at worked
  rw [started, length] at shows
  exact ⟨b, sized, fields, worked, shows⟩

/-- The block of the stages of the level 4 of `tower20` for an accepted proof of `layout20`:
94 stages from the stage 284. -/
theorem block20_fourth {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : verifyProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (expanded : (pub.setupFor (fold20 4).setupCount).isSome)
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    (injective₄ : Function.Injective block₄) (slots : 22 ≤ Fintype.card Coord) (work : ℕ → CodeSlot E)
    (wire₀ : Key20.Wire E) (data₀ : wire₀.data = note20 pub proof messages tLast eLast) :
    ∃ b : List ((Option Coord → CodeSlot E) × Key20.Wire E),
      b.length = 94 ∧ (∀ x ∈ b, x.2.data = note20 pub proof messages tLast eLast ∧ x.2.nonce = 0) ∧
      (∀ i (inside : i < b.length), b[i].1 none = work (284 + i)) ∧
      ∀ point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total, point.wire 284 = wire₀ →
        (point.rest.drop 284).take 94 = b →
        Tier.Shows (fourth) (challengeTrace20 pub proof messages 4 Key20.level₄) (path20 pub proof messages E 4) 0 point := by
  obtain ⟨started, length, -, -, -⟩ := fourth20_numbers o
  have placed : o.block₄ = block₄ := by subst tower; rfl
  obtain ⟨s, decodes⟩ := slots20₄ (E := E) layout accepted terminalSetup decoded
  obtain ⟨b, sized, fields, worked, shows⟩ := Tier.exists_block20 (fourth) (by subst tower; rfl)
    (note20 pub proof messages tLast eLast) work
    (fun wire data => runs20_fourth _ pub proof messages tLast eLast layout accepted terminalSetup
      decoded expanded tower wire data)
    (fun wire => (checks20_fourth _ slots wire _).1) (fun wire => (checks20_fourth _ slots wire _).2)
    (fits_path20 layout accepted terminalSetup decoded 4).1
    (Extract.foldAnswer o.block₄ s (work (fourth).start) fun _ => work 0) rfl
    (fun b => Extract.foldAnswer_decode _ (fun _ => rfl) (placed ▸ injective₄) decodes _ _ b) wire₀ data₀
  rw [length] at sized
  rw [started] at worked
  rw [started, length] at shows
  exact ⟨b, sized, fields, worked, shows⟩

/-- The block of the stages of the level 5 of `tower20` for an accepted proof of `layout20`:
94 stages from the stage 378. -/
theorem block20_fifth {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : verifyProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (expanded : (pub.setupFor (fold20 5).setupCount).isSome)
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    (injective₅ : Function.Injective block₅) (slots : 22 ≤ Fintype.card Coord) (work : ℕ → CodeSlot E)
    (wire₀ : Key20.Wire E) (data₀ : wire₀.data = note20 pub proof messages tLast eLast) :
    ∃ b : List ((Option Coord → CodeSlot E) × Key20.Wire E),
      b.length = 94 ∧ (∀ x ∈ b, x.2.data = note20 pub proof messages tLast eLast ∧ x.2.nonce = 0) ∧
      (∀ i (inside : i < b.length), b[i].1 none = work (378 + i)) ∧
      ∀ point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total, point.wire 378 = wire₀ →
        (point.rest.drop 378).take 94 = b →
        Tier.Shows (fifth) (challengeTrace20 pub proof messages 5 Key20.level₅) (path20 pub proof messages E 5) 0 point := by
  obtain ⟨started, length, -, -, -⟩ := fifth20_numbers o
  have placed : o.block₅ = block₅ := by subst tower; rfl
  obtain ⟨s, decodes⟩ := slots20₅ (E := E) layout accepted terminalSetup decoded
  obtain ⟨b, sized, fields, worked, shows⟩ := Tier.exists_block20 (fifth) (by subst tower; rfl)
    (note20 pub proof messages tLast eLast) work
    (fun wire data => runs20_fifth _ pub proof messages tLast eLast layout accepted terminalSetup
      decoded expanded tower wire data)
    (fun wire => (checks20_fifth _ slots wire _).1) (fun wire => (checks20_fifth _ slots wire _).2)
    (fits_path20 layout accepted terminalSetup decoded 5).1
    (Extract.foldAnswer o.block₅ s (work (fifth).start) fun _ => work 0) rfl
    (fun b => Extract.foldAnswer_decode _ (fun _ => rfl) (placed ▸ injective₅) decodes _ _ b) wire₀ data₀
  rw [length] at sized
  rw [started] at worked
  rw [started, length] at shows
  exact ⟨b, sized, fields, worked, shows⟩

end NightstreamFPrime.Spec.Compression.Akita.Replay
