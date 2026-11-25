import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinTerminal

/-! The slots of the fold challenges of the code for theorem A of the join of `key20.aks` (Akita
revision `d50f700`, `crates/akita-challenges/src/sampler/stream.rs`, `VerifierFoldDraw`; ePrint
2026/1983, revision of 2026-11-23, Appendix E.1).

- `drawn20`: the drawn fold challenges of a group of a fold of an accepted proof of `layout20` are
  the challenges `cs` of the sampler of the code on a message, moved by `ambient`.
- `exists_open_drawn`, `exists_kept64_drawn`, `exists_kept128_drawn`: each challenge of the sampler
  is the challenge of a slot for the decoder of its family: without a filter the decoded
  candidate of the message (`sample_unfiltered`, `open_iff`), with a filter a member of the family
  that passes the filter (`sample_mem`, `kept64_surjective`, `kept128_surjective`).
- `carrier_drawn`, `trace_drawn`, `terminal_drawn`: the fold challenges of the specification from
  the drawn challenges are the challenges of the sampler.
- `slots20₀` to `slots20₅` and `slots20L`: for an accepted proof, each fold coordinate of each level
  and of the terminal has a slot that decodes to the fold challenge of the code. There is no
  hypothesis `Key20.Kept`: each drawn challenge is the decode of an accepted word. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita

open Replayer

section Sampler

variable {E : Type} [Nonempty E]

/-- Without a filter each challenge of the sampler is the challenge of a slot of `CodeSlot.open`
of the family: the decoded candidate of the message. -/
theorem exists_open_drawn {n r d p₁ p₂ : ℕ} [NeZero d] {bytes : ByteArray} {cs : List Sparse}
    (sampled : sample ⟨n, r, d, p₁, p₂, false⟩ bytes = some cs)
    (positive : 0 < (⟨n, r, d, p₁, p₂, false⟩ : FoldGroup).rankSpace)
    (fits : (⟨0, d, d, p₁, p₂, false⟩ : FoldGroup).candidateBits ≤ 131) {j : ℕ}
    (inside : j < cs.length) :
    ∃ s : CodeSlot E, CodeSlot.open ⟨0, d, d, p₁, p₂, false⟩ fits s = some (Sparse.ring d cs[j]) := by
  have found := sample_unfiltered rfl positive sampled j
  rw [List.getElem?_eq_getElem inside] at found
  split_ifs at found with small
  refine (open_iff fits).mpr ⟨⟨readBits bytes (j * (⟨n, r, d, p₁, p₂, false⟩ : FoldGroup).candidateBits)
    (⟨n, r, d, p₁, p₂, false⟩ : FoldGroup).candidateBits, readBits_lt _ _ _⟩, ?_⟩
  show ((candidate ⟨n, r, d, p₁, p₂, false⟩ Nat.choose _).filter _).map (Sparse.ring d) = _
  rw [← found]
  simp

/-- With the filter `(64, 31, 11)` each challenge of the sampler is a challenge of
`CodeSlot.kept64`. -/
theorem exists_kept64_drawn {n : ℕ} {bytes : ByteArray} {cs : List Sparse}
    (sampled : sample ⟨n, 64, 64, 31, 11, true⟩ bytes = some cs) {c : Sparse} (member : c ∈ cs) :
    ∃ s : CodeSlot E, CodeSlot.kept64 s = some (Sparse.ring 64 c) :=
  kept64_surjective ((sample_mem sampled).2 c member).1 (((sample_mem sampled).2 c member).2 rfl)

/-- With the filter `(128, 31, 0)` each challenge of the sampler is a challenge of
`CodeSlot.kept128`. -/
theorem exists_kept128_drawn {n : ℕ} {bytes : ByteArray} {cs : List Sparse}
    (sampled : sample ⟨n, 128, 128, 31, 0, true⟩ bytes = some cs) {c : Sparse} (member : c ∈ cs) :
    ∃ s : CodeSlot E, CodeSlot.kept128 s = some (Sparse.ring 128 c) :=
  kept128_surjective ((sample_mem sampled).2 c member).1 (((sample_mem sampled).2 c member).2 rfl)

end Sampler

section Forms

/-- A drawn challenge of a group with stride `k` in the subring of the carrier, with its positions
divided by `k`, is the challenge of the sampler (`drawnCarrier`, `challenge20₁`). -/
theorem carrier_drawn {G : FoldGroup} {dCar k : ℕ} [NeZero dCar] (stride : G.ring / G.degree = k)
    (positive : 0 < k) (cs : List Sparse) {i : ℕ} (inside : i < cs.length) :
    Poly.toRq dCar (((cs.map (ambient G)).map fun c =>
      Poly.ofSparse dCar (c.map fun x => (x.1 / k, x.2))).getD i (Poly.zero dCar)) =
      Sparse.ring dCar cs[i] := by
  rw [List.getD_eq_getElem _ _ (by simpa using inside), List.getElem_map, List.getElem_map]
  have back : (ambient G cs[i]).map (fun x => (x.1 / k, x.2)) = cs[i] := by
    unfold ambient
    rw [List.map_map]
    conv_rhs => rw [← List.map_id cs[i]]
    refine List.map_congr_left fun x _ => ?_
    simp [stride, Nat.mul_div_cancel _ positive]
  rw [back, Poly.toRq_ofSparse_ring]

/-- A drawn challenge of a group of stride 1 is the challenge of the sampler (`challengeTrace20`). -/
theorem trace_drawn {G : FoldGroup} {d : ℕ} [NeZero d] (same : G.ring = G.degree)
    (positive : 0 < G.degree) (cs : List Sparse) {i : ℕ} (inside : i < cs.length) :
    Poly.toRq d (((cs.map (ambient G)).map (Poly.ofSparse d)).getD i (Poly.zero d)) =
      Sparse.ring d cs[i] := by
  rw [List.getD_eq_getElem _ _ (by simpa using inside), List.getElem_map, List.getElem_map,
    ambient_one positive same, Poly.toRq_ofSparse_ring]

/-- A drawn challenge of the terminal group is the challenge of the sampler (`challenge20L`). -/
theorem terminal_drawn {G : FoldGroup} {d : ℕ} [NeZero d] (same : G.ring = G.degree)
    (positive : 0 < G.degree) (cs : List Sparse) {i : ℕ} (inside : i < cs.length) :
    Poly.toRq d (Poly.ofSparse d ((cs.map (ambient G)).getD i [])) = Sparse.ring d cs[i] := by
  rw [List.getD_eq_getElem _ _ (by simpa using inside), List.getElem_map, ambient_one positive same,
    Poly.toRq_ofSparse_ring]

end Forms

section Challenges

variable (pub : Public) (proof : ByteArray) (messages : Array Message) {G : FoldGroup}
  {cs : List Sparse}

/-- The fold challenge of the root fold at `⟨g, i⟩` from the challenges `cs` of the group `g`. -/
theorem challenge20₀_eq (g : Fin 3) (i : Fin (Key20.root.shape g).blocks)
    (drawn : drawnIn (fused20 pub proof messages 0) 0 g = cs.map (ambient G))
    (stride : G.ring / G.degree = (Key20.root.shape g).innerDegree / Key20.root.carrierDegree g)
    (positive : 0 < (Key20.root.shape g).innerDegree / Key20.root.carrierDegree g)
    (inside : (i : ℕ) < cs.length) :
    challenge20₀ pub proof messages ⟨g, i⟩ = Sparse.ring (Key20.root.carrierDegree g) cs[i] := by
  simp only [challenge20₀, drawnCarrier, Equiv.apply_symm_apply]
  rw [drawn]
  exact carrier_drawn stride positive cs inside

/-- The fold challenge of the level 1 at `i` from the challenges `cs` of its group. -/
theorem challenge20₁_eq (i : Fin Key20.level₁.shape.blocks)
    (drawn : drawnIn (fused20 pub proof messages 1) 1 0 = cs.map (ambient G))
    (stride : G.ring / G.degree =
      Key20.level₁.shape.innerDegree / Key20.level₁.opening.carrierDegree)
    (positive : 0 < Key20.level₁.shape.innerDegree / Key20.level₁.opening.carrierDegree)
    (inside : (i : ℕ) < cs.length) :
    challenge20₁ pub proof messages i = Sparse.ring Key20.level₁.opening.carrierDegree cs[i] := by
  unfold challenge20₁
  rw [drawn]
  exact carrier_drawn stride positive cs inside

/-- The fold challenge of the trace level `ℓ` at `i` from the challenges `cs` of its group. -/
theorem challengeTrace20_eq (ℓ : ℕ) (L : Key20.Level) (i : Fin L.shape.blocks)
    (drawn : drawnIn (fused20 pub proof messages ℓ) ℓ 0 = cs.map (ambient G))
    (same : G.ring = G.degree) (positive : 0 < G.degree) (inside : (i : ℕ) < cs.length) :
    challengeTrace20 pub proof messages ℓ L i = Sparse.ring L.shape.innerDegree cs[i] := by
  unfold challengeTrace20
  rw [drawn]
  exact trace_drawn same positive cs inside

/-- The fold challenge of the terminal at `i` from the challenges `cs` of its group. -/
theorem challenge20L_eq (i : Fin Key20.terminal.blocks)
    (drawn : drawnIn (terminalState20 pub proof messages) 6 0 = cs.map (ambient G))
    (same : G.ring = G.degree) (positive : 0 < G.degree) (inside : (i : ℕ) < cs.length) :
    challenge20L pub proof messages i = Sparse.ring Key20.terminal.innerDegree cs[i] := by
  show Poly.toRq _ (Poly.ofSparse _ ((drawnIn (terminalState20 pub proof messages) 6 0).getD i [])) = _
  rw [drawn]
  exact terminal_drawn same positive cs inside

end Challenges

variable {pub : Public} {proof : ByteArray} {messages : Array Message} {E : Type} [Nonempty E]
  {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
  (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
  (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) {rest : ℕ}
  (decoded : decode pub.ops proof = .ok (messages, rest))
include layout accepted terminalSetup decoded

omit [Nonempty E] in
/-- The drawn fold challenges of the group `g` of the fold of the level `ℓ` of an accepted proof of
`layout20` are the challenges `cs` of the sampler on a message, moved by `ambient`. -/
theorem drawn20 (ℓ : Fin 6) {g : ℕ} {G : FoldGroup} (listed : (fold20 ℓ).groups[g]? = some G) :
    ∃ bytes cs, sample G bytes = some cs ∧
      drawnIn (fused20 pub proof messages ℓ) ℓ g = cs.map (ambient G) := by
  obtain ⟨scripted, ends, levels⟩ := verifyProof_script20 layout accepted
  have ran := (replay_step_ok pub proof messages
    (replay_of_decoded pub proof scripted ends terminalSetup accepted decoded).2 (levels ℓ).at_).1
  obtain ⟨bytes, drawn⟩ := drawnIn_sampled pub proof messages ℓ g G _ _ (by simp [channelStart])
    (squeezeOf_of_shape (levels ℓ).drawn listed) ran
  unfold foldChallenges at drawn
  obtain ⟨cs, sampled, mapped⟩ := Option.map_eq_some_iff.mp drawn
  exact ⟨bytes, cs, sampled, mapped.symm⟩

/-- Each fold coordinate of the root fold has a slot that decodes to the fold challenge of the
code. -/
theorem slots20₀ : ∃ s : Key20.root.Block → CodeSlot E,
    ∀ b, Key20.rootDecode E b.1 (s b) = some (challenge20₀ pub proof messages b) := by
  have group₀ : ∀ i : Fin (Key20.root.shape 0).blocks, ∃ slot : CodeSlot E,
      Key20.rootDecode E 0 slot = some (challenge20₀ pub proof messages ⟨0, i⟩) := by
    obtain ⟨bytes, cs, sampled, drawn⟩ := drawn20 layout accepted terminalSetup decoded 0 (g := 0)
      (G := ⟨32, 1024, 128, 31, 0, false⟩) (by decide +kernel)
    intro i
    have inside : (i : ℕ) < cs.length := by rw [(sample_mem sampled).1]; exact i.isLt
    rw [challenge20₀_eq pub proof messages 0 i drawn (by decide) (by decide) inside]
    exact exists_open_drawn sampled (by decide) (Key20.rootGroup_fits 0) inside
  have group₁ : ∀ i : Fin (Key20.root.shape 1).blocks, ∃ slot : CodeSlot E,
      Key20.rootDecode E 1 slot = some (challenge20₀ pub proof messages ⟨1, i⟩) := by
    obtain ⟨bytes, cs, sampled, drawn⟩ := drawn20 layout accepted terminalSetup decoded 0 (g := 1)
      (G := ⟨16, 512, 64, 31, 10, false⟩) (by decide +kernel)
    intro i
    have inside : (i : ℕ) < cs.length := by rw [(sample_mem sampled).1]; exact i.isLt
    rw [challenge20₀_eq pub proof messages 1 i drawn (by decide) (by decide) inside]
    exact exists_open_drawn sampled (by decide) (Key20.rootGroup_fits 1) inside
  have group₂ : ∀ i : Fin (Key20.root.shape 2).blocks, ∃ slot : CodeSlot E,
      Key20.rootDecode E 2 slot = some (challenge20₀ pub proof messages ⟨2, i⟩) := by
    obtain ⟨bytes, cs, sampled, drawn⟩ := drawn20 layout accepted terminalSetup decoded 0 (g := 2)
      (G := ⟨8, 1024, 64, 31, 10, false⟩) (by decide +kernel)
    intro i
    have inside : (i : ℕ) < cs.length := by rw [(sample_mem sampled).1]; exact i.isLt
    rw [challenge20₀_eq pub proof messages 2 i drawn (by decide) (by decide) inside]
    exact exists_open_drawn sampled (by decide) (Key20.rootGroup_fits 2) inside
  have each : ∀ b : Key20.root.Block, ∃ slot : CodeSlot E,
      Key20.rootDecode E b.1 slot = some (challenge20₀ pub proof messages b) := by
    rintro ⟨g, i⟩
    fin_cases g
    exacts [group₀ i, group₁ i, group₂ i]
  choose s decodes using each
  exact ⟨s, decodes⟩

/-- Each fold coordinate of the level 1 has a slot that decodes to the fold challenge of the
code. -/
theorem slots20₁ : ∃ s : Fin Key20.level₁.shape.blocks → CodeSlot E,
    ∀ i, CodeSlot.open Key20.open64 Key20.open64_fits (s i) =
      some (challenge20₁ pub proof messages i) := by
  obtain ⟨bytes, cs, sampled, drawn⟩ := drawn20 layout accepted terminalSetup decoded 1 (g := 0)
    (G := ⟨28, 1024, 64, 31, 10, false⟩) (by decide +kernel)
  have each : ∀ i : Fin Key20.level₁.shape.blocks, ∃ slot : CodeSlot E,
      CodeSlot.open Key20.open64 Key20.open64_fits slot = some (challenge20₁ pub proof messages i) := by
    intro i
    have inside : (i : ℕ) < cs.length := by rw [(sample_mem sampled).1]; exact i.isLt
    rw [challenge20₁_eq pub proof messages i drawn (by decide) (by decide) inside]
    exact exists_open_drawn sampled (by decide) Key20.open64_fits inside
  choose s decodes using each
  exact ⟨s, decodes⟩

/-- Each fold coordinate of the level 2 has a slot that decodes to the fold challenge of the
code. -/
theorem slots20₂ : ∃ s : Fin Key20.level₂.shape.blocks → CodeSlot E,
    ∀ i, CodeSlot.open Key20.open128 Key20.open128_fits (s i) =
      some (challengeTrace20 pub proof messages 2 Key20.level₂ i) := by
  obtain ⟨bytes, cs, sampled, drawn⟩ := drawn20 layout accepted terminalSetup decoded 2 (g := 0)
    (G := ⟨21, 128, 128, 31, 0, false⟩) (by decide +kernel)
  have each : ∀ i : Fin Key20.level₂.shape.blocks, ∃ slot : CodeSlot E,
      CodeSlot.open Key20.open128 Key20.open128_fits slot =
        some (challengeTrace20 pub proof messages 2 Key20.level₂ i) := by
    intro i
    have inside : (i : ℕ) < cs.length := by rw [(sample_mem sampled).1]; exact i.isLt
    rw [challengeTrace20_eq pub proof messages 2 Key20.level₂ i drawn rfl (by decide) inside]
    exact exists_open_drawn sampled (by decide) Key20.open128_fits inside
  choose s decodes using each
  exact ⟨s, decodes⟩

/-- Each fold coordinate of the level 3 has a slot that decodes to the fold challenge of the
code. -/
theorem slots20₃ : ∃ s : Fin Key20.level₃.shape.blocks → CodeSlot E,
    ∀ i, CodeSlot.kept128 (s i) = some (challengeTrace20 pub proof messages 3 Key20.level₃ i) := by
  obtain ⟨bytes, cs, sampled, drawn⟩ := drawn20 layout accepted terminalSetup decoded 3 (g := 0)
    (G := ⟨11, 128, 128, 31, 0, true⟩) (by decide +kernel)
  have each : ∀ i : Fin Key20.level₃.shape.blocks, ∃ slot : CodeSlot E,
      CodeSlot.kept128 slot = some (challengeTrace20 pub proof messages 3 Key20.level₃ i) := by
    intro i
    have inside : (i : ℕ) < cs.length := by rw [(sample_mem sampled).1]; exact i.isLt
    rw [challengeTrace20_eq pub proof messages 3 Key20.level₃ i drawn rfl (by decide) inside]
    exact exists_kept128_drawn sampled (List.getElem_mem inside)
  choose s decodes using each
  exact ⟨s, decodes⟩

/-- Each fold coordinate of the level 4 has a slot that decodes to the fold challenge of the
code. -/
theorem slots20₄ : ∃ s : Fin Key20.level₄.shape.blocks → CodeSlot E,
    ∀ i, CodeSlot.kept64 (s i) = some (challengeTrace20 pub proof messages 4 Key20.level₄ i) := by
  obtain ⟨bytes, cs, sampled, drawn⟩ := drawn20 layout accepted terminalSetup decoded 4 (g := 0)
    (G := ⟨7, 64, 64, 31, 11, true⟩) (by decide +kernel)
  have each : ∀ i : Fin Key20.level₄.shape.blocks, ∃ slot : CodeSlot E,
      CodeSlot.kept64 slot = some (challengeTrace20 pub proof messages 4 Key20.level₄ i) := by
    intro i
    have inside : (i : ℕ) < cs.length := by rw [(sample_mem sampled).1]; exact i.isLt
    rw [challengeTrace20_eq pub proof messages 4 Key20.level₄ i drawn rfl (by decide) inside]
    exact exists_kept64_drawn sampled (List.getElem_mem inside)
  choose s decodes using each
  exact ⟨s, decodes⟩

/-- Each fold coordinate of the level 5 has a slot that decodes to the fold challenge of the
code. -/
theorem slots20₅ : ∃ s : Fin Key20.level₅.shape.blocks → CodeSlot E,
    ∀ i, CodeSlot.kept64 (s i) = some (challengeTrace20 pub proof messages 5 Key20.level₅ i) := by
  obtain ⟨bytes, cs, sampled, drawn⟩ := drawn20 layout accepted terminalSetup decoded 5 (g := 0)
    (G := ⟨7, 64, 64, 31, 11, true⟩) (by decide +kernel)
  have each : ∀ i : Fin Key20.level₅.shape.blocks, ∃ slot : CodeSlot E,
      CodeSlot.kept64 slot = some (challengeTrace20 pub proof messages 5 Key20.level₅ i) := by
    intro i
    have inside : (i : ℕ) < cs.length := by rw [(sample_mem sampled).1]; exact i.isLt
    rw [challengeTrace20_eq pub proof messages 5 Key20.level₅ i drawn rfl (by decide) inside]
    exact exists_kept64_drawn sampled (List.getElem_mem inside)
  choose s decodes using each
  exact ⟨s, decodes⟩

/-- Each fold coordinate of the terminal has a slot that decodes to the fold challenge of the
code. -/
theorem slots20L : ∃ s : Fin Key20.terminal.blocks → CodeSlot E,
    ∀ i, CodeSlot.kept64 (s i) = some (challenge20L pub proof messages i) := by
  obtain ⟨bytes, drawn⟩ := verifyProof_drawn20 layout accepted terminalSetup decoded
  have group : (layout20 zMax rice).terminal.group = ⟨5, 64, 64, 31, 11, true⟩ := rfl
  rw [group] at drawn
  unfold foldChallenges at drawn
  obtain ⟨cs, sampled, mapped⟩ := Option.map_eq_some_iff.mp drawn
  have each : ∀ i : Fin Key20.terminal.blocks, ∃ slot : CodeSlot E,
      CodeSlot.kept64 slot = some (challenge20L pub proof messages i) := by
    intro i
    have inside : (i : ℕ) < cs.length := by rw [(sample_mem sampled).1]; exact i.isLt
    rw [challenge20L_eq pub proof messages i mapped.symm rfl (by decide) inside]
    exact exists_kept64_drawn sampled (List.getElem_mem inside)
  choose s decodes using each
  exact ⟨s, decodes⟩

end NightstreamFPrime.Spec.Compression.Akita.Replay
