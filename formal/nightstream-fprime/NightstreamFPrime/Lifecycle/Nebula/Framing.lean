import NightstreamFPrime.Lifecycle.Transcript
import NightstreamFPrime.Spec.Nebula

/-! Owns the concrete Poseidon2 framing of the Nebula memory phase over the
Stage 1 v1.1 transcript: the record-chain digests of spec §9.1–§9.2, the plan
digest of §4.3, the `η` transcript of §9.3, and the concrete verifier context.
It also owns obligation Ob3 for the record chains: canonical chain inputs
absorb different padded chunk sequences, so a collision among the chain inputs
of an accepted run is a Poseidon2 transcript collision. -/

namespace NightstreamFPrime.Lifecycle.Nebula

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula

/-- A digest: four field words (spec §3). -/
abbrev Digest := Fin 4 → F

/-- The words of a digest, in lane order. -/
def digestWords (d : Digest) : List F := List.ofFn d

/-- ASCII text as field words, one per byte (spec §9.1). -/
def textWords (text : String) : List F := text.toList.map fun c => natWord c.toNat

/-- The header tags of spec §9.2. -/
def headerTag : Lane → String
  | .ops => "Nightstream/Nebula/v3/header-ops"
  | .mem => "Nightstream/Nebula/v3/header-mem"

/-- The chain tags of spec §9.2. -/
def chainTag : Lane → String
  | .ops => "Nightstream/Nebula/v3/chain-ops"
  | .mem => "Nightstream/Nebula/v3/chain-mem"

/-- The absorbed blocks of `Digest(tag, block_1, …, block_n)` (spec §9.1). -/
def blocks : HashInput Digest → List (List F)
  | .header lane pd => [textWords (headerTag lane), digestWords pd]
  | .chain lane j D P => [textWords (chainTag lane), natWord j :: digestWords D, P.map natWord]

/-- The transcript state after `reset_v1_1` and one `absorb_block_v1_1` per
block. -/
def absorbed (bs : List (List F)) : Poseidon2.State :=
  Transcript.absorbBlocks Transcript.initialState bs

/-- `squeeze_digest_v1_1`: the first four lanes. -/
def squeezeDigest (s : Poseidon2.State) : Digest := fun i => s.getD i 0

/-- The spec §9.1 digest of a record-chain input. -/
def hash (x : HashInput Digest) : Digest := squeezeDigest (absorbed (blocks x))

/-- Spec §4.3 `plan_digest`. -/
def planDigest (p : Plan) : Digest :=
  squeezeDigest (absorbed [textWords "Nightstream/Nebula/v3/plan",
    [p.r, p.μ, p.wTs, p.bOps, p.bScan, p.n, p.sMax].map natWord,
    (List.range p.romSize).map fun a => natWord (p.rom a),
    (List.range p.ramSize).map fun a => natWord (p.ram a)])

/-- Spec §9.3: the two memory challenges of a segment. -/
def etaChallenges (inp : EtaInput Digest) : K × K :=
  let s := absorbed [textWords "Nightstream/Nebula/v3/eta", digestWords inp.planDigest,
    [natWord inp.ts],
    digestWords inp.opsRoot ++ digestWords inp.memRoot ++ digestWords inp.finalRoot]
  let (η1, s) := Transcript.squeezeK s
  let (η2, _) := Transcript.squeezeK s
  (η1, η2)

/-- The verifier context of a plan: Poseidon2 chains, the §9.3 transcript, and
the plan digest that the verifier computes. -/
def context (p : Plan) : Context K Digest := ⟨p, hash, fun _ => etaChallenges, planDigest p⟩

/-! ### Absorbed chunks -/

/-- One rate chunk padded with zeros: the words that `Poseidon2.absorbBlock`
adds to the state. -/
def pad (c : List F) : Fin Poseidon2.rate → F := fun i => c.getD i 0

/-- The padded rate chunks that `Transcript.absorb` adds for one word list. -/
def chunks (xs : List F) : List (Fin Poseidon2.rate → F) :=
  (List.range ((xs.length + Poseidon2.rate - 1) / Poseidon2.rate)).map fun c =>
    pad ((xs.drop (c * Poseidon2.rate)).take Poseidon2.rate)

/-- The padded chunks of self-delimiting blocks, in absorb order. -/
def blockChunks (bs : List (List F)) : List (Fin Poseidon2.rate → F) :=
  bs.flatMap fun b => chunks (block b)

theorem blockChunks_cons (b : List F) (bs : List (List F)) :
    blockChunks (b :: bs) = chunks (block b) ++ blockChunks bs :=
  List.flatMap_cons

/-- A Poseidon2 transcript collision: two different padded chunk sequences
whose absorption gives the same digest. -/
def TranscriptCollision (bs bs' : List (List F)) : Prop :=
  blockChunks bs ≠ blockChunks bs' ∧ squeezeDigest (absorbed bs) = squeezeDigest (absorbed bs')

private theorem absorbBlock_pad (s : Poseidon2.State) {c : List F}
    (short : c.length ≤ Poseidon2.rate) :
    Poseidon2.absorbBlock s c = Poseidon2.absorbBlock s (List.ofFn (pad c)) := by
  unfold Poseidon2.absorbBlock
  congr 1
  apply List.map_congr_left
  intro i _
  congr 1
  by_cases low : i < Poseidon2.rate
  · simp [pad, List.getD_eq_getElem?_getD, low]
  · rw [List.getD_eq_default _ _ (by omega), List.getD_eq_default _ _ (by simp; omega)]

/-- One `absorb` adds exactly the padded chunks of its words. -/
private theorem absorb_eq (s : Poseidon2.State) (xs : List F) :
    Transcript.absorb s xs =
      (chunks xs).foldl (fun s c => Poseidon2.absorbBlock s (List.ofFn c)) s := by
  simp only [Transcript.absorb, chunks, List.foldl_map]
  exact List.foldl_ext _ _ _ fun s c _ => absorbBlock_pad s (List.length_take_le _ _)

/-- The transcript state depends only on the padded chunk sequence. -/
theorem absorbBlocks_eq (s : Poseidon2.State) (bs : List (List F)) :
    Transcript.absorbBlocks s bs =
      (blockChunks bs).foldl (fun s c => Poseidon2.absorbBlock s (List.ofFn c)) s := by
  induction bs generalizing s with
  | nil => rfl
  | cons b bs ih =>
    show Transcript.absorbBlocks (Transcript.absorb s (block b)) bs = _
    rw [ih, absorb_eq, blockChunks_cons, List.foldl_append]

private theorem natWord_injective {a b : ℕ} (ha : a < goldilocksModulus)
    (hb : b < goldilocksModulus) (same : natWord a = natWord b) : a = b := by
  have := congrArg Fin.val same
  simp only [natWord, Poseidon2.ofNat, Nat.mod_eq_of_lt ha, Nat.mod_eq_of_lt hb] at this
  exact this

private theorem chunk_word (xs : List F) {c : ℕ} (i : Fin Poseidon2.rate) :
    pad ((xs.drop (c * Poseidon2.rate)).take Poseidon2.rate) i =
      xs.getD (c * Poseidon2.rate + i) 0 := by
  simp [pad, List.getD_eq_getElem?_getD, i.isLt]

private theorem chunks_length (xs : List F) :
    (chunks xs).length = (xs.length + Poseidon2.rate - 1) / Poseidon2.rate := by
  simp [chunks]

/-- Equal-length word lists with the same padded chunks are equal. -/
private theorem chunks_injective {xs ys : List F} (length : xs.length = ys.length)
    (same : chunks xs = chunks ys) : xs = ys := by
  apply List.ext_getElem length
  intro t hx hy
  have rate : Poseidon2.rate = 12 := by unfold Poseidon2.rate; rfl
  have hc : t / Poseidon2.rate < (xs.length + Poseidon2.rate - 1) / Poseidon2.rate := by
    rw [rate]; omega
  have hc' : t / Poseidon2.rate < (ys.length + Poseidon2.rate - 1) / Poseidon2.rate := by
    rw [← length]; exact hc
  have entry := congrArg (fun l => l[t / Poseidon2.rate]?) same
  simp only [chunks, List.getElem?_map, List.getElem?_range hc, List.getElem?_range hc',
    Option.map_some, Option.some.injEq] at entry
  have word := congrFun entry ⟨t % Poseidon2.rate, Nat.mod_lt _ (by decide)⟩
  rw [chunk_word, chunk_word, Nat.div_add_mod'] at word
  rwa [List.getD_eq_getElem _ _ hx, List.getD_eq_getElem _ _ hy] at word

private theorem chunks_block_ne_nil (b : List F) : chunks (block b) ≠ [] := by
  simp [chunks, block, Poseidon2.rate]

private theorem chunks_block_head (b : List F) (h : chunks (block b) ≠ []) :
    (chunks (block b)).head h ⟨0, by decide⟩ = natWord b.length := by
  simp [chunks, block, pad, List.head_map, List.getD_eq_getElem?_getD, Poseidon2.rate]

/-- Blocks whose lengths are canonical have different padded chunk
sequences when they differ. -/
theorem blockChunks_injective : ∀ {bs bs' : List (List F)},
    (∀ b ∈ bs, b.length < goldilocksModulus) → (∀ b ∈ bs', b.length < goldilocksModulus) →
      blockChunks bs = blockChunks bs' → bs = bs'
  | [], [], _, _, _ => rfl
  | [], b :: _, _, _, same => by
    simp only [blockChunks, List.flatMap_nil, List.flatMap_cons] at same
    exact absurd (List.append_eq_nil_iff.1 same.symm).1 (chunks_block_ne_nil b)
  | b :: _, [], _, _, same => by
    simp only [blockChunks, List.flatMap_nil, List.flatMap_cons] at same
    exact absurd (List.append_eq_nil_iff.1 same).1 (chunks_block_ne_nil b)
  | b :: bs, b' :: bs', small, small', same => by
    simp only [blockChunks, List.flatMap_cons] at same
    have nonempty := chunks_block_ne_nil b
    have nonempty' := chunks_block_ne_nil b'
    have heads : (chunks (block b)).head nonempty = (chunks (block b')).head nonempty' := by
      have := congrArg List.head? same
      rw [List.head?_append_of_ne_nil _ nonempty, List.head?_append_of_ne_nil _ nonempty',
        List.head?_eq_some_head nonempty, List.head?_eq_some_head nonempty'] at this
      exact Option.some.inj this
    have lengths : b.length = b'.length := natWord_injective (small b (by simp))
      (small' b' (by simp)) (by
        rw [← chunks_block_head b nonempty, ← chunks_block_head b' nonempty', heads])
    have blockLength : (block b).length = (block b').length := by simp [block, lengths]
    obtain ⟨first, rest⟩ := List.append_inj same (by rw [chunks_length, chunks_length, blockLength])
    have equal := chunks_injective blockLength first
    rw [block, block, List.cons.injEq] at equal
    rw [equal.2, blockChunks_injective (fun x hx => small x (by simp [hx]))
      (fun x hx => small' x (by simp [hx])) rest]

/-! ### Ob3 for the record chains -/

private theorem headerTag_injective {l l' : Lane}
    (same : textWords (headerTag l) = textWords (headerTag l')) : l = l' := by
  cases l <;> cases l' <;> first | rfl | exact absurd same (by decide)

private theorem chainTag_injective {l l' : Lane}
    (same : textWords (chainTag l) = textWords (chainTag l')) : l = l' := by
  cases l <;> cases l' <;> first | rfl | exact absurd same (by decide)

private theorem packed_injective {P P' : List ℕ} (small : ∀ x ∈ P, x < 2 ^ 63)
    (small' : ∀ x ∈ P', x < 2 ^ 63) (same : P.map natWord = P'.map natWord) : P = P' := by
  have modulus : (2 : ℕ) ^ 63 < goldilocksModulus := by decide
  apply List.ext_getElem (by simpa using congrArg List.length same)
  intro i h h'
  have word := congrArg (fun l => l[i]?) same
  simp only [List.getElem?_map, List.getElem?_eq_getElem h, List.getElem?_eq_getElem h',
    Option.map_some, Option.some.injEq] at word
  exact natWord_injective ((small _ (List.getElem_mem h)).trans modulus)
    ((small' _ (List.getElem_mem h')).trans modulus) word

/-- Canonical inputs have canonical block lengths. -/
private theorem blocks_small {p : Plan} (valid : p.Valid) {x : HashInput Digest}
    (canonical : x.Canonical p.laneLength p.n) :
    ∀ b ∈ blocks x, b.length < goldilocksModulus := by
  cases x with
  | header lane pd =>
    intro b hb
    simp only [blocks, List.mem_cons, List.mem_nil_iff, or_false] at hb
    rcases hb with rfl | rfl
    · cases lane <;> decide
    · simp [digestWords, goldilocksModulus]
  | chain lane j D P =>
    obtain ⟨-, length, -⟩ := canonical
    intro b hb
    simp only [blocks, List.mem_cons, List.mem_nil_iff, or_false] at hb
    rcases hb with rfl | rfl | rfl
    · cases lane <;> decide
    · simp [digestWords, goldilocksModulus]
    · rw [List.length_map, length]
      exact Plan.laneLength_lt valid lane

/-- The framing separates canonical inputs (Ob3). -/
theorem blocks_injective {p : Plan} (valid : p.Valid) {x y : HashInput Digest}
    (canonicalX : x.Canonical p.laneLength p.n) (canonicalY : y.Canonical p.laneLength p.n)
    (same : blocks x = blocks y) : x = y := by
  have indices : p.n < goldilocksModulus := by
    have := valid.belowModulus
    have := valid.exactCover
    have := valid.positive.2.1
    unfold Plan.cells at *
    nlinarith
  cases x with
  | header lane pd =>
    cases y with
    | header lane' pd' =>
      simp only [blocks, List.cons.injEq, and_true] at same
      rw [headerTag_injective same.1, List.ofFn_injective same.2]
    | chain => simp [blocks] at same
  | chain lane j D P =>
    cases y with
    | header => simp [blocks] at same
    | chain lane' j' D' P' =>
      obtain ⟨hj, -, hP⟩ := canonicalX
      obtain ⟨hj', -, hP'⟩ := canonicalY
      simp only [blocks, List.cons.injEq, and_true] at same
      obtain ⟨tag, ⟨index, digest⟩, packed⟩ := same
      rw [chainTag_injective tag, natWord_injective (hj.trans indices) (hj'.trans indices) index,
        List.ofFn_injective digest, packed_injective hP hP' packed]

open scoped NightstreamFPrime.Spec.GoldilocksExtensionRing

/-- Ob3 for two input lists: if every input is canonical, a collision of the
Poseidon2 chains between the lists is a Poseidon2 transcript collision. -/
theorem collision_transcript {ctx : Context K Digest} (concrete : ctx.hash = hash)
    (valid : ctx.plan.Valid) {xs ys : List (HashInput Digest)}
    (canonicalX : ∀ x ∈ xs, x.Canonical ctx.plan.laneLength ctx.plan.n)
    (canonicalY : ∀ y ∈ ys, y.Canonical ctx.plan.laneLength ctx.plan.n)
    (collision : CollisionIn ctx.hash xs ys) :
    ∃ a ∈ xs, ∃ b ∈ ys, TranscriptCollision (blocks a) (blocks b) := by
  obtain ⟨a, ha, b, hb, differ, same⟩ := collision
  rw [concrete] at same
  refine ⟨a, ha, b, hb, fun chunksEqual => differ ?_, same⟩
  exact blocks_injective valid (canonicalX a ha) (canonicalY b hb)
    (blockChunks_injective (blocks_small valid (canonicalX a ha))
      (blocks_small valid (canonicalY b hb)) chunksEqual)

variable {σ : Type} {ctx : Context K Digest} {app : Application ctx.plan σ}
  {stmt : Statement σ Digest} {run : List (Invocation σ Digest)}

/-- Ob3 for the record chains: when the context uses the Poseidon2 chains, a
collision among the hash inputs of an accepted run is a Poseidon2 transcript
collision. -/
theorem runCollision_transcript (concrete : ctx.hash = hash) (valid : ctx.plan.Valid)
    (accepted : Accepts ctx app stmt run) (collision : RunCollision ctx run stmt.segments) :
    ∃ a ∈ runInputs ctx run stmt.segments, ∃ b ∈ runInputs ctx run stmt.segments,
      TranscriptCollision (blocks a) (blocks b) :=
  collision_transcript concrete valid (runInputs_canonical valid accepted)
    (runInputs_canonical valid accepted) collision

/-- Security note Lemma 6 with the Poseidon2 chains: an accepted run gives an
execution, or a Poseidon2 transcript collision among its own chain inputs, or a
segment whose challenges pass the product test with unbalanced multisets. -/
theorem poseidon2_soundness (concrete : ctx.hash = hash) (valid : ctx.plan.Valid)
    (accepted : Accepts ctx app stmt run) :
    Attests ctx app stmt run ∨
      (∃ a ∈ runInputs ctx run stmt.segments, ∃ b ∈ runInputs ctx run stmt.segments,
        TranscriptCollision (blocks a) (blocks b)) ∨
      ∃ k < stmt.segments, BadChallenge ctx (segmentView ctx run k) :=
  (Spec.Nebula.soundness valid accepted).imp_right
    (Or.imp_left (runCollision_transcript concrete valid accepted))

end NightstreamFPrime.Lifecycle.Nebula
