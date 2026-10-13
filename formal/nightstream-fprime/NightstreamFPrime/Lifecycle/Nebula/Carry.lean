import NightstreamFPrime.Lifecycle.Nebula.Framing

/-! Owns the memory carry of spec §11.1 over `K` and four-word digests: its 39
words, their canonical decoding (Ob1), and the state digest that binds the
application words and the carry words into the four Stage 1 application-state
words. It does not own the memory rows or the carry actions. -/

namespace NightstreamFPrime.Lifecycle.Nebula

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula

/-- The memory carry over the trunk challenge field and four-word digests. -/
abbrev MemoryCarry := Carry K Digest

/-- The 39 carry words of spec §11.1, in field order: each counter as one
word, each `K` element as `(c0, c1)`, each digest as four words. -/
def carryVector (c : MemoryCarry) : Fin 39 → F :=
  ![natWord c.segIdx, natWord c.idx, natWord c.ts,
    c.eta.1.c0, c.eta.1.c1, c.eta.2.c0, c.eta.2.c1,
    c.products.read.c0, c.products.read.c1, c.products.write.c0, c.products.write.c1,
    c.products.initial.c0, c.products.initial.c1, c.products.final.c0, c.products.final.c1,
    c.proposed.1 0, c.proposed.1 1, c.proposed.1 2, c.proposed.1 3,
    c.proposed.2 0, c.proposed.2 1, c.proposed.2 2, c.proposed.2 3,
    c.seen.ops 0, c.seen.ops 1, c.seen.ops 2, c.seen.ops 3,
    c.seen.initial 0, c.seen.initial 1, c.seen.initial 2, c.seen.initial 3,
    c.seen.final 0, c.seen.final 1, c.seen.final 2, c.seen.final 3,
    c.memRoot 0, c.memRoot 1, c.memRoot 2, c.memRoot 3]

/-- The carry words as a list. -/
def carryWords (c : MemoryCarry) : List F := List.ofFn (carryVector c)

theorem carryWords_length (c : MemoryCarry) : (carryWords c).length = 39 := by
  simp [carryWords]

/-- Four consecutive carry words as a digest. -/
def readDigest (v : Fin 39 → F) (start : ℕ) (fits : start + 3 < 39) : Digest :=
  fun i => v ⟨start + i, by omega⟩

/-- Spec §11.1 decoding of the 39 carry words. Every counter is the canonical
value of its word. -/
def decodeCarry (v : Fin 39 → F) : MemoryCarry where
  segIdx := (v 0).val
  idx := (v 1).val
  ts := (v 2).val
  eta := (⟨v 3, v 4⟩, ⟨v 5, v 6⟩)
  products := ⟨⟨v 7, v 8⟩, ⟨v 9, v 10⟩, ⟨v 11, v 12⟩, ⟨v 13, v 14⟩⟩
  proposed := (readDigest v 15 (by decide), readDigest v 19 (by decide))
  seen := ⟨readDigest v 23 (by decide), readDigest v 27 (by decide),
    readDigest v 31 (by decide)⟩
  memRoot := readDigest v 35 (by decide)

/-- A carry whose counters are canonical field values. -/
def MemoryCarry.Canonical (c : MemoryCarry) : Prop :=
  c.segIdx < goldilocksModulus ∧ c.idx < goldilocksModulus ∧ c.ts < goldilocksModulus

/-- A natural value below `q` is its field word's value. -/
theorem natWord_val {n : ℕ} (small : n < goldilocksModulus) : (natWord n).val = n := by
  simp [natWord, Poseidon2.ofNat, Nat.mod_eq_of_lt small]

theorem natWord_of_val (x : F) : natWord x.val = x := by
  apply Fin.ext
  simp [natWord, Poseidon2.ofNat, Nat.mod_eq_of_lt x.isLt]

theorem natWord_zero : natWord 0 = 0 := rfl

theorem natWord_one : natWord 1 = 1 := rfl

private theorem digest_ext {d : Digest} (start : ℕ) (fits : start + 3 < 39)
    (v : Fin 39 → F) (at0 : v ⟨start, by omega⟩ = d 0) (at1 : v ⟨start + 1, by omega⟩ = d 1)
    (at2 : v ⟨start + 2, by omega⟩ = d 2) (at3 : v ⟨start + 3, by omega⟩ = d 3) :
    readDigest v start fits = d := by
  funext i
  fin_cases i
  · exact at0
  · exact at1
  · exact at2
  · exact at3

/-- Ob1: decoding inverts the encoding on canonical carries. -/
theorem decodeCarry_carryVector {c : MemoryCarry} (canonical : c.Canonical) :
    decodeCarry (carryVector c) = c := by
  obtain ⟨segIdx, idx, ts, ⟨η1, η2⟩, ⟨rd, wr, ini, fin⟩, ⟨pOps, pFs⟩, ⟨sOps, sIni, sFin⟩,
    root⟩ := c
  obtain ⟨hs, hi, ht⟩ := canonical
  simp only [decodeCarry]
  congr 1
  · exact natWord_val hs
  · exact natWord_val hi
  · exact natWord_val ht
  · congr 1 <;> exact digest_ext _ _ _ rfl rfl rfl rfl
  · congr 1 <;> exact digest_ext _ _ _ rfl rfl rfl rfl
  · exact digest_ext _ _ _ rfl rfl rfl rfl

/-- Ob1: every word vector is the encoding of its canonical decoding. -/
theorem carryVector_decodeCarry (v : Fin 39 → F) : carryVector (decodeCarry v) = v := by
  funext i
  fin_cases i <;> first | exact natWord_of_val _ | rfl

/-- Ob1: the encoding is injective on canonical carries. -/
theorem carryVector_injective {c c' : MemoryCarry} (canonical : c.Canonical)
    (canonical' : c'.Canonical) (same : carryVector c = carryVector c') : c = c' := by
  rw [← decodeCarry_carryVector canonical, ← decodeCarry_carryVector canonical', same]

/-- Every decoded carry is canonical. -/
theorem decodeCarry_canonical (v : Fin 39 → F) : (decodeCarry v).Canonical :=
  ⟨(v 0).isLt, (v 1).isLt, (v 2).isLt⟩

/-- The four Stage 1 application-state words of a memory application: the
spec §11.1 state digest of the application words and the carry words. -/
def stateWords (app carry : List F) : List F :=
  digestWords (squeezeDigest (absorbed [textWords "Nightstream/Nebula/v3/state", app, carry]))

theorem stateWords_length (app carry : List F) : (stateWords app carry).length = 4 := by
  simp [stateWords, digestWords]

/-- A state collision: two different absorbed chunk sequences of the state
digest with the same four words. -/
def StateCollision (app carry app' carry' : List F) : Prop :=
  TranscriptCollision [textWords "Nightstream/Nebula/v3/state", app, carry]
    [textWords "Nightstream/Nebula/v3/state", app', carry']

/-- Equal state words bind the application words and the carry words, or give
a Poseidon2 transcript collision. -/
theorem stateWords_eq_or_collision {app carry app' carry' : List F}
    (small : app.length < goldilocksModulus) (small' : app'.length < goldilocksModulus)
    (carrySmall : carry.length < goldilocksModulus)
    (carrySmall' : carry'.length < goldilocksModulus)
    (same : stateWords app carry = stateWords app' carry') :
    (app = app' ∧ carry = carry') ∨ StateCollision app carry app' carry' := by
  by_cases chunks : blockChunks [textWords "Nightstream/Nebula/v3/state", app, carry] =
      blockChunks [textWords "Nightstream/Nebula/v3/state", app', carry']
  · have tag : (textWords "Nightstream/Nebula/v3/state").length < goldilocksModulus := by
      simp [textWords, goldilocksModulus]
    have blocksEq := blockChunks_injective
      (bs := [textWords "Nightstream/Nebula/v3/state", app, carry])
      (bs' := [textWords "Nightstream/Nebula/v3/state", app', carry'])
      (by simp only [List.mem_cons, List.mem_nil_iff, or_false]
          rintro b (rfl | rfl | rfl) <;> assumption)
      (by simp only [List.mem_cons, List.mem_nil_iff, or_false]
          rintro b (rfl | rfl | rfl) <;> assumption)
      chunks
    simp only [List.cons.injEq, true_and, and_true] at blocksEq
    exact Or.inl blocksEq
  · exact Or.inr ⟨chunks, List.ofFn_injective same⟩

end NightstreamFPrime.Lifecycle.Nebula
