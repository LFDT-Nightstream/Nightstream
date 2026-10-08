import NightstreamFPrime.Lifecycle.Nebula.Sponge
import NightstreamFPrime.Lifecycle.Nebula.Framing

/-! Owns the v1.1 transcript framing over expressions for the sponge child:
self-delimiting blocks, the rate chunks of a block sequence, and the proof that
the sponge's chunk fold is `Transcript.absorbBlocks`. It also owns the two
state facts the memory circuit reads: the zero start state and one permutation
as the absorption of an empty chunk. -/

namespace NightstreamFPrime.Lifecycle.Nebula

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2

/-- A self-delimiting block of word expressions (spec §9.1): its length, then
its words. -/
def blockE (words : List Circuit.Expr) : List Circuit.Expr :=
  .const (natWord words.length) :: words

theorem evalList_blockE (env : Env) (words : List Circuit.Expr) :
    Hash.evalList env (blockE words) = block (Hash.evalList env words) := by
  simp [blockE, block, Hash.evalList]

/-- The rate chunks that the v1.1 transcript absorbs for a block sequence. -/
def transcriptChunks (blocks : List (List Circuit.Expr)) : List (List Circuit.Expr) :=
  blocks.flatMap fun b => Hash.inputChunks (blockE b)

/-- The sponge's chunk fold over transcript chunks is the transcript's block
absorption. -/
theorem fold_transcriptChunks (env : Env) (blocks : List (List Circuit.Expr))
    (s : Spec.Poseidon2.State) :
    ((transcriptChunks blocks).map (Hash.evalList env)).foldl Spec.Poseidon2.absorbBlock s =
      Transcript.absorbBlocks s (blocks.map (Hash.evalList env)) := by
  induction blocks generalizing s with
  | nil => rfl
  | cons b bs ih =>
    rw [transcriptChunks, List.flatMap_cons, List.map_append, List.foldl_append,
      ← transcriptChunks, ih, Hash.inputChunks_eval, evalList_blockE]
    rfl

/-- The constant zero state. -/
theorem evalState_zero (env : Env) : Sponge.evalState env Hash.zeroE = Transcript.initialState := by
  simp [Sponge.evalState, Transcript.initialState, Spec.Poseidon2.zeroState, Spec.Poseidon2.width,
    Hash.zeroF]

/-- Absorbing an empty chunk is one permutation. -/
theorem absorbBlock_nil {s : Spec.Poseidon2.State} (width : s.length = Spec.Poseidon2.width) :
    Spec.Poseidon2.absorbBlock s [] = Spec.Poseidon2.permute s := by
  rw [Spec.Poseidon2.absorbBlock]
  congr 1
  apply List.ext_getElem (by simp [width])
  intro i h₁ h₂
  simp [List.getD_eq_getElem?_getD, h₂]

theorem evalState_length (env : Env) (state : Sponge.EState) :
    (Sponge.evalState env state).length = Spec.Poseidon2.width := by
  simp [Sponge.evalState, Spec.Poseidon2.width]

/-- Lane `k` of an evaluated state. -/
theorem lane_eval (env : Env) (state : Sponge.EState) (k : Fin 16) :
    (state k).eval env = (Sponge.evalState env state).getD k 0 := by
  rw [Sponge.evalState, List.getD_eq_getElem _ _ (by simp), List.getElem_ofFn]
  rfl

end NightstreamFPrime.Lifecycle.Nebula
