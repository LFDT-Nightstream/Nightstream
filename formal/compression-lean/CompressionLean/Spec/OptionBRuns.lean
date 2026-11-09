import CompressionLean.Spec.Bound
import CompressionLean.Ports.OptionB.Program

/-! The runs of option B's final program that its bound reads (CW-6.5), as `Spec.Program` and
`Spec.Bound` give today's: the program, its shrink run and counts, the words of a witness, and for
each slot the statement, seed and proof that its layer-1 verifier reads (`Reads`). The terminal of
a statement is an environment of the finish columns in which the program's checks hold
(`FinishTerminal`): the statement and the state keys on their words, the finish package's rows,
and each Poseidon2 block's outputs as the permutation of its lanes. Each slot's layer-1 statement
is its next claim's words (`slotClaim`). -/

namespace CompressionLean.Spec.OptionB

open CompressionLean.Circuit CompressionLean.OptionB
open CompressionLean.Layer1
open CompressionLean.Replay (Word)
open NightstreamFPrime.Export.OptionB.StepPackage (Package PackagePiece)
open NightstreamFPrime.Gadgets.Poseidon2 (Permutation.runF Permutation.schedule)

section Runs

variable (finish : Package) (stateKeys : List Word) (keys : Fin 2 → Layer1Key)

/-- `final_program.rs::FinalProgram::run`. Irreducible, so that no check evaluates the program. -/
@[irreducible] def program : Build Unit := finalProgram finish stateKeys keys

noncomputable def shrinkRun (x : StateWords) : Shrink.Run NightstreamFPrime.Spec.F :=
  lower (program finish stateKeys keys).program (List.ofFn x)

noncomputable def counts : Shrink.Counts :=
  (lower (program finish stateKeys keys).program ([] : List NightstreamFPrime.Spec.F)).counts

noncomputable def wordsOf (x : StateWords) (w : ℕ → NightstreamFPrime.Spec.F) : Var → Word :=
  let L := (program finish stateKeys keys).program
  let stmt := List.ofFn x
  fun h => Word.ofF
    ((Lowering.Exec.run NightstreamFPrime.Spec.Poseidon2.permute stmt
      (Lowering.tapeOf Circuit.Block.poseidonTemplate L stmt w) L).value h)

@[irreducible] def prefixRun : (Array Var × Var) × Builder := (finishPrefix finish stateKeys).run {}

/-- The builder before slot `slot`'s layer-1 call. -/
@[irreducible] def slotBuilder (slot : Fin 2) : Builder :=
  if slot = 0 then (prefixRun finish stateKeys).2
  else ((slotLayer1 finish (keys 0) (prefixRun finish stateKeys).1.2 0).run (prefixRun finish stateKeys).2).2

@[irreducible] def seedRun (slot : Fin 2) : (Fin 16 → Var) × Builder :=
  (seed (fun k => (prefixRun finish stateKeys).1.2 + k) (bitsStart finish.application.1) slot).run
    (slotBuilder finish stateKeys keys slot)

@[irreducible] def handoffRun (slot : Fin 2) : DigestVar × Builder :=
  (FoldTranscript.fromState (seedRun finish stateKeys keys slot).1).handoff.run
    (seedRun finish stateKeys keys slot).2

@[irreducible] def viewRun (slot : Fin 2) : ProofView × Builder :=
  (ProofView.read (keys slot).relation (keys slot).p1Key.rows (keys slot).p0Key.rows).run
    (handoffRun finish stateKeys keys slot).2

def statementWith (slot : Fin 2) (w : Var → Word) : Replay.Statement :=
  (claimAt (fun k => (prefixRun finish stateKeys).1.2 + k) (slotStart finish.application.1 (5 + slot))).decode w

def seedWith (slot : Fin 2) (w : Var → Word) : Array Word :=
  FoldTranscript.handoffWords (Array.ofFn fun lane => w ((seedRun finish stateKeys keys slot).1 lane))

def proofWith (slot : Fin 2) (w : Var → Word) : Replay.Proof := (viewRun finish stateKeys keys slot).1.decode w

end Runs

/-- A layer-1 position reads `st`, `seed` and `proof`: its messages are the proof's, its draws the
replay's, the proof opens where the replay runs, and its leaves bind the setup (`Spec.Reads`). -/
def ReadsWith (key : Layer1Key) (S : Setup Gl) (L0 L1 : ℕ) (st : Replay.Statement) (seed : Array Word)
    (proof : Replay.Proof) (p : (game key S L0 L1).Pos) : Prop :=
  ∃ (cands : List (P0Tables Gl)) (short0 : cands.length ≤ L0) (p1 : List (P1Tables Cubic))
    (short1 : p1.length ≤ L1),
    Game.Pos.msgs _ p = layer1Msgs key.relation st proof ⟨cands, short0⟩ ⟨p1, short1⟩ ∧
    Game.Pos.draws _ p =
      ((eval (verifyStages key.relation seed st proof) start).2.draws.toList).map
        (drawn (K := Cubic) key.relation.queries) ∧
    (∀ points t', Runs (Replay.verify key.relation seed st proof) start points t' →
      WhirOpens (shapeOf key.relation) proof points cands p1) ∧
    MerkleBinds key.relation S proof.leaves

/-- Slot `slot`'s layer-1 position reads what the witness's run gives its verifier. -/
def Reads (finish : Package) (stateKeys : List Word) (keys : Fin 2 → Layer1Key) (slot : Fin 2)
    (S : Setup Gl) (L0 L1 : ℕ) (x : StateWords) (wit : ℕ → Gl) (p : (game (keys slot) S L0 L1).Pos) : Prop :=
  let w := wordsOf finish stateKeys keys x wit
  ReadsWith (keys slot) S L0 L1 (statementWith finish stateKeys slot w)
    (seedWith finish stateKeys keys slot w) (proofWith finish stateKeys keys slot w) p

/-- The final program's checks for statement `x` hold on the finish columns `env`: a positive
iteration; column 0 is one; the iteration, `z0`, the current state and the state keys are on their
words; every row of the package's pieces holds; and each Poseidon2 block's outputs are the
permutation of its input lanes. -/
def FinishTerminal (finish : Package) (stateKeys : List Word) (x : StateWords)
    (env : NightstreamFPrime.Circuit.Env) : Prop :=
  x 0 ≠ 0 ∧ env 0 = 1 ∧ env iterationWord = x 0 ∧
    (∀ lane : Fin 4, env (z0Word + lane) = x ⟨1 + lane, by omega⟩ ∧
      env (currentWord + lane) = x ⟨5 + lane, by omega⟩) ∧
    (∀ k (inside : k < stateKeys.length), env (keysWord + k) = Word.cast stateKeys[k]) ∧
    (∀ r ∈ finish.pieces.flatMap PackagePiece.relationRows, r.Holds env) ∧
    ∀ block ∈ finish.blocks, (fun lane : Fin 16 => env (block.start + 1080 + lane)) =
      Permutation.runF Permutation.schedule fun lane => (block.lanes.getD lane 0).eval env

/-- A lane is linear and reads columns below `bound`. -/
def laneFits (bound : ℕ) (lane : NightstreamFPrime.Circuit.Expr) : Bool :=
  match linearForm lane with
  | some form => form.2.all fun t => decide (t.1 < bound)
  | none => false

/-- The finish package fits its columns: every word the program reads lies below the S-box
columns, and each block has 16 linear lanes and its outputs there. -/
def fits (finish : Package) : Bool :=
  decide (bitsStart finish.application.1 + 256 ≤ finish.sboxStart) &&
    finish.blocks.all fun block => decide (block.lanes.length = 16) &&
      decide (block.start + 1096 ≤ finish.sboxStart) && block.lanes.all (laneFits finish.sboxStart)

/-- Slot `slot`'s layer-1 statement: its next claim's words in the finish columns `env`. -/
def slotClaim (finish : Package) (slot : ℕ) (env : NightstreamFPrime.Circuit.Env) : Layer1.Statement Gl :=
  statementOf ((claimAt id (slotStart finish.application.1 (5 + slot))).decode fun k => Word.ofF (env k))

end CompressionLean.Spec.OptionB
