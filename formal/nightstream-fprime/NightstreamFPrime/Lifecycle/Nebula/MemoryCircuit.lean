import NightstreamFPrime.Lifecycle.Nebula.SpongeTranscript
import NightstreamFPrime.Lifecycle.Nebula.RowMeaning
import NightstreamFPrime.Lifecycle.Stage1.Application
import NightstreamFPrime.Circuit.Sequence

/-! Owns the circuit of a memory application as a Stage 1 application: its
witness wires (the `Words` layout), nine sponge children (the two state
digests of spec §11.1, the three record chains of §9.2, and the §9.3 challenge
transcript with its three squeeze permutations), and the parent rows (digest
lanes against wires, the memory rows, and the application's rows). It also
owns the exact step and validity predicate that the circuit proves. It does
not own the proofs. -/

namespace NightstreamFPrime.Lifecycle.Nebula.MemoryApp

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Lifecycle.Stage1

variable (p : Geometry) (app : PortApp p)

abbrev AppInterface := Application.Interface (Words.count p app.toAppShape)

/-! ### Semantics -/

/-- Witness word `k` of a witness value list. -/
def wordOf (witness : List F) (k : ℕ) : F := witness.getD k 0

/-- The decoded step witness of a witness value list. -/
def decode (witness : List F) : StepWitness p app.toAppShape :=
  StepWitness.ofWords p app.toAppShape (wordOf witness)

/-- The step function: the state digest of the output application words and
the output carry words (spec §11.1). -/
def step (_input witness : List F) : List F :=
  stateWords (List.ofFn fun k : Fin app.width => wordOf witness (Words.appOut p app.toAppShape k))
    (List.ofFn fun k : Fin 39 => wordOf witness (Words.carryOut k))

/-- The validity predicate: the memory rows on the input state, and the
application's rows. -/
def valid (input witness : List F) : Prop :=
  (decode p app witness).RowsHold input ∧ ConstraintsHold (decode p app witness).view.env app.rows

/-! ### Wires and blocks -/

variable (i : AppInterface p app) (offset : ℕ)

/-- Witness word `k` as a wire; `0` past the witness. -/
def wire (k : ℕ) : Circuit.Expr :=
  if h : k < Words.count p app.toAppShape then i.witness offset ⟨k, h⟩ else 0

/-- A word expression with the wires in place of the word variables. -/
def sub (e : Circuit.Expr) : Circuit.Expr := Expr.subst (wire p app i offset) e

/-- `n` consecutive witness words from index map `f`. -/
def wires (f : ℕ → ℕ) (n : ℕ) : List Circuit.Expr := List.ofFn fun k : Fin n => wire p app i offset (f k)

/-- Constant words of a text tag. -/
def textE (text : String) : List Circuit.Expr := (textWords text).map Circuit.Expr.const

/-- The ops lane bits of all slots, over word variables. -/
def opsLanes : List Circuit.Expr := (List.ofFn fun j : Fin p.bOps => Sym.opLane p j.val).flatten

/-- The IS or FS lane bits of all scan slots, over word variables. -/
def scanLanes (start : ℕ → ℕ) : List Circuit.Expr :=
  (List.ofFn fun j : Fin p.bScan => Sym.scanLane p (start j.val)).flatten

def stateBlocks (appWords carry : ℕ → ℕ) : List (List Circuit.Expr) :=
  [textE "Nightstream/Nebula/v3/state", wires p app i offset appWords app.width,
    wires p app i offset carry 39]

def chainBlocks (lane : Lane) (previous : ℕ) (lanes : List Circuit.Expr) : List (List Circuit.Expr) :=
  [textE (chainTag lane),
    wire p app i offset Words.idxEff :: wires p app i offset (fun k => Words.seenPrev (previous + k)) 4,
    Expr.pack (lanes.map (sub p app i offset))]

def etaBlocks : List (List Circuit.Expr) :=
  [textE "Nightstream/Nebula/v3/eta", (digestWords (planDigest p)).map Circuit.Expr.const,
    [sub p app i offset (Sym.cIn 2)],
    wires p app i offset Words.proposal 4 ++ wires p app i offset (fun k => Words.carryIn (35 + k)) 4 ++
      wires p app i offset (fun k => Words.proposal (4 + k)) 4]

/-! ### Children -/

def absorbing (blocks : List (List Circuit.Expr)) : Sponge.Interface :=
  ⟨fun _ => Hash.zeroE, fun _ => transcriptChunks blocks⟩

/-- One permutation of `state`: the absorption of one empty chunk. -/
def permuting (state : Sponge.EState) : Sponge.Interface := ⟨fun _ => state, fun _ => [[]]⟩

def stateIn : Sponge.Interface :=
  absorbing (stateBlocks p app i offset (Words.appIn p) Words.carryIn)
def stateOut : Sponge.Interface :=
  absorbing (stateBlocks p app i offset (Words.appOut p app.toAppShape) Words.carryOut)
def chainOps : Sponge.Interface := absorbing (chainBlocks p app i offset .ops 0 (opsLanes p))
def chainInitial : Sponge.Interface :=
  absorbing (chainBlocks p app i offset .mem 4 (scanLanes p fun j => Words.initial p j 0))
def chainFinal : Sponge.Interface :=
  absorbing (chainBlocks p app i offset .mem 8 (scanLanes p fun j => Words.final p j 0))
def eta : Sponge.Interface := absorbing (etaBlocks p app i offset)

/-- The sponge length of an absorbing child. -/
def span (blocks : List (List Circuit.Expr)) : ℕ := (transcriptChunks blocks).length * 1096

/-- The rate chunks of blocks with the given word counts. -/
def chunkCount (lengths : List ℕ) : ℕ := (lengths.map fun l => (l + 1 + 11) / 12).sum

/-- The challenge transcript's chunk count. Its block lengths do not depend on
the plan digest's value, so start offsets never hash the plan. -/
def etaChunkCount : ℕ := chunkCount [(textWords "Nightstream/Nebula/v3/eta").length, 4, 1, 12]

def stateOutStart : ℕ := offset + span (stateBlocks p app i offset (Words.appIn p) Words.carryIn)
def chainOpsStart : ℕ := stateOutStart p app i offset +
  span (stateBlocks p app i offset (Words.appOut p app.toAppShape) Words.carryOut)
def chainInitialStart : ℕ :=
  chainOpsStart p app i offset + span (chainBlocks p app i offset .ops 0 (opsLanes p))
def chainFinalStart : ℕ := chainInitialStart p app i offset +
  span (chainBlocks p app i offset .mem 4 (scanLanes p fun j => Words.initial p j 0))
def etaStart : ℕ := chainFinalStart p app i offset +
  span (chainBlocks p app i offset .mem 8 (scanLanes p fun j => Words.final p j 0))
def squeeze1Start : ℕ := etaStart p app i offset + etaChunkCount * 1096
def squeeze2Start : ℕ := squeeze1Start p app i offset + 1096
def squeeze3Start : ℕ := squeeze2Start p app i offset + 1096
def endOffset : ℕ := squeeze3Start p app i offset + 1096

/-- The challenge transcript's final state in closed form (`etaState_output`). -/
def etaState : Sponge.EState :=
  Permutation.freshState (etaStart p app i offset + (etaChunkCount - 1) * 1096 + 1080)
def squeeze1 : Sponge.Interface := permuting (etaState p app i offset)
def squeeze1State : Sponge.EState := Sponge.output (squeeze1 p app i offset) (squeeze1Start p app i offset)
def squeeze2 : Sponge.Interface := permuting (squeeze1State p app i offset)
def squeeze2State : Sponge.EState := Sponge.output (squeeze2 p app i offset) (squeeze2Start p app i offset)
def squeeze3 : Sponge.Interface := permuting (squeeze2State p app i offset)
def squeeze3State : Sponge.EState := Sponge.output (squeeze3 p app i offset) (squeeze3Start p app i offset)

/-- The nine children in order, each at its start. -/
def childOps : List Op :=
  [Sequence.childOp "nebula.state_in" (Sponge.circuit (stateIn p app i offset)) offset,
    Sequence.childOp "nebula.state_out" (Sponge.circuit (stateOut p app i offset))
      (stateOutStart p app i offset),
    Sequence.childOp "nebula.chain_ops" (Sponge.circuit (chainOps p app i offset))
      (chainOpsStart p app i offset),
    Sequence.childOp "nebula.chain_is" (Sponge.circuit (chainInitial p app i offset))
      (chainInitialStart p app i offset),
    Sequence.childOp "nebula.chain_fs" (Sponge.circuit (chainFinal p app i offset))
      (chainFinalStart p app i offset),
    Sequence.childOp "nebula.eta" (Sponge.circuit (eta p app i offset)) (etaStart p app i offset),
    Sequence.childOp "nebula.eta_squeeze_1" (Sponge.circuit (squeeze1 p app i offset))
      (squeeze1Start p app i offset),
    Sequence.childOp "nebula.eta_squeeze_2" (Sponge.circuit (squeeze2 p app i offset))
      (squeeze2Start p app i offset),
    Sequence.childOp "nebula.eta_squeeze_3" (Sponge.circuit (squeeze3 p app i offset))
      (squeeze3Start p app i offset)]

/-- Lane `k` of a digest state. -/
def lane (state : Sponge.EState) (k : Fin 4) : Circuit.Expr := state ⟨k.val, by omega⟩

/-- The hash rows by check: each digest against its wires, and the challenge
words. -/
def hashChecks : Checks :=
  [("state_in", (List.finRange 4).map (fun k =>
      lane (Sponge.output (stateIn p app i offset) offset) k - i.input offset k)),
    ("state_out", (List.finRange 4).map (fun k =>
      lane (Sponge.output (stateOut p app i offset) (stateOutStart p app i offset)) k -
        i.output offset k)),
    ("chain_ops", (List.finRange 4).map (fun k =>
      lane (Sponge.output (chainOps p app i offset) (chainOpsStart p app i offset)) k -
        wire p app i offset (Words.carryOut (23 + k.val)))),
    ("chain_is", (List.finRange 4).map (fun k =>
      lane (Sponge.output (chainInitial p app i offset) (chainInitialStart p app i offset)) k -
        wire p app i offset (Words.carryOut (27 + k.val)))),
    ("chain_fs", (List.finRange 4).map (fun k =>
      lane (Sponge.output (chainFinal p app i offset) (chainFinalStart p app i offset)) k -
        wire p app i offset (Words.carryOut (31 + k.val)))),
    ("eta", [lane (etaState p app i offset) 0 - wire p app i offset (Words.etaFresh 0),
      lane (squeeze1State p app i offset) 0 - wire p app i offset (Words.etaFresh 1),
      lane (squeeze2State p app i offset) 0 - wire p app i offset (Words.etaFresh 2),
      lane (squeeze3State p app i offset) 0 - wire p app i offset (Words.etaFresh 3)])]

/-- The parent rows: the hash rows and the polynomial rows. -/
def assertions : List Circuit.Expr :=
  joinChecks (hashChecks p app i offset) ++ (Rows.polyRows p app).map (sub p app i offset)

def opsAt : List Op := childOps p app i offset ++ (assertions p app i offset).map Op.assertZero

def main : Circuit Unit := fun offset => ((), endOffset p app i offset, opsAt p app i offset)

theorem transcriptChunks_length (blocks : List (List Circuit.Expr)) :
    (transcriptChunks blocks).length = chunkCount (blocks.map List.length) := by
  induction blocks with
  | nil => rfl
  | cons b rest ih =>
    rw [transcriptChunks, List.flatMap_cons, List.length_append, ← transcriptChunks, ih]
    simp only [chunkCount, Gadgets.Poseidon2.Hash.inputChunks, blockE, Spec.Poseidon2.rate,
      List.length_map, List.length_range, List.length_cons, List.map_cons, List.sum_cons]
    omega

theorem etaChunks_length (start : ℕ) : ((eta p app i offset).chunks start).length = etaChunkCount := by
  change (transcriptChunks (etaBlocks p app i offset)).length = _
  rw [transcriptChunks_length]
  simp [etaBlocks, wires, textE, etaChunkCount, digestWords]

theorem etaState_output : etaState p app i offset = Sponge.output (eta p app i offset) (etaStart p app i offset) := by
  have nonempty : (eta p app i offset).chunks (etaStart p app i offset) ≠ [] := by
    intro empty
    have length := etaChunks_length p app i offset (etaStart p app i offset)
    rw [empty] at length
    exact absurd length (by decide)
  rw [Sponge.output_eq_program, Sponge.program, Sponge.compileAbsorptions_output _ nonempty,
    etaChunks_length]
  rfl

end NightstreamFPrime.Lifecycle.Nebula.MemoryApp
