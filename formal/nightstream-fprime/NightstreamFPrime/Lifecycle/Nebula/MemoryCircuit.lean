import NightstreamFPrime.Lifecycle.Nebula.SpongeTranscript
import NightstreamFPrime.Lifecycle.Nebula.RowMeaning
import NightstreamFPrime.Lifecycle.Stage1.Application
import NightstreamFPrime.Circuit.Sequence

/-! Owns the circuit of the first memory application as a Stage 1 application:
its witness wires (the `Words` layout), nine sponge children (the two state
digests of spec §11.1, the three record chains of §9.2, and the §9.3 challenge
transcript with its three squeeze permutations), and the parent rows (digest
lanes against wires, and the polynomial rows). It also owns the exact step and
validity predicate that the circuit proves. It does not own the proofs. -/

namespace NightstreamFPrime.Lifecycle.Nebula.MemoryApp

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Lifecycle.Stage1

variable (p : Plan)

abbrev AppInterface := Application.Interface (Words.count p)

/-! ### Semantics -/

/-- Witness word `k` of a witness value list. -/
def wordOf (witness : List F) (k : ℕ) : F := witness.getD k 0

/-- The decoded step witness of a witness value list. -/
def decode (witness : List F) : StepWitness p := StepWitness.ofWords p (wordOf witness)

/-- The step function: the state digest of the output application words and
the output carry words (spec §11.1). -/
def step (_input witness : List F) : List F :=
  stateWords (List.ofFn fun k : Fin 2 => wordOf witness (Words.appOut k))
    (List.ofFn fun k : Fin 39 => wordOf witness (Words.carryOut k))

/-- The validity predicate: the memory rows on the input state, and the
machine rows. -/
def valid (two : p.bOps = 2) (input witness : List F) : Prop :=
  (decode p witness).RowsHold input ∧ (decode p witness).MachineRows two

/-! ### Wires and blocks -/

variable (i : AppInterface p) (offset : ℕ)

/-- Witness word `k` as a wire; `0` past the witness. -/
def wire (k : ℕ) : Circuit.Expr :=
  if h : k < Words.count p then i.witness offset ⟨k, h⟩ else 0

/-- A word expression with the wires in place of the word variables. -/
def sub (e : Circuit.Expr) : Circuit.Expr := Expr.subst (wire p i offset) e

/-- `n` consecutive witness words from index map `f`. -/
def wires (f : ℕ → ℕ) (n : ℕ) : List Circuit.Expr := List.ofFn fun k : Fin n => wire p i offset (f k)

/-- Constant words of a text tag. -/
def textE (text : String) : List Circuit.Expr := (textWords text).map Circuit.Expr.const

/-- The ops lane bits of all slots, over word variables. -/
def opsLanes : List Circuit.Expr := (List.ofFn fun j : Fin p.bOps => Sym.opLane p j.val).flatten

/-- The IS or FS lane bits of all scan slots, over word variables. -/
def scanLanes (start : ℕ → ℕ) : List Circuit.Expr :=
  (List.ofFn fun j : Fin p.bScan => Sym.scanLane p (start j.val)).flatten

def stateBlocks (app carry : ℕ → ℕ) : List (List Circuit.Expr) :=
  [textE "Nightstream/Nebula/v3/state", wires p i offset app 2, wires p i offset carry 39]

def chainBlocks (lane : Lane) (previous : ℕ) (lanes : List Circuit.Expr) : List (List Circuit.Expr) :=
  [textE (chainTag lane),
    wire p i offset Words.idxEff :: wires p i offset (fun k => Words.seenPrev (previous + k)) 4,
    Expr.pack (lanes.map (sub p i offset))]

def etaBlocks : List (List Circuit.Expr) :=
  [textE "Nightstream/Nebula/v3/eta", (digestWords (planDigest p)).map Circuit.Expr.const,
    [sub p i offset (Sym.cIn 2)],
    wires p i offset Words.proposal 4 ++ wires p i offset (fun k => Words.carryIn (35 + k)) 4 ++
      wires p i offset (fun k => Words.proposal (4 + k)) 4]

/-! ### Children -/

def absorbing (blocks : List (List Circuit.Expr)) : Sponge.Interface :=
  ⟨fun _ => Hash.zeroE, fun _ => transcriptChunks blocks⟩

/-- One permutation of `state`: the absorption of one empty chunk. -/
def permuting (state : Sponge.EState) : Sponge.Interface := ⟨fun _ => state, fun _ => [[]]⟩

def stateIn : Sponge.Interface := absorbing (stateBlocks p i offset Words.appIn Words.carryIn)
def stateOut : Sponge.Interface := absorbing (stateBlocks p i offset Words.appOut Words.carryOut)
def chainOps : Sponge.Interface := absorbing (chainBlocks p i offset .ops 0 (opsLanes p))
def chainInitial : Sponge.Interface :=
  absorbing (chainBlocks p i offset .mem 4 (scanLanes p fun j => Words.initial p j 0))
def chainFinal : Sponge.Interface :=
  absorbing (chainBlocks p i offset .mem 8 (scanLanes p fun j => Words.final p j 0))
def eta : Sponge.Interface := absorbing (etaBlocks p i offset)

/-- The sponge length of an absorbing child. -/
def span (blocks : List (List Circuit.Expr)) : ℕ := (transcriptChunks blocks).length * 1096

def stateOutStart : ℕ := offset + span (stateBlocks p i offset Words.appIn Words.carryIn)
def chainOpsStart : ℕ :=
  stateOutStart p i offset + span (stateBlocks p i offset Words.appOut Words.carryOut)
def chainInitialStart : ℕ :=
  chainOpsStart p i offset + span (chainBlocks p i offset .ops 0 (opsLanes p))
def chainFinalStart : ℕ := chainInitialStart p i offset +
  span (chainBlocks p i offset .mem 4 (scanLanes p fun j => Words.initial p j 0))
def etaStart : ℕ := chainFinalStart p i offset +
  span (chainBlocks p i offset .mem 8 (scanLanes p fun j => Words.final p j 0))
def squeeze1Start : ℕ := etaStart p i offset + span (etaBlocks p i offset)
def squeeze2Start : ℕ := squeeze1Start p i offset + 1096
def squeeze3Start : ℕ := squeeze2Start p i offset + 1096
def endOffset : ℕ := squeeze3Start p i offset + 1096

def etaState : Sponge.EState := Sponge.output (eta p i offset) (etaStart p i offset)
def squeeze1 : Sponge.Interface := permuting (etaState p i offset)
def squeeze1State : Sponge.EState := Sponge.output (squeeze1 p i offset) (squeeze1Start p i offset)
def squeeze2 : Sponge.Interface := permuting (squeeze1State p i offset)
def squeeze2State : Sponge.EState := Sponge.output (squeeze2 p i offset) (squeeze2Start p i offset)
def squeeze3 : Sponge.Interface := permuting (squeeze2State p i offset)
def squeeze3State : Sponge.EState := Sponge.output (squeeze3 p i offset) (squeeze3Start p i offset)

/-- The nine children in order, each at its start. -/
def childOps : List Op :=
  [Sequence.childOp "nebula.state_in" (Sponge.circuit (stateIn p i offset)) offset,
    Sequence.childOp "nebula.state_out" (Sponge.circuit (stateOut p i offset))
      (stateOutStart p i offset),
    Sequence.childOp "nebula.chain_ops" (Sponge.circuit (chainOps p i offset))
      (chainOpsStart p i offset),
    Sequence.childOp "nebula.chain_is" (Sponge.circuit (chainInitial p i offset))
      (chainInitialStart p i offset),
    Sequence.childOp "nebula.chain_fs" (Sponge.circuit (chainFinal p i offset))
      (chainFinalStart p i offset),
    Sequence.childOp "nebula.eta" (Sponge.circuit (eta p i offset)) (etaStart p i offset),
    Sequence.childOp "nebula.eta_squeeze_1" (Sponge.circuit (squeeze1 p i offset))
      (squeeze1Start p i offset),
    Sequence.childOp "nebula.eta_squeeze_2" (Sponge.circuit (squeeze2 p i offset))
      (squeeze2Start p i offset),
    Sequence.childOp "nebula.eta_squeeze_3" (Sponge.circuit (squeeze3 p i offset))
      (squeeze3Start p i offset)]

/-- Lane `k` of a digest state. -/
def lane (state : Sponge.EState) (k : Fin 4) : Circuit.Expr := state ⟨k.val, by omega⟩

/-- The parent rows: each digest against its wires, the challenge words, and
the polynomial rows. -/
def assertions : List Circuit.Expr :=
  (List.finRange 4).map (fun k =>
    lane (Sponge.output (stateIn p i offset) offset) k - i.input offset k) ++
  (List.finRange 4).map (fun k =>
    lane (Sponge.output (stateOut p i offset) (stateOutStart p i offset)) k - i.output offset k) ++
  (List.finRange 4).map (fun k =>
    lane (Sponge.output (chainOps p i offset) (chainOpsStart p i offset)) k -
      wire p i offset (Words.carryOut (23 + k.val))) ++
  (List.finRange 4).map (fun k =>
    lane (Sponge.output (chainInitial p i offset) (chainInitialStart p i offset)) k -
      wire p i offset (Words.carryOut (27 + k.val))) ++
  (List.finRange 4).map (fun k =>
    lane (Sponge.output (chainFinal p i offset) (chainFinalStart p i offset)) k -
      wire p i offset (Words.carryOut (31 + k.val))) ++
  [lane (etaState p i offset) 0 - wire p i offset (Words.etaFresh 0),
    lane (squeeze1State p i offset) 0 - wire p i offset (Words.etaFresh 1),
    lane (squeeze2State p i offset) 0 - wire p i offset (Words.etaFresh 2),
    lane (squeeze3State p i offset) 0 - wire p i offset (Words.etaFresh 3)] ++
  (Rows.polyRows p).map (sub p i offset)

def opsAt : List Op := childOps p i offset ++ (assertions p i offset).map Op.assertZero

def main : Circuit Unit := fun offset => ((), endOffset p i offset, opsAt p i offset)

end NightstreamFPrime.Lifecycle.Nebula.MemoryApp
