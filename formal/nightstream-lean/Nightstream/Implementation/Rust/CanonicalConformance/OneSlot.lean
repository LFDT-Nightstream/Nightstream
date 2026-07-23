import Nightstream.Protocol.FPrime.CanonicalTerminalVerifier
import Nightstream.Protocol.FPrime.CanonicalVerifier

/-!
One-slot execution receipts for differential testing of the frozen canonical
F' verifiers.

Owns: a proof-free schema for recording the primitive calls made by one
canonical step or terminal check, reconstruction of the corresponding typed
paper inputs, and executable comparison against an externally supplied Rust
Boolean.

Does not own: Rust correctness, serialization, generated artifacts, R1CS,
production profile guards, or a refinement theorem.  In particular,
`rustAccepted` is never an input to a reconstructed setup, machine, primitive
receipt, or canonical evaluation; it occurs only on the right-hand side of the
two differential comparisons.

This first schema covers full accepted primitive paths and mutations of their
post-execution claims.  It does not yet represent early prior-link or NIFS
rejection traces, so the comparison hook is not itself a Rust differential
conformance result.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Rust.CanonicalConformance.OneSlot

open Nightstream.HyperNova.Construction2.Paper

/-- Sort tags keep unrelated receipt values distinct without storing proofs. -/
inductive AtomSort where
  | key
  | digest
  | state
  | witness
  | running
  | runningWitness
  | fresh
  | freshWitness
  | nifsProof
  | encoded
deriving Repr, DecidableEq

/-- A compact, proof-free identifier at one protocol sort.  Zero is reserved
as the failed-lookup value used by reconstructed primitive functions. -/
structure Atom (sort : AtomSort) where
  value : Nat
deriving Repr, DecidableEq

abbrev Key := Atom .key
abbrev Digest := Atom .digest
abbrev State := Atom .state
abbrev Witness := Atom .witness
abbrev Running := Atom .running
abbrev RunningWitness := Atom .runningWitness
abbrev Fresh := Atom .fresh
abbrev FreshWitness := Atom .freshWitness
abbrev NifsProof := Atom .nifsProof
abbrev Encoded := Atom .encoded

def slotCount : Nat := 1

def onlySlot : Fin slotCount :=
  ⟨0, by simp [slotCount]⟩

def poison (sort : AtomSort) : Atom sort :=
  ⟨0⟩

def live {sort : AtomSort} (value : Atom sort) : Bool :=
  decide (value.value ≠ 0)

/-- Function-valued one-slot paper preimages are normalized to compact data
before comparison. -/
structure HashInput where
  verifierKey : Key
  iteration : Nat
  z0 : State
  current : State
  running : Running
  pc : Nat
deriving Repr, DecidableEq

structure HashReceipt where
  input : HashInput
  output : Digest
deriving Repr, DecidableEq

structure StepReceipt where
  state : State
  witness : Witness
  output : State
deriving Repr, DecidableEq

structure FreshPublicReceipt where
  input : Fresh
  output : Encoded
deriving Repr, DecidableEq

structure EncodeReceipt where
  input : Digest
  output : Encoded
deriving Repr, DecidableEq

structure NifsReceipt where
  key : Key
  running : Running
  fresh : Fresh
  proof : NifsProof
  output : Running
deriving Repr, DecidableEq

/-- Branch-tagged receipts in canonical call order.  The base branch executes
only the next-output hash after the common application step.  The recursive
branch executes the prior hash, public encodings, NIFS verifier, and next hash. -/
inductive StepTrace where
  | base (nextHash : HashReceipt)
  | recursive
      (priorHash : HashReceipt)
      (freshPublic : FreshPublicReceipt)
      (encode : EncodeReceipt)
      (nifs : NifsReceipt)
      (nextHash : HashReceipt)
deriving Repr, DecidableEq

/-- Claimed public output, with the one-slot program counter represented as a
plain natural so malformed external values remain expressible. -/
structure StepClaim where
  zNext : State
  runningNext : Running
  pcNext : Nat
  x : Digest
deriving Repr, DecidableEq

/-- One proof-free step comparison case. -/
structure StepCase where
  verifierKey : Key
  defaultRunning : Running
  iteration : Nat
  z0 : State
  zi : State
  running : Running
  fresh : Fresh
  priorPc : Nat
  witness : Witness
  nifsProof : NifsProof
  stepReceipt : StepReceipt
  trace : StepTrace
  claim : StepClaim
  rustAccepted : Bool
deriving Repr, DecidableEq

def normalizeHashInput
    (input : HashPreimage Key State Running slotCount) : HashInput where
  verifierKey := input.verifierKeys onlySlot
  iteration := input.iteration
  z0 := input.z0
  current := input.current
  running := input.running onlySlot
  pc := input.pc

def hashReceiptOutput (receipt : HashReceipt) (input : HashInput) : Digest :=
  if receipt.input = input then receipt.output else poison .digest

def stepHashOutput (trace : StepTrace) (input : HashInput) : Digest :=
  match trace with
  | .base nextHash => hashReceiptOutput nextHash input
  | .recursive priorHash _ _ _ nextHash =>
      if priorHash.input = input then priorHash.output
      else hashReceiptOutput nextHash input

def stepSetup (case : StepCase) :
    Setup Key Running Fresh NifsProof slotCount where
  verifierKeys := fun _ => case.verifierKey
  nifs := {
    verify := fun key running fresh proof =>
      match case.trace with
      | .base _ => none
      | .recursive _ _ _ receipt _ =>
          if key = receipt.key ∧ running = receipt.running ∧
              fresh = receipt.fresh ∧ proof = receipt.proof then
            some receipt.output
          else
            none
  }
  defaultRunning := case.defaultRunning

def stepMachine (case : StepCase) :
    Machine Key Digest State Witness Running Fresh Encoded slotCount where
  -- A one-slot dispatch has exactly one possible result.
  control := fun _ _ => onlySlot
  step := fun _ state witness =>
    if state = case.stepReceipt.state ∧ witness = case.stepReceipt.witness then
      case.stepReceipt.output
    else
      poison .state
  freshPublic := fun fresh =>
    match case.trace with
    | .base _ => poison .encoded
    | .recursive _ receipt _ _ _ =>
        if fresh = receipt.input then receipt.output else poison .encoded
  encodeInstance := fun digest =>
    match case.trace with
    | .base _ => poison .encoded
    | .recursive _ _ receipt _ _ =>
        if digest = receipt.input then receipt.output else poison .encoded
  hash := fun input => stepHashOutput case.trace (normalizeHashInput input)

def stepInput (case : StepCase) :
    Input Key State Witness Running Fresh NifsProof slotCount where
  iteration := case.iteration
  z0 := case.z0
  zi := case.zi
  running := fun _ => case.running
  fresh := case.fresh
  priorPc := case.priorPc
  witness := case.witness
  nifsProof := case.nifsProof

/-- The recursive prior-link call, normalized before any receipt lookup. -/
def stepPriorHashInput (case : StepCase) : HashInput where
  verifierKey := case.verifierKey
  iteration := case.iteration
  z0 := case.z0
  current := case.zi
  running := case.running
  pc := case.priorPc

def stepBaseNextHashInput (case : StepCase) : HashInput where
  verifierKey := case.verifierKey
  iteration := case.iteration + 1
  z0 := case.z0
  current := case.stepReceipt.output
  running := case.defaultRunning
  pc := 1

def stepRecursiveNextHashInput (case : StepCase) (folded : Running) : HashInput where
  verifierKey := case.verifierKey
  iteration := case.iteration + 1
  z0 := case.z0
  current := case.stepReceipt.output
  running := folded
  pc := 1

/-- Exact branch and call conservation for a one-slot canonical step.  Every
stored receipt is matched to one canonical call position; base cases carry no
unused recursive primitive data. -/
def stepSchemaAccepted (case : StepCase) : Bool :=
  decide (case.stepReceipt.state = case.zi ∧
    case.stepReceipt.witness = case.witness) &&
  live case.stepReceipt.output &&
  live case.claim.zNext &&
  live case.claim.runningNext &&
  live case.claim.x &&
  match case.trace with
  | .base nextHash =>
      decide (case.iteration = 0 ∧ case.z0 = case.zi ∧
        nextHash.input = stepBaseNextHashInput case) &&
      live nextHash.output
  | .recursive priorHash freshPublic encode nifs nextHash =>
      decide (case.iteration ≠ 0 ∧
        case.priorPc = 1 ∧
        priorHash.input = stepPriorHashInput case ∧
        freshPublic.input = case.fresh ∧
        encode.input = priorHash.output ∧
        freshPublic.output = encode.output ∧
        nifs.key = case.verifierKey ∧
        nifs.running = case.running ∧
        nifs.fresh = case.fresh ∧
        nifs.proof = case.nifsProof ∧
        nextHash.input = stepRecursiveNextHashInput case nifs.output) &&
      live priorHash.output &&
      live nextHash.output &&
      live freshPublic.output &&
      live encode.output &&
      live nifs.output

def outputMatches
    (claim : StepClaim)
    (output : Output Digest State Running slotCount) : Bool :=
  decide (output.zNext = claim.zNext ∧
    output.runningNext onlySlot = claim.runningNext ∧
    output.pcNext.val = claim.pcNext ∧
    output.x = claim.x)

/-- Execute the frozen canonical step verifier over the reconstructed receipt
machine and compare every public output field. -/
def stepAccepted (case : StepCase) : Bool :=
  stepSchemaAccepted case &&
    match Nightstream.Protocol.FPrime.CanonicalVerifier.eval
        (stepSetup case) (stepMachine case) onlySlot (stepInput case) with
    | none => false
    | some output => outputMatches case.claim output

/-- The external Rust result is deliberately isolated to this final equality. -/
def stepAgrees (case : StepCase) : Bool :=
  decide (stepAccepted case = case.rustAccepted)

theorem stepAgrees_eq_true_iff (case : StepCase) :
    stepAgrees case = true ↔ stepAccepted case = case.rustAccepted := by
  simp [stepAgrees]

structure RunningRelationReceipt where
  key : Key
  value : Running
  witness : RunningWitness
  accepted : Bool
deriving Repr, DecidableEq

structure FreshRelationReceipt where
  key : Key
  value : Fresh
  witness : FreshWitness
  accepted : Bool
deriving Repr, DecidableEq

/-- Branch-tagged terminal receipts in canonical call order.  The base branch
has no primitive receipts because it checks only the advertised endpoint. -/
inductive TerminalTrace where
  | base
  | recursive
      (priorHash : HashReceipt)
      (freshPublic : FreshPublicReceipt)
      (encode : EncodeReceipt)
      (runningRelation : RunningRelationReceipt)
      (freshRelation : FreshRelationReceipt)
deriving Repr, DecidableEq

/-- One proof-free terminal comparison case.  The base branch ignores proof
data, exactly as the frozen canonical terminal verifier does. -/
structure TerminalCase where
  verifierKey : Key
  defaultRunning : Running
  iteration : Nat
  z0 : State
  zi : State
  running : Running
  runningWitness : RunningWitness
  fresh : Fresh
  freshWitness : FreshWitness
  pc : Nat
  trace : TerminalTrace
  rustAccepted : Bool
deriving Repr, DecidableEq

def terminalSetup (case : TerminalCase) :
    Setup Key Running Fresh NifsProof slotCount where
  verifierKeys := fun _ => case.verifierKey
  nifs := { verify := fun _ _ _ _ => none }
  defaultRunning := case.defaultRunning

def terminalMachine (case : TerminalCase) :
    Machine Key Digest State Witness Running Fresh Encoded slotCount where
  control := fun _ _ => onlySlot
  step := fun _ _ _ => poison .state
  freshPublic := fun fresh =>
    match case.trace with
    | .base => poison .encoded
    | .recursive _ receipt _ _ _ =>
        if fresh = receipt.input then receipt.output else poison .encoded
  encodeInstance := fun digest =>
    match case.trace with
    | .base => poison .encoded
    | .recursive _ _ receipt _ _ =>
        if digest = receipt.input then receipt.output else poison .encoded
  hash := fun input =>
    match case.trace with
    | .base => poison .digest
    | .recursive receipt _ _ _ _ =>
        hashReceiptOutput receipt (normalizeHashInput input)

def runningReceiptCheck (case : TerminalCase)
    (key : Key) (value : Running) (witness : RunningWitness) : Bool :=
  match case.trace with
  | .base => false
  | .recursive _ _ _ receipt _ =>
      decide (key = receipt.key ∧ value = receipt.value ∧
        witness = receipt.witness) && receipt.accepted

def freshReceiptCheck (case : TerminalCase)
    (key : Key) (value : Fresh) (witness : FreshWitness) : Bool :=
  match case.trace with
  | .base => false
  | .recursive _ _ _ _ receipt =>
      decide (key = receipt.key ∧ value = receipt.value ∧
        witness = receipt.witness) && receipt.accepted

def terminalRelations (case : TerminalCase) :
    TerminalRelations Key Running RunningWitness Fresh FreshWitness slotCount where
  runningHolds := fun _ key value witness =>
    runningReceiptCheck case key value witness = true
  freshHolds := fun _ key value witness =>
    freshReceiptCheck case key value witness = true

def terminalChecks (case : TerminalCase) :
    Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
      (terminalRelations case) where
  runningCheck := fun _ => runningReceiptCheck case
  freshCheck := fun _ => freshReceiptCheck case
  runningCheck_iff := by intros; rfl
  freshCheck_iff := by intros; rfl

def terminalStatement (case : TerminalCase) : TerminalStatement State where
  iteration := case.iteration
  z0 := case.z0
  zi := case.zi

def terminalProof (case : TerminalCase) :
    TerminalProof Running RunningWitness Fresh FreshWitness slotCount where
  running := fun _ => case.running
  runningWitness := fun _ => case.runningWitness
  fresh := case.fresh
  freshWitness := case.freshWitness
  pc := case.pc

def terminalPriorHashInput (case : TerminalCase) : HashInput where
  verifierKey := case.verifierKey
  iteration := case.iteration
  z0 := case.z0
  current := case.zi
  running := case.running
  pc := case.pc

/-- Exact branch and call conservation for the terminal boundary. -/
def terminalSchemaAccepted (case : TerminalCase) : Bool :=
  match case.trace with
  | .base => decide (case.iteration = 0)
  | .recursive priorHash freshPublic encode runningRelation freshRelation =>
      decide (case.iteration ≠ 0 ∧
        case.pc = 1 ∧
        priorHash.input = terminalPriorHashInput case ∧
        freshPublic.input = case.fresh ∧
        encode.input = priorHash.output ∧
        freshPublic.output = encode.output ∧
        runningRelation.key = case.verifierKey ∧
        runningRelation.value = case.running ∧
        runningRelation.witness = case.runningWitness ∧
        freshRelation.key = case.verifierKey ∧
        freshRelation.value = case.fresh ∧
        freshRelation.witness = case.freshWitness) &&
      live priorHash.output &&
      live freshPublic.output &&
      live encode.output

/-- Execute the frozen canonical terminal verifier over reconstructed receipts. -/
def terminalAccepted (case : TerminalCase) : Bool :=
  terminalSchemaAccepted case &&
    Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.eval
      (terminalSetup case) (terminalMachine case) (terminalRelations case)
      (terminalChecks case) (terminalStatement case) (terminalProof case)

/-- The external Rust result is deliberately isolated to this final equality. -/
def terminalAgrees (case : TerminalCase) : Bool :=
  decide (terminalAccepted case = case.rustAccepted)

theorem terminalAgrees_eq_true_iff (case : TerminalCase) :
    terminalAgrees case = true ↔ terminalAccepted case = case.rustAccepted := by
  simp [terminalAgrees]

end Nightstream.Implementation.Rust.CanonicalConformance.OneSlot
