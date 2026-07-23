import Nightstream.Implementation.Lowering.Typed.Program
import Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne
import Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne

/-!
Contract: artifact-independent typed lowering vocabulary for the fixed-one
HyperNova `F'` step and terminal verifiers.

Owns:
- the protocol-data sorts consumed by the fixed-one verifiers;
- the exact logical input and result schemas and their ownership classes;
- a closed set of deterministic partial calls for the paper dataflow;
- intrinsic footprint parameters for operations whose encoding is selected
  later.

Does not own: a whole-verifier call, a caller-provided acceptance proposition,
physical row or column numbers, Rust data, generated artifacts, commitment or
digest authority, or a concrete encoding recipe.

The fixed-one program counter and verifier key are verifier-static.  They are
therefore captured by the vocabulary rather than exposed as prover ports.  A
step's advice is committed, its derived state and running result are committed,
and only its digest `x` is public.  At the terminal boundary the statement is
public and the proof is committed.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed

namespace Step

abbrev Input :=
  Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Input

def selected : Fin 1 :=
  Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected

end Step

namespace Terminal

abbrev Proof :=
  Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.Proof

end Terminal

/-- Coordinate widths selected by a later encoding.  Ownership is not
parameterized: each use-site below fixes it from the paper dataflow. -/
structure Widths where
  iteration : Nat
  state : Nat
  witness : Nat
  running : Nat
  fresh : Nat
  nifsProof : Nat
  digest : Nat
  encoded : Nat
  runningWitness : Nat
  freshWitness : Nat
  bit : Nat
deriving DecidableEq, Repr

/-- Intrinsic costs not derivable from logical result allocation.

Every selected call has an explicit footprint parameter.  Exact hash
preimages are assembled inside the two hash calls from their separately
addressed authoritative operands; no zero-width semantic record is carried
between instructions. -/
structure Footprints where
  iterationZero : CallFootprint
  stateEqual : CallFootprint
  step : CallFootprint
  hash : CallFootprint
  freshPublic : CallFootprint
  encodeInstance : CallFootprint
  encodedEqual : CallFootprint
  nifsVerify : CallFootprint
  runningCheck : CallFootprint
  freshCheck : CallFootprint
deriving DecidableEq, Repr

/-- All paper operations captured by one closed fixed-one vocabulary.

`terminalRelations` is the independently stated paper relation and
`terminalChecks` is its exact executable checker.  Neither is a
caller-selected whole-verifier acceptance predicate. -/
structure Parameters where
  Field : Type
  fieldZero : Field
  fieldAdd : Field -> Field -> Field
  fieldMul : Field -> Field -> Field
  Key : Type
  Digest : Type
  State : Type
  Witness : Type
  Running : Type
  Fresh : Type
  NifsProof : Type
  Encoded : Type
  RunningWitness : Type
  FreshWitness : Type
  stateDecidableEq : DecidableEq State
  encodedDecidableEq : DecidableEq Encoded
  setup : Setup Key Running Fresh NifsProof 1
  machine :
    Machine Key Digest State Witness Running Fresh Encoded 1
  terminalRelations :
    TerminalRelations
      Key Running RunningWitness Fresh FreshWitness 1
  terminalChecks :
    Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
      terminalRelations
  widths : Widths
  footprints : Footprints

/-- Reusable protocol-data tags.  Role-specific ownership remains in ports,
not in these semantic sorts. -/
inductive DataTag where
  | nat
  | digest
  | state
  | witness
  | running
  | fresh
  | nifsProof
  | encoded
  | runningWitness
  | freshWitness
deriving DecidableEq, Repr

namespace DataTag

def denote (parameters : Parameters) : DataTag -> Type
  | .nat => Nat
  | .digest => parameters.Digest
  | .state => parameters.State
  | .witness => parameters.Witness
  | .running => parameters.Running
  | .fresh => parameters.Fresh
  | .nifsProof => parameters.NifsProof
  | .encoded => parameters.Encoded
  | .runningWitness => parameters.RunningWitness
  | .freshWitness => parameters.FreshWitness

end DataTag

/-- The semantic universe used by this lowering vocabulary.  Field operations
are present for the generic typed IR but the protocol calls below operate only
on typed paper data. -/
def typeSystem (parameters : Parameters) : TypeSystem where
  Field := parameters.Field
  zero := parameters.fieldZero
  add := parameters.fieldAdd
  mul := parameters.fieldMul
  Bit := Bool
  bitValue := id
  Data := DataTag
  dataValue := DataTag.denote parameters

def ownedLayout (ownership : Ownership) (width : Nat) : Layout :=
  ⟨List.replicate width ownership⟩

def committedLayout (width : Nat) : Layout :=
  ownedLayout .committedColumn width

def publicLayout (width : Nat) : Layout :=
  ownedLayout .publicColumn width

def auxiliaryLayout (width : Nat) : Layout :=
  ownedLayout .auxiliaryColumn width

def dataPort
    (parameters : Parameters)
    (tag : DataTag)
    (layout : Layout) :
    Port (typeSystem parameters) :=
  { kind := .data tag, layout := layout }

def bitPort
    (parameters : Parameters)
    (layout : Layout) :
    Port (typeSystem parameters) :=
  { kind := .bit, layout := layout }

namespace Ports

def committedNat (parameters : Parameters) :
    Port (typeSystem parameters) :=
  dataPort parameters .nat
    (committedLayout parameters.widths.iteration)

def publicNat (parameters : Parameters) :
    Port (typeSystem parameters) :=
  dataPort parameters .nat
    (publicLayout parameters.widths.iteration)

def committedState (parameters : Parameters) :
    Port (typeSystem parameters) :=
  dataPort parameters .state
    (committedLayout parameters.widths.state)

def publicState (parameters : Parameters) :
    Port (typeSystem parameters) :=
  dataPort parameters .state
    (publicLayout parameters.widths.state)

def committedWitness (parameters : Parameters) :
    Port (typeSystem parameters) :=
  dataPort parameters .witness
    (committedLayout parameters.widths.witness)

def committedRunning (parameters : Parameters) :
    Port (typeSystem parameters) :=
  dataPort parameters .running
    (committedLayout parameters.widths.running)

def committedFresh (parameters : Parameters) :
    Port (typeSystem parameters) :=
  dataPort parameters .fresh
    (committedLayout parameters.widths.fresh)

def committedNifsProof (parameters : Parameters) :
    Port (typeSystem parameters) :=
  dataPort parameters .nifsProof
    (committedLayout parameters.widths.nifsProof)

def publicDigest (parameters : Parameters) :
    Port (typeSystem parameters) :=
  dataPort parameters .digest
    (publicLayout parameters.widths.digest)

def auxiliaryDigest (parameters : Parameters) :
    Port (typeSystem parameters) :=
  dataPort parameters .digest
    (auxiliaryLayout parameters.widths.digest)

def auxiliaryEncoded (parameters : Parameters) :
    Port (typeSystem parameters) :=
  dataPort parameters .encoded
    (auxiliaryLayout parameters.widths.encoded)

def committedRunningWitness (parameters : Parameters) :
    Port (typeSystem parameters) :=
  dataPort parameters .runningWitness
    (committedLayout parameters.widths.runningWitness)

def committedFreshWitness (parameters : Parameters) :
    Port (typeSystem parameters) :=
  dataPort parameters .freshWitness
    (committedLayout parameters.widths.freshWitness)

def auxiliaryBit (parameters : Parameters) :
    Port (typeSystem parameters) :=
  bitPort parameters (auxiliaryLayout parameters.widths.bit)

end Ports

/-- Exact fixed-one step advice order.  Every input is committed. -/
def stepInputSchema (parameters : Parameters) :
    Schema (typeSystem parameters) :=
  [Ports.committedNat parameters,
    Ports.committedState parameters,
    Ports.committedState parameters,
    Ports.committedRunning parameters,
    Ports.committedFresh parameters,
    Ports.committedWitness parameters,
    Ports.committedNifsProof parameters]

/-- Exact exposed step result.  The fixed `pcNext = 0` is verifier-static and
therefore absent; `x` is the only public result. -/
def stepResultSchema (parameters : Parameters) :
    Schema (typeSystem parameters) :=
  [Ports.committedState parameters,
    Ports.committedRunning parameters,
    Ports.publicDigest parameters]

/-- Exact terminal input order: public statement followed by committed proof.
The fixed-one program counter is verifier-static and absent. -/
def terminalInputSchema (parameters : Parameters) :
    Schema (typeSystem parameters) :=
  [Ports.publicNat parameters,
    Ports.publicState parameters,
    Ports.publicState parameters,
    Ports.committedRunning parameters,
    Ports.committedRunningWitness parameters,
    Ports.committedFresh parameters,
    Ports.committedFreshWitness parameters]

/-- Materialize the paper advice in the exact step input schema. -/
def stepInputValues
    (parameters : Parameters)
    (input :
      Step.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof) :
    Schema.Values (typeSystem parameters) (stepInputSchema parameters) :=
  .cons input.iteration
    (.cons input.z0
      (.cons input.zi
        (.cons (input.running Step.selected)
          (.cons input.fresh
            (.cons input.witness
              (.cons input.nifsProof .nil))))))

/-- Materialize a paper result in the exact exposed step result schema. -/
def stepResultValues
    (parameters : Parameters)
    (output :
      Output parameters.Digest parameters.State parameters.Running 1) :
    Schema.Values (typeSystem parameters) (stepResultSchema parameters) :=
  .cons output.zNext
    (.cons (output.runningNext Step.selected)
      (.cons output.x .nil))

/-- Materialize the public terminal statement and committed proof in the
exact terminal input schema. -/
def terminalInputValues
    (parameters : Parameters)
    (statement : TerminalStatement parameters.State)
    (proof :
      Terminal.Proof
        parameters.Running parameters.RunningWitness parameters.Fresh
        parameters.FreshWitness) :
    Schema.Values (typeSystem parameters) (terminalInputSchema parameters) :=
  .cons statement.iteration
    (.cons statement.z0
      (.cons statement.zi
        (.cons proof.running
          (.cons proof.runningWitness
            (.cons proof.fresh
              (.cons proof.freshWitness .nil))))))

/-- Closed call vocabulary.  `hashPrior` and `hashNext` have the same
semantics but different ownership: the former is an auxiliary link value and
the latter is the sole public step output. -/
inductive Call where
  | iterationZero
  | stateEqual
  | step
  | hashPrior
  | hashNext
  | freshPublic
  | encodeInstance
  | encodedEqual
  | nifsVerify
  | runningCheck
  | freshCheck
deriving DecidableEq, Repr

def callInputs (parameters : Parameters) :
    Call -> List (typeSystem parameters).Kind
  | .iterationZero => [.data .nat]
  | .stateEqual => [.data .state, .data .state]
  | .step => [.data .state, .data .witness]
  | .hashPrior =>
      [.data .nat, .data .state, .data .state, .data .running]
  | .hashNext =>
      [.data .nat, .data .state, .data .state, .data .running]
  | .freshPublic => [.data .fresh]
  | .encodeInstance => [.data .digest]
  | .encodedEqual => [.data .encoded, .data .encoded]
  | .nifsVerify => [.data .running, .data .fresh, .data .nifsProof]
  | .runningCheck => [.data .running, .data .runningWitness]
  | .freshCheck => [.data .fresh, .data .freshWitness]

def callOutputs (parameters : Parameters) :
    Call -> Schema (typeSystem parameters)
  | .iterationZero => [Ports.auxiliaryBit parameters]
  | .stateEqual => [Ports.auxiliaryBit parameters]
  | .step => [Ports.committedState parameters]
  | .hashPrior => [Ports.auxiliaryDigest parameters]
  | .hashNext => [Ports.publicDigest parameters]
  | .freshPublic => [Ports.auxiliaryEncoded parameters]
  | .encodeInstance => [Ports.auxiliaryEncoded parameters]
  | .encodedEqual => [Ports.auxiliaryBit parameters]
  | .nifsVerify => [Ports.committedRunning parameters]
  | .runningCheck => [Ports.auxiliaryBit parameters]
  | .freshCheck => [Ports.auxiliaryBit parameters]

def stateEqual
    (parameters : Parameters)
    (left right : parameters.State) : Bool :=
  @decide (left = right) (parameters.stateDecidableEq left right)

def encodedEqual
    (parameters : Parameters)
    (left right : parameters.Encoded) : Bool :=
  @decide (left = right) (parameters.encodedDecidableEq left right)

/-- Deterministic partial semantics of every selected call.

Only `nifsVerify` is partial: its `none` result is the paper verifier's
rejection.  No call delegates to either whole fixed-one verifier. -/
def callEval (parameters : Parameters) :
    (call : Call) ->
    HVec (typeSystem parameters).Value (callInputs parameters call) ->
    Option
      (Schema.Values (typeSystem parameters) (callOutputs parameters call))
  | .iterationZero, .cons iteration .nil =>
      some (.cons (decide ((show Nat from iteration) = 0)) .nil)
  | .stateEqual, .cons left (.cons right .nil) =>
      some (.cons (stateEqual parameters left right) .nil)
  | .step, .cons state (.cons witness .nil) =>
      some (.cons
        (parameters.machine.step Step.selected state witness) .nil)
  | .hashPrior,
      .cons iteration (.cons z0 (.cons current (.cons running .nil))) =>
      some (.cons (parameters.machine.hash {
        verifierKeys := parameters.setup.verifierKeys
        iteration := iteration
        z0 := z0
        current := current
        running := fun _ => running
        pc := oneBased Step.selected
      }) .nil)
  | .hashNext,
      .cons iteration (.cons z0 (.cons current (.cons running .nil))) =>
      some (.cons (parameters.machine.hash {
        verifierKeys := parameters.setup.verifierKeys
        iteration := (show Nat from iteration) + 1
        z0 := z0
        current := current
        running := fun _ => running
        pc := oneBased Step.selected
      }) .nil)
  | .freshPublic, .cons fresh .nil =>
      some (.cons (parameters.machine.freshPublic fresh) .nil)
  | .encodeInstance, .cons digest .nil =>
      some (.cons (parameters.machine.encodeInstance digest) .nil)
  | .encodedEqual, .cons left (.cons right .nil) =>
      some (.cons (encodedEqual parameters left right) .nil)
  | .nifsVerify,
      .cons running (.cons fresh (.cons proof .nil)) =>
      match parameters.setup.nifs.verify
          (parameters.setup.verifierKeys Step.selected)
          running fresh proof with
      | none => none
      | some folded => some (.cons folded .nil)
  | .runningCheck, .cons running (.cons witness .nil) =>
      some (.cons
        (parameters.terminalChecks.runningCheck Step.selected
          (parameters.setup.verifierKeys Step.selected) running witness)
        .nil)
  | .freshCheck, .cons fresh (.cons witness .nil) =>
      some (.cons
        (parameters.terminalChecks.freshCheck Step.selected
          (parameters.setup.verifierKeys Step.selected) fresh witness)
        .nil)

def callFootprint (parameters : Parameters) : Call -> CallFootprint
  | .iterationZero => parameters.footprints.iterationZero
  | .stateEqual => parameters.footprints.stateEqual
  | .step => parameters.footprints.step
  | .hashPrior => parameters.footprints.hash
  | .hashNext => parameters.footprints.hash
  | .freshPublic => parameters.footprints.freshPublic
  | .encodeInstance => parameters.footprints.encodeInstance
  | .encodedEqual => parameters.footprints.encodedEqual
  | .nifsVerify => parameters.footprints.nifsVerify
  | .runningCheck => parameters.footprints.runningCheck
  | .freshCheck => parameters.footprints.freshCheck

/-- Closed artifact-independent signature used by fixed-one step and terminal
programs. -/
def signature (parameters : Parameters) : Signature where
  types := typeSystem parameters
  Call := Call
  callInputs := callInputs parameters
  callOutputs := callOutputs parameters
  callEval := callEval parameters
  callFootprint := callFootprint parameters

/-- Verifier-owned base running value.  A lowering program introduces it with
`Primitive.literal` at the committed running port; it is not prover advice. -/
def defaultRunning (parameters : Parameters) : parameters.Running :=
  parameters.setup.defaultRunning

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
