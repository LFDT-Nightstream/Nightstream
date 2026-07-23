import Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

/-!
Contract: artifact-independent typed lowering program for the payload-minimal
fixed-one HyperNova terminal verifier.

Owns:
- the exact seven-port public-statement/committed-proof input;
- intrinsic base/recursive control flow selected by the iteration-zero call;
- the base endpoint assertion;
- the recursive prior-link and two terminal-relation assertions;
- equivalence with the independently defined fixed-one terminal verifier and
  paper transition.

Does not own: NIFS verification, a whole-verifier opaque call, an external
branch selector, physical rows or columns, Rust behavior, or an encoding
recipe.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

namespace FixedOneTerminal

abbrev Proof (parameters : Parameters) :=
  Vocabulary.Terminal.Proof
    parameters.Running parameters.RunningWitness parameters.Fresh
    parameters.FreshWitness

end FixedOneTerminal

/-- The terminal verifier has no exposed result payload: successful execution
is witnessed only by reaching the empty result schema. -/
def resultSchema (parameters : Parameters) :
    Schema (typeSystem parameters) :=
  []

namespace InputRef

def iteration (parameters : Parameters) :
    Ref (typeSystem parameters) (terminalInputSchema parameters) (.data .nat) :=
  .here (Ports.publicNat parameters)

def z0 (parameters : Parameters) :
    Ref (typeSystem parameters) (terminalInputSchema parameters) (.data .state) :=
  .there (.here (Ports.publicState parameters))

def zi (parameters : Parameters) :
    Ref (typeSystem parameters) (terminalInputSchema parameters) (.data .state) :=
  .there (.there (.here (Ports.publicState parameters)))

def running (parameters : Parameters) :
    Ref (typeSystem parameters) (terminalInputSchema parameters) (.data .running) :=
  .there (.there (.there (.here (Ports.committedRunning parameters))))

def runningWitness (parameters : Parameters) :
    Ref (typeSystem parameters) (terminalInputSchema parameters)
      (.data .runningWitness) :=
  .there (.there (.there (.there
    (.here (Ports.committedRunningWitness parameters)))))

def fresh (parameters : Parameters) :
    Ref (typeSystem parameters) (terminalInputSchema parameters) (.data .fresh) :=
  .there (.there (.there (.there (.there
    (.here (Ports.committedFresh parameters))))))

def freshWitness (parameters : Parameters) :
    Ref (typeSystem parameters) (terminalInputSchema parameters)
      (.data .freshWitness) :=
  .there (.there (.there (.there (.there (.there
    (.here (Ports.committedFreshWitness parameters)))))))

end InputRef

/-- Lift a typed reference across a statically known prefix. -/
def liftRef
    {types : TypeSystem}
    {context : Schema types}
    {kind : types.Kind} :
    (added : Schema types) ->
    Ref types context kind ->
    Ref types (added ++ context) kind
  | [], reference => reference
  | _ :: tail, reference => .there (liftRef tail reference)

/-- Context after computing whether the public iteration is zero. -/
def branchInputSchema (parameters : Parameters) :
    Schema (typeSystem parameters) :=
  Ports.auxiliaryBit parameters :: terminalInputSchema parameters

namespace BranchRef

def iterationZero (parameters : Parameters) :
    Ref (typeSystem parameters) (branchInputSchema parameters) .bit :=
  .here (Ports.auxiliaryBit parameters)

def iteration (parameters : Parameters) :
    Ref (typeSystem parameters) (branchInputSchema parameters) (.data .nat) :=
  .there (InputRef.iteration parameters)

def z0 (parameters : Parameters) :
    Ref (typeSystem parameters) (branchInputSchema parameters) (.data .state) :=
  .there (InputRef.z0 parameters)

def zi (parameters : Parameters) :
    Ref (typeSystem parameters) (branchInputSchema parameters) (.data .state) :=
  .there (InputRef.zi parameters)

def running (parameters : Parameters) :
    Ref (typeSystem parameters) (branchInputSchema parameters) (.data .running) :=
  .there (InputRef.running parameters)

def runningWitness (parameters : Parameters) :
    Ref (typeSystem parameters) (branchInputSchema parameters)
      (.data .runningWitness) :=
  .there (InputRef.runningWitness parameters)

def fresh (parameters : Parameters) :
    Ref (typeSystem parameters) (branchInputSchema parameters) (.data .fresh) :=
  .there (InputRef.fresh parameters)

def freshWitness (parameters : Parameters) :
    Ref (typeSystem parameters) (branchInputSchema parameters)
      (.data .freshWitness) :=
  .there (InputRef.freshWitness parameters)

end BranchRef

def afterBaseEqualitySchema (parameters : Parameters) :
    Schema (typeSystem parameters) :=
  Ports.auxiliaryBit parameters :: branchInputSchema parameters

def baseStateEqualCall (parameters : Parameters) :
    Primitive (signature parameters) (branchInputSchema parameters)
      (afterBaseEqualitySchema parameters) :=
  Primitive.invoke (signature := signature parameters)
    Vocabulary.Call.stateEqual
    (.cons (BranchRef.zi parameters)
      (.cons (BranchRef.z0 parameters) .nil))

/-- Iteration-zero arm: assert only the paper endpoint `zi = z0`. -/
def baseArm (parameters : Parameters) :
    Block (signature parameters) (branchInputSchema parameters)
      (resultSchema parameters) :=
  .step
    (baseStateEqualCall parameters)
    (.step
      (Primitive.assertTrue (signature := signature parameters)
        (.here (Ports.auxiliaryBit parameters) :
          Ref (typeSystem parameters) (afterBaseEqualitySchema parameters) .bit))
      (.yield .nil))

def afterHashSchema (parameters : Parameters) :
    Schema (typeSystem parameters) :=
  Ports.auxiliaryDigest parameters :: branchInputSchema parameters

def afterFreshPublicSchema (parameters : Parameters) :
    Schema (typeSystem parameters) :=
  Ports.auxiliaryEncoded parameters :: afterHashSchema parameters

def afterEncodeSchema (parameters : Parameters) :
    Schema (typeSystem parameters) :=
  Ports.auxiliaryEncoded parameters :: afterFreshPublicSchema parameters

def afterEncodedEqualitySchema (parameters : Parameters) :
    Schema (typeSystem parameters) :=
  Ports.auxiliaryBit parameters :: afterEncodeSchema parameters

def afterRunningCheckSchema (parameters : Parameters) :
    Schema (typeSystem parameters) :=
  Ports.auxiliaryBit parameters :: afterEncodedEqualitySchema parameters

def afterFreshCheckSchema (parameters : Parameters) :
    Schema (typeSystem parameters) :=
  Ports.auxiliaryBit parameters :: afterRunningCheckSchema parameters

namespace RecursiveRef

def freshAfterHash (parameters : Parameters) :
    Ref (typeSystem parameters) (afterHashSchema parameters) (.data .fresh) :=
  .there (BranchRef.fresh parameters)

def hashAfterFreshPublic (parameters : Parameters) :
    Ref (typeSystem parameters) (afterFreshPublicSchema parameters)
      (.data .digest) :=
  .there (.here (Ports.auxiliaryDigest parameters))

def encoded (parameters : Parameters) :
    Ref (typeSystem parameters) (afterEncodeSchema parameters) (.data .encoded) :=
  .here (Ports.auxiliaryEncoded parameters)

def freshPublic (parameters : Parameters) :
    Ref (typeSystem parameters) (afterEncodeSchema parameters) (.data .encoded) :=
  .there (.here (Ports.auxiliaryEncoded parameters))

def encodedEqual (parameters : Parameters) :
    Ref (typeSystem parameters) (afterEncodedEqualitySchema parameters) .bit :=
  .here (Ports.auxiliaryBit parameters)

def running (parameters : Parameters) :
    Ref (typeSystem parameters) (afterEncodedEqualitySchema parameters)
      (.data .running) :=
  liftRef
    [Ports.auxiliaryBit parameters,
      Ports.auxiliaryEncoded parameters,
      Ports.auxiliaryEncoded parameters,
      Ports.auxiliaryDigest parameters]
    (BranchRef.running parameters)

def runningWitness (parameters : Parameters) :
    Ref (typeSystem parameters) (afterEncodedEqualitySchema parameters)
      (.data .runningWitness) :=
  liftRef
    [Ports.auxiliaryBit parameters,
      Ports.auxiliaryEncoded parameters,
      Ports.auxiliaryEncoded parameters,
      Ports.auxiliaryDigest parameters]
    (BranchRef.runningWitness parameters)

def runningAccepted (parameters : Parameters) :
    Ref (typeSystem parameters) (afterRunningCheckSchema parameters) .bit :=
  .here (Ports.auxiliaryBit parameters)

def fresh (parameters : Parameters) :
    Ref (typeSystem parameters) (afterRunningCheckSchema parameters)
      (.data .fresh) :=
  liftRef
    [Ports.auxiliaryBit parameters,
      Ports.auxiliaryBit parameters,
      Ports.auxiliaryEncoded parameters,
      Ports.auxiliaryEncoded parameters,
      Ports.auxiliaryDigest parameters]
    (BranchRef.fresh parameters)

def freshWitness (parameters : Parameters) :
    Ref (typeSystem parameters) (afterRunningCheckSchema parameters)
      (.data .freshWitness) :=
  liftRef
    [Ports.auxiliaryBit parameters,
      Ports.auxiliaryBit parameters,
      Ports.auxiliaryEncoded parameters,
      Ports.auxiliaryEncoded parameters,
      Ports.auxiliaryDigest parameters]
    (BranchRef.freshWitness parameters)

def freshAccepted (parameters : Parameters) :
    Ref (typeSystem parameters) (afterFreshCheckSchema parameters) .bit :=
  .here (Ports.auxiliaryBit parameters)

end RecursiveRef

def hashPriorCall (parameters : Parameters) :
    Primitive (signature parameters) (branchInputSchema parameters)
      (afterHashSchema parameters) :=
  Primitive.invoke (signature := signature parameters)
    Vocabulary.Call.hashPrior
    (.cons (BranchRef.iteration parameters)
      (.cons (BranchRef.z0 parameters)
        (.cons (BranchRef.zi parameters)
          (.cons (BranchRef.running parameters) .nil))))

def freshPublicCall (parameters : Parameters) :
    Primitive (signature parameters) (afterHashSchema parameters)
      (afterFreshPublicSchema parameters) :=
  Primitive.invoke (signature := signature parameters)
    Vocabulary.Call.freshPublic
    (.cons (RecursiveRef.freshAfterHash parameters) .nil)

def encodeInstanceCall (parameters : Parameters) :
    Primitive (signature parameters) (afterFreshPublicSchema parameters)
      (afterEncodeSchema parameters) :=
  Primitive.invoke (signature := signature parameters)
    Vocabulary.Call.encodeInstance
    (.cons (RecursiveRef.hashAfterFreshPublic parameters) .nil)

def encodedEqualCall (parameters : Parameters) :
    Primitive (signature parameters) (afterEncodeSchema parameters)
      (afterEncodedEqualitySchema parameters) :=
  Primitive.invoke (signature := signature parameters)
    Vocabulary.Call.encodedEqual
    (.cons (RecursiveRef.freshPublic parameters)
      (.cons (RecursiveRef.encoded parameters) .nil))

def runningCheckCall (parameters : Parameters) :
    Primitive (signature parameters) (afterEncodedEqualitySchema parameters)
      (afterRunningCheckSchema parameters) :=
  Primitive.invoke (signature := signature parameters)
    Vocabulary.Call.runningCheck
    (.cons (RecursiveRef.running parameters)
      (.cons (RecursiveRef.runningWitness parameters) .nil))

def freshCheckCall (parameters : Parameters) :
    Primitive (signature parameters) (afterRunningCheckSchema parameters)
      (afterFreshCheckSchema parameters) :=
  Primitive.invoke (signature := signature parameters)
    Vocabulary.Call.freshCheck
    (.cons (RecursiveRef.fresh parameters)
      (.cons (RecursiveRef.freshWitness parameters) .nil))

/-- Positive-iteration arm.  The prior hash is built directly from the public
statement and authoritative running proof value.  The arm contains no NIFS
call; it checks only the prior public link and the two paper terminal
relations. -/
def recursiveArm (parameters : Parameters) :
    Block (signature parameters) (branchInputSchema parameters)
      (resultSchema parameters) :=
  .step
    (hashPriorCall parameters)
    (.step
      (freshPublicCall parameters)
      (.step
        (encodeInstanceCall parameters)
        (.step
          (encodedEqualCall parameters)
          (.step
            (Primitive.assertTrue (signature := signature parameters)
              (RecursiveRef.encodedEqual parameters))
            (.step
              (runningCheckCall parameters)
              (.step
                (Primitive.assertTrue (signature := signature parameters)
                  (RecursiveRef.runningAccepted parameters))
                (.step
                  (freshCheckCall parameters)
                  (.step
                    (Primitive.assertTrue (signature := signature parameters)
                      (RecursiveRef.freshAccepted parameters))
                    (.yield .nil)))))))))

/-- Continuation after either private arm.  Both arms export the same empty
schema, so no arm-local witness can escape the join. -/
def continuation (parameters : Parameters) :
    Block (signature parameters)
      (resultSchema parameters ++ branchInputSchema parameters)
      (resultSchema parameters) :=
  .yield .nil

def iterationZeroCall (parameters : Parameters) :
    Primitive (signature parameters) (terminalInputSchema parameters)
      (branchInputSchema parameters) :=
  Primitive.invoke (signature := signature parameters)
    Vocabulary.Call.iterationZero
    (.cons (InputRef.iteration parameters) .nil)

def branchBlock (parameters : Parameters) :
    Block (signature parameters) (branchInputSchema parameters)
      (resultSchema parameters) :=
  .branch
    (BranchRef.iterationZero parameters)
    (baseArm parameters)
    (recursiveArm parameters)
    (continuation parameters)

/-- Exact intrinsic lowering block: compute `iteration = 0`, then select the
base or recursive paper arm inside the typed program. -/
def block (parameters : Parameters) :
    Block (signature parameters) (terminalInputSchema parameters)
      (resultSchema parameters) :=
  .step
    (iterationZeroCall parameters)
    (branchBlock parameters)

/-- Artifact-independent fixed-one terminal lowering program. -/
def program (parameters : Parameters) :
    Program (signature parameters) (terminalInputSchema parameters)
      (resultSchema parameters) :=
  ⟨block parameters⟩

/-- Exact runtime value at the intrinsic branch boundary. -/
def branchValues
    (parameters : Parameters)
    (iterationIsZero : Bool)
    (statement : TerminalStatement parameters.State)
    (proof : FixedOneTerminal.Proof parameters) :
    Schema.Values (typeSystem parameters) (branchInputSchema parameters) :=
  .cons iterationIsZero (terminalInputValues parameters statement proof)

def priorDigest
    (parameters : Parameters)
    (statement : TerminalStatement parameters.State)
    (proof : FixedOneTerminal.Proof parameters) : parameters.Digest :=
  parameters.machine.hash {
    verifierKeys := parameters.setup.verifierKeys
    iteration := statement.iteration
    z0 := statement.z0
    current := statement.zi
    running := fun _ => proof.running
    pc := oneBased Vocabulary.Step.selected
  }

def afterBaseEqualityValues
    (parameters : Parameters)
    (iterationIsZero : Bool)
    (statement : TerminalStatement parameters.State)
    (proof : FixedOneTerminal.Proof parameters) :
    Schema.Values (typeSystem parameters) (afterBaseEqualitySchema parameters) :=
  .cons (stateEqual parameters statement.zi statement.z0)
    (branchValues parameters iterationIsZero statement proof)

def afterHashValues
    (parameters : Parameters)
    (iterationIsZero : Bool)
    (statement : TerminalStatement parameters.State)
    (proof : FixedOneTerminal.Proof parameters) :
    Schema.Values (typeSystem parameters) (afterHashSchema parameters) :=
  .cons (priorDigest parameters statement proof)
    (branchValues parameters iterationIsZero statement proof)

def afterFreshPublicValues
    (parameters : Parameters)
    (iterationIsZero : Bool)
    (statement : TerminalStatement parameters.State)
    (proof : FixedOneTerminal.Proof parameters) :
    Schema.Values (typeSystem parameters) (afterFreshPublicSchema parameters) :=
  .cons (parameters.machine.freshPublic proof.fresh)
    (afterHashValues parameters iterationIsZero statement proof)

def afterEncodeValues
    (parameters : Parameters)
    (iterationIsZero : Bool)
    (statement : TerminalStatement parameters.State)
    (proof : FixedOneTerminal.Proof parameters) :
    Schema.Values (typeSystem parameters) (afterEncodeSchema parameters) :=
  .cons (parameters.machine.encodeInstance
      (priorDigest parameters statement proof))
    (afterFreshPublicValues parameters iterationIsZero statement proof)

def priorLinkAccepted
    (parameters : Parameters)
    (statement : TerminalStatement parameters.State)
    (proof : FixedOneTerminal.Proof parameters) : Bool :=
  encodedEqual parameters
    (parameters.machine.freshPublic proof.fresh)
    (parameters.machine.encodeInstance
      (priorDigest parameters statement proof))

def afterEncodedEqualityValues
    (parameters : Parameters)
    (iterationIsZero : Bool)
    (statement : TerminalStatement parameters.State)
    (proof : FixedOneTerminal.Proof parameters) :
    Schema.Values (typeSystem parameters)
      (afterEncodedEqualitySchema parameters) :=
  .cons (priorLinkAccepted parameters statement proof)
    (afterEncodeValues parameters iterationIsZero statement proof)

def runningAcceptedValue
    (parameters : Parameters)
    (proof : FixedOneTerminal.Proof parameters) : Bool :=
  parameters.terminalChecks.runningCheck Vocabulary.Step.selected
    (parameters.setup.verifierKeys Vocabulary.Step.selected)
    proof.running proof.runningWitness

def afterRunningCheckValues
    (parameters : Parameters)
    (iterationIsZero : Bool)
    (statement : TerminalStatement parameters.State)
    (proof : FixedOneTerminal.Proof parameters) :
    Schema.Values (typeSystem parameters) (afterRunningCheckSchema parameters) :=
  .cons (runningAcceptedValue parameters proof)
    (afterEncodedEqualityValues parameters iterationIsZero statement proof)

def freshAcceptedValue
    (parameters : Parameters)
    (proof : FixedOneTerminal.Proof parameters) : Bool :=
  parameters.terminalChecks.freshCheck Vocabulary.Step.selected
    (parameters.setup.verifierKeys Vocabulary.Step.selected)
    proof.fresh proof.freshWitness

def afterFreshCheckValues
    (parameters : Parameters)
    (iterationIsZero : Bool)
    (statement : TerminalStatement parameters.State)
    (proof : FixedOneTerminal.Proof parameters) :
    Schema.Values (typeSystem parameters) (afterFreshCheckSchema parameters) :=
  .cons (freshAcceptedValue parameters proof)
    (afterRunningCheckValues parameters iterationIsZero statement proof)

def baseReferenceExec
    (parameters : Parameters)
    (statement : TerminalStatement parameters.State) :
    Option (Schema.Values (typeSystem parameters) (resultSchema parameters)) :=
  if stateEqual parameters statement.zi statement.z0 then some .nil else none

def recursiveReferenceExec
    (parameters : Parameters)
    (statement : TerminalStatement parameters.State)
    (proof : FixedOneTerminal.Proof parameters) :
    Option (Schema.Values (typeSystem parameters) (resultSchema parameters)) :=
  if priorLinkAccepted parameters statement proof then
    if runningAcceptedValue parameters proof then
      if freshAcceptedValue parameters proof then some .nil else none
    else
      none
  else
    none

def branchReferenceExec
    (parameters : Parameters)
    (iterationIsZero : Bool)
    (statement : TerminalStatement parameters.State)
    (proof : FixedOneTerminal.Proof parameters) :
    Option (Schema.Values (typeSystem parameters) (resultSchema parameters)) :=
  match iterationIsZero with
  | true => baseReferenceExec parameters statement
  | false => recursiveReferenceExec parameters statement proof

def referenceExec
    (parameters : Parameters)
    (statement : TerminalStatement parameters.State)
    (proof : FixedOneTerminal.Proof parameters) :
    Option (Schema.Values (typeSystem parameters) (resultSchema parameters)) :=
  branchReferenceExec parameters (decide (statement.iteration = 0))
    statement proof

/-- The first call constructs the exact branch input and nothing else. -/
theorem iterationZeroCall_exec
    (parameters : Parameters)
    (statement : TerminalStatement parameters.State)
    (proof : FixedOneTerminal.Proof parameters) :
    (iterationZeroCall parameters).exec
        (terminalInputValues parameters statement proof) =
      some (branchValues parameters (decide (statement.iteration = 0))
        statement proof) := by
  rfl

theorem baseStateEqualCall_exec
    (parameters : Parameters)
    (iterationIsZero : Bool)
    (statement : TerminalStatement parameters.State)
    (proof : FixedOneTerminal.Proof parameters) :
    (baseStateEqualCall parameters).exec
        (branchValues parameters iterationIsZero statement proof) =
      some (afterBaseEqualityValues parameters iterationIsZero statement proof) := by
  rfl

theorem hashPriorCall_exec
    (parameters : Parameters)
    (iterationIsZero : Bool)
    (statement : TerminalStatement parameters.State)
    (proof : FixedOneTerminal.Proof parameters) :
    (hashPriorCall parameters).exec
        (branchValues parameters iterationIsZero statement proof) =
      some (afterHashValues parameters iterationIsZero statement proof) := by
  rfl

theorem freshPublicCall_exec
    (parameters : Parameters)
    (iterationIsZero : Bool)
    (statement : TerminalStatement parameters.State)
    (proof : FixedOneTerminal.Proof parameters) :
    (freshPublicCall parameters).exec
        (afterHashValues parameters iterationIsZero statement proof) =
      some
        (afterFreshPublicValues parameters iterationIsZero statement proof) := by
  rfl

theorem encodeInstanceCall_exec
    (parameters : Parameters)
    (iterationIsZero : Bool)
    (statement : TerminalStatement parameters.State)
    (proof : FixedOneTerminal.Proof parameters) :
    (encodeInstanceCall parameters).exec
        (afterFreshPublicValues parameters iterationIsZero statement proof) =
      some (afterEncodeValues parameters iterationIsZero statement proof) := by
  rfl

theorem encodedEqualCall_exec
    (parameters : Parameters)
    (iterationIsZero : Bool)
    (statement : TerminalStatement parameters.State)
    (proof : FixedOneTerminal.Proof parameters) :
    (encodedEqualCall parameters).exec
        (afterEncodeValues parameters iterationIsZero statement proof) =
      some
        (afterEncodedEqualityValues parameters iterationIsZero statement proof) := by
  rfl

theorem runningCheckCall_exec
    (parameters : Parameters)
    (iterationIsZero : Bool)
    (statement : TerminalStatement parameters.State)
    (proof : FixedOneTerminal.Proof parameters) :
    (runningCheckCall parameters).exec
        (afterEncodedEqualityValues parameters iterationIsZero statement proof) =
      some
        (afterRunningCheckValues parameters iterationIsZero statement proof) := by
  rfl

theorem freshCheckCall_exec
    (parameters : Parameters)
    (iterationIsZero : Bool)
    (statement : TerminalStatement parameters.State)
    (proof : FixedOneTerminal.Proof parameters) :
    (freshCheckCall parameters).exec
        (afterRunningCheckValues parameters iterationIsZero statement proof) =
      some (afterFreshCheckValues parameters iterationIsZero statement proof) := by
  rfl

theorem baseAssertion_exec
    (parameters : Parameters)
    (iterationIsZero : Bool)
    (statement : TerminalStatement parameters.State)
    (proof : FixedOneTerminal.Proof parameters) :
    (Primitive.assertTrue (signature := signature parameters)
        (.here (Ports.auxiliaryBit parameters) :
          Ref (typeSystem parameters)
            (afterBaseEqualitySchema parameters) .bit)).exec
        (afterBaseEqualityValues parameters iterationIsZero statement proof) =
      if stateEqual parameters statement.zi statement.z0 then
        some
          (afterBaseEqualityValues parameters iterationIsZero statement proof)
      else
        none := by
  change
    (match stateEqual parameters statement.zi statement.z0 with
      | false => none
      | true =>
          some
            (afterBaseEqualityValues
              parameters iterationIsZero statement proof)) =
      if stateEqual parameters statement.zi statement.z0 then
        some
          (afterBaseEqualityValues parameters iterationIsZero statement proof)
      else
        none
  cases stateEqual parameters statement.zi statement.z0 <;> rfl

theorem encodedAssertion_exec
    (parameters : Parameters)
    (iterationIsZero : Bool)
    (statement : TerminalStatement parameters.State)
    (proof : FixedOneTerminal.Proof parameters) :
    (Primitive.assertTrue (signature := signature parameters)
        (RecursiveRef.encodedEqual parameters)).exec
        (afterEncodedEqualityValues
          parameters iterationIsZero statement proof) =
      if priorLinkAccepted parameters statement proof then
        some
          (afterEncodedEqualityValues
            parameters iterationIsZero statement proof)
      else
        none := by
  change
    (match priorLinkAccepted parameters statement proof with
      | false => none
      | true =>
          some
            (afterEncodedEqualityValues
              parameters iterationIsZero statement proof)) =
      if priorLinkAccepted parameters statement proof then
        some
          (afterEncodedEqualityValues
            parameters iterationIsZero statement proof)
      else
        none
  cases priorLinkAccepted parameters statement proof <;> rfl

theorem runningAssertion_exec
    (parameters : Parameters)
    (iterationIsZero : Bool)
    (statement : TerminalStatement parameters.State)
    (proof : FixedOneTerminal.Proof parameters) :
    (Primitive.assertTrue (signature := signature parameters)
        (RecursiveRef.runningAccepted parameters)).exec
        (afterRunningCheckValues parameters iterationIsZero statement proof) =
      if runningAcceptedValue parameters proof then
        some
          (afterRunningCheckValues parameters iterationIsZero statement proof)
      else
        none := by
  change
    (match runningAcceptedValue parameters proof with
      | false => none
      | true =>
          some
            (afterRunningCheckValues
              parameters iterationIsZero statement proof)) =
      if runningAcceptedValue parameters proof then
        some
          (afterRunningCheckValues parameters iterationIsZero statement proof)
      else
        none
  cases runningAcceptedValue parameters proof <;> rfl

theorem freshAssertion_exec
    (parameters : Parameters)
    (iterationIsZero : Bool)
    (statement : TerminalStatement parameters.State)
    (proof : FixedOneTerminal.Proof parameters) :
    (Primitive.assertTrue (signature := signature parameters)
        (RecursiveRef.freshAccepted parameters)).exec
        (afterFreshCheckValues parameters iterationIsZero statement proof) =
      if freshAcceptedValue parameters proof then
        some
          (afterFreshCheckValues parameters iterationIsZero statement proof)
      else
        none := by
  change
    (match freshAcceptedValue parameters proof with
      | false => none
      | true =>
          some
            (afterFreshCheckValues
              parameters iterationIsZero statement proof)) =
      if freshAcceptedValue parameters proof then
        some
          (afterFreshCheckValues parameters iterationIsZero statement proof)
      else
        none
  cases freshAcceptedValue parameters proof <;> rfl

theorem baseArm_exec
    (parameters : Parameters)
    (iterationIsZero : Bool)
    (statement : TerminalStatement parameters.State)
    (proof : FixedOneTerminal.Proof parameters) :
    (baseArm parameters).exec
        (branchValues parameters iterationIsZero statement proof) =
      baseReferenceExec parameters statement := by
  unfold baseArm
  rw [Block.step_exec_of_eq_some
    (baseStateEqualCall_exec parameters iterationIsZero statement proof)]
  let finish :
      Block (signature parameters) (afterBaseEqualitySchema parameters)
        (resultSchema parameters) := .yield .nil
  change
    (Block.step
      (Primitive.assertTrue (signature := signature parameters)
        (.here (Ports.auxiliaryBit parameters) :
          Ref (typeSystem parameters)
            (afterBaseEqualitySchema parameters) .bit))
      finish).exec
        (afterBaseEqualityValues parameters iterationIsZero statement proof) =
      baseReferenceExec parameters statement
  cases accepted : stateEqual parameters statement.zi statement.z0 with
  | false =>
      have assertionFailed :
          (Primitive.assertTrue (signature := signature parameters)
              (.here (Ports.auxiliaryBit parameters) :
                Ref (typeSystem parameters)
                  (afterBaseEqualitySchema parameters) .bit)).exec
              (afterBaseEqualityValues
                parameters iterationIsZero statement proof) = none := by
        simpa [accepted] using
          (baseAssertion_exec parameters iterationIsZero statement proof)
      calc
        _ = none :=
          Block.step_exec_of_eq_none (rest := finish) assertionFailed
        _ = baseReferenceExec parameters statement := by
          simp [baseReferenceExec, accepted]
          rfl
  | true =>
      have assertionPassed :
          (Primitive.assertTrue (signature := signature parameters)
              (.here (Ports.auxiliaryBit parameters) :
                Ref (typeSystem parameters)
                  (afterBaseEqualitySchema parameters) .bit)).exec
              (afterBaseEqualityValues
                parameters iterationIsZero statement proof) =
            some (afterBaseEqualityValues
              parameters iterationIsZero statement proof) := by
        simpa [accepted] using
          (baseAssertion_exec parameters iterationIsZero statement proof)
      calc
        _ = finish.exec
              (afterBaseEqualityValues
                parameters iterationIsZero statement proof) :=
          Block.step_exec_of_eq_some (rest := finish) assertionPassed
        _ = baseReferenceExec parameters statement := by
          simp [finish, Block.exec, baseReferenceExec, accepted, Exports.get]
          rfl

theorem recursiveArm_exec
    (parameters : Parameters)
    (iterationIsZero : Bool)
    (statement : TerminalStatement parameters.State)
    (proof : FixedOneTerminal.Proof parameters) :
    (recursiveArm parameters).exec
        (branchValues parameters iterationIsZero statement proof) =
      recursiveReferenceExec parameters statement proof := by
  unfold recursiveArm
  rw [Block.step_exec_of_eq_some
    (hashPriorCall_exec parameters iterationIsZero statement proof)]
  rw [Block.step_exec_of_eq_some
    (freshPublicCall_exec parameters iterationIsZero statement proof)]
  rw [Block.step_exec_of_eq_some
    (encodeInstanceCall_exec parameters iterationIsZero statement proof)]
  rw [Block.step_exec_of_eq_some
    (encodedEqualCall_exec parameters iterationIsZero statement proof)]
  cases priorLink : priorLinkAccepted parameters statement proof
  · have assertionFailed :
        (Primitive.assertTrue (signature := signature parameters)
            (RecursiveRef.encodedEqual parameters)).exec
            (afterEncodedEqualityValues
              parameters iterationIsZero statement proof) = none := by
      simpa [priorLink] using
        (encodedAssertion_exec parameters iterationIsZero statement proof)
    rw [Block.step_exec_of_eq_none assertionFailed]
    simp [recursiveReferenceExec, priorLink]
    rfl
  · have assertionPassed :
        (Primitive.assertTrue (signature := signature parameters)
            (RecursiveRef.encodedEqual parameters)).exec
            (afterEncodedEqualityValues
              parameters iterationIsZero statement proof) =
          some (afterEncodedEqualityValues
            parameters iterationIsZero statement proof) := by
      simpa [priorLink] using
        (encodedAssertion_exec parameters iterationIsZero statement proof)
    rw [Block.step_exec_of_eq_some assertionPassed]
    rw [Block.step_exec_of_eq_some
      (runningCheckCall_exec parameters iterationIsZero statement proof)]
    cases runningAccepted : runningAcceptedValue parameters proof
    · have assertionFailed :
          (Primitive.assertTrue (signature := signature parameters)
              (RecursiveRef.runningAccepted parameters)).exec
              (afterRunningCheckValues
                parameters iterationIsZero statement proof) = none := by
        simpa [runningAccepted] using
          (runningAssertion_exec parameters iterationIsZero statement proof)
      rw [Block.step_exec_of_eq_none assertionFailed]
      simp [recursiveReferenceExec, priorLink, runningAccepted]
      rfl
    · have assertionPassed :
          (Primitive.assertTrue (signature := signature parameters)
              (RecursiveRef.runningAccepted parameters)).exec
              (afterRunningCheckValues
                parameters iterationIsZero statement proof) =
            some (afterRunningCheckValues
              parameters iterationIsZero statement proof) := by
        simpa [runningAccepted] using
          (runningAssertion_exec parameters iterationIsZero statement proof)
      rw [Block.step_exec_of_eq_some assertionPassed]
      rw [Block.step_exec_of_eq_some
        (freshCheckCall_exec parameters iterationIsZero statement proof)]
      cases freshAccepted : freshAcceptedValue parameters proof
      · have assertionFailed :
            (Primitive.assertTrue (signature := signature parameters)
                (RecursiveRef.freshAccepted parameters)).exec
                (afterFreshCheckValues
                  parameters iterationIsZero statement proof) = none := by
          simpa [freshAccepted] using
            (freshAssertion_exec parameters iterationIsZero statement proof)
        rw [Block.step_exec_of_eq_none assertionFailed]
        simp [recursiveReferenceExec, priorLink, runningAccepted,
          freshAccepted]
        rfl
      · have assertionPassed :
            (Primitive.assertTrue (signature := signature parameters)
                (RecursiveRef.freshAccepted parameters)).exec
                (afterFreshCheckValues
                  parameters iterationIsZero statement proof) =
              some (afterFreshCheckValues
                parameters iterationIsZero statement proof) := by
          simpa [freshAccepted] using
            (freshAssertion_exec parameters iterationIsZero statement proof)
        rw [Block.step_exec_of_eq_some assertionPassed]
        simp [Block.exec, recursiveReferenceExec, priorLink, runningAccepted,
          freshAccepted, Exports.get]
        rfl

/-- The join cannot change a successful empty result.  Pattern matching the
empty heterogeneous vector closes the otherwise hidden `some result =
some .nil` obligation. -/
theorem continuation_preserves
    (parameters : Parameters)
    (source :
      Schema.Values (typeSystem parameters) (branchInputSchema parameters))
    (candidate :
      Option (Schema.Values (typeSystem parameters) (resultSchema parameters))) :
    (match candidate with
      | none => none
      | some joined =>
          (continuation parameters).exec (joined.append source)) =
        candidate := by
  cases candidate with
  | none => rfl
  | some joined =>
      cases joined
      rfl

theorem branchBlock_exec
    (parameters : Parameters)
    (iterationIsZero : Bool)
    (statement : TerminalStatement parameters.State)
    (proof : FixedOneTerminal.Proof parameters) :
    (branchBlock parameters).exec
        (branchValues parameters iterationIsZero statement proof) =
      branchReferenceExec parameters iterationIsZero statement proof := by
  cases iterationIsZero with
  | false =>
      unfold branchBlock
      rw [Block.branch_exec_of_selector_false (by rfl)]
      rw [recursiveArm_exec]
      unfold branchReferenceExec
      cases executed : recursiveReferenceExec parameters statement proof with
      | none => rfl
      | some joined =>
          cases joined
          rfl
  | true =>
      unfold branchBlock
      rw [Block.branch_exec_of_selector_true (by rfl)]
      rw [baseArm_exec]
      unfold branchReferenceExec
      cases executed : baseReferenceExec parameters statement with
      | none => rfl
      | some joined =>
          cases joined
          rfl

theorem block_exec_eq_reference
    (parameters : Parameters)
    (statement : TerminalStatement parameters.State)
    (proof : FixedOneTerminal.Proof parameters) :
    (block parameters).exec
        (terminalInputValues parameters statement proof) =
      referenceExec parameters statement proof := by
  change
    (match (iterationZeroCall parameters).exec
        (terminalInputValues parameters statement proof) with
      | none => none
      | some values => (branchBlock parameters).exec values) =
        referenceExec parameters statement proof
  simp only [iterationZeroCall_exec]
  rw [branchBlock_exec]
  rfl

/-- The concrete typed program reduces to the compact reference execution. -/
theorem program_exec_eq_reference
    (parameters : Parameters)
    (statement : TerminalStatement parameters.State)
    (proof : FixedOneTerminal.Proof parameters) :
    (program parameters).exec
        (terminalInputValues parameters statement proof) =
      referenceExec parameters statement proof := by
  exact block_exec_eq_reference parameters statement proof

/-- Relational acceptance of the typed lowering program. -/
def Accepts
    (parameters : Parameters)
    (statement : TerminalStatement parameters.State)
    (proof : FixedOneTerminal.Proof parameters) : Prop :=
  (program parameters).Holds
    (terminalInputValues parameters statement proof) .nil

/-- The independently defined fixed-one executable acceptance predicate,
using the equality procedures fixed by this vocabulary. -/
def fixedOneAccepts
    (parameters : Parameters)
    (statement : TerminalStatement parameters.State)
    (proof : FixedOneTerminal.Proof parameters) : Prop :=
  letI : DecidableEq parameters.State := parameters.stateDecidableEq
  letI : DecidableEq parameters.Encoded := parameters.encodedDecidableEq
  Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.Accepts
    parameters.setup parameters.machine parameters.terminalRelations
      parameters.terminalChecks statement proof

/-- The step and terminal vocabularies name the unique fixed-one slot
independently; uniqueness, rather than a definitional coincidence, identifies
them. -/
theorem step_selected_eq_terminal_selected :
    Vocabulary.Step.selected =
      Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.selected :=
  Subsingleton.elim _ _

theorem priorDigest_eq_terminal
    (parameters : Parameters)
    (statement : TerminalStatement parameters.State)
    (proof : FixedOneTerminal.Proof parameters) :
    priorDigest parameters statement proof =
      parameters.machine.hash {
        verifierKeys := parameters.setup.verifierKeys
        iteration := statement.iteration
        z0 := statement.z0
        current := statement.zi
        running := fun _ => proof.running
        pc := 1
      } := by
  simp [priorDigest, step_selected_eq_terminal_selected,
    Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.oneBased_selected]

theorem runningAcceptedValue_eq_terminal
    (parameters : Parameters)
    (proof : FixedOneTerminal.Proof parameters) :
    runningAcceptedValue parameters proof =
      parameters.terminalChecks.runningCheck
        Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.selected
        (parameters.setup.verifierKeys
          Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.selected)
        proof.running proof.runningWitness := by
  simp only [runningAcceptedValue, step_selected_eq_terminal_selected]

theorem freshAcceptedValue_eq_terminal
    (parameters : Parameters)
    (proof : FixedOneTerminal.Proof parameters) :
    freshAcceptedValue parameters proof =
      parameters.terminalChecks.freshCheck
        Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.selected
        (parameters.setup.verifierKeys
          Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.selected)
        proof.fresh proof.freshWitness := by
  simp only [freshAcceptedValue, step_selected_eq_terminal_selected]

/-- The compact branch-shaped execution is exactly the independently defined
fixed-one terminal Boolean checker. -/
theorem referenceExec_eq_some_iff_fixedOne
    (parameters : Parameters)
    (statement : TerminalStatement parameters.State)
    (proof : FixedOneTerminal.Proof parameters) :
    referenceExec parameters statement proof = some .nil ↔
      fixedOneAccepts parameters statement proof := by
  letI : DecidableEq parameters.State := parameters.stateDecidableEq
  letI : DecidableEq parameters.Encoded := parameters.encodedDecidableEq
  unfold fixedOneAccepts
  by_cases iterationZero : statement.iteration = 0
  · by_cases endpoint : statement.zi = statement.z0
    <;> simp [referenceExec, branchReferenceExec, baseReferenceExec,
      stateEqual, iterationZero, endpoint,
      Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.Accepts,
      Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.eval]
  · by_cases priorLink :
      parameters.machine.freshPublic proof.fresh =
        parameters.machine.encodeInstance (parameters.machine.hash {
          verifierKeys := parameters.setup.verifierKeys
          iteration := statement.iteration
          z0 := statement.z0
          current := statement.zi
          running := fun _ => proof.running
          pc := 1
        })
    · by_cases runningAccepted :
        parameters.terminalChecks.runningCheck
            Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.selected
            (parameters.setup.verifierKeys
              Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.selected)
            proof.running proof.runningWitness = true
      · by_cases freshAccepted :
          parameters.terminalChecks.freshCheck
              Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.selected
              (parameters.setup.verifierKeys
                Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.selected)
              proof.fresh proof.freshWitness = true
        <;> simp [referenceExec, branchReferenceExec, recursiveReferenceExec,
          priorLinkAccepted, encodedEqual, iterationZero, priorLink,
          runningAccepted, freshAccepted, priorDigest_eq_terminal,
          runningAcceptedValue_eq_terminal, freshAcceptedValue_eq_terminal,
          Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.Accepts,
          Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.eval,
          Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.runningAccepted,
          Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.oneBased_selected]
      · simp [referenceExec, branchReferenceExec, recursiveReferenceExec,
          priorLinkAccepted, encodedEqual, iterationZero, priorLink,
          runningAccepted, priorDigest_eq_terminal,
          runningAcceptedValue_eq_terminal, freshAcceptedValue_eq_terminal,
          Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.Accepts,
          Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.eval,
          Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.runningAccepted,
          Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.oneBased_selected]
    · simp [referenceExec, branchReferenceExec, recursiveReferenceExec,
        priorLinkAccepted, encodedEqual, iterationZero, priorLink,
        priorDigest_eq_terminal, runningAcceptedValue_eq_terminal,
        freshAcceptedValue_eq_terminal,
        Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.Accepts,
        Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.eval,
        Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.runningAccepted,
        Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.oneBased_selected]

/-- Executing the intrinsic block succeeds exactly when the independent
payload-minimal fixed-one terminal checker accepts. -/
theorem accepts_iff_fixedOne
    (parameters : Parameters)
    (statement : TerminalStatement parameters.State)
    (proof : FixedOneTerminal.Proof parameters) :
    Accepts parameters statement proof ↔
      fixedOneAccepts parameters statement proof := by
  letI : DecidableEq parameters.State := parameters.stateDecidableEq
  letI : DecidableEq parameters.Encoded := parameters.encodedDecidableEq
  unfold Accepts fixedOneAccepts
  rw [← (program parameters).exec_eq_some_iff_holds
    (terminalInputValues parameters statement proof) .nil]
  rw [program_exec_eq_reference]
  exact referenceExec_eq_some_iff_fixedOne parameters statement proof

/-- The typed terminal lowering program recognizes exactly the independent
paper terminal transition, including its base and positive-iteration
boundaries. -/
theorem accepts_iff_transition
    (parameters : Parameters)
    (statement : TerminalStatement parameters.State)
    (proof : FixedOneTerminal.Proof parameters) :
    Accepts parameters statement proof ↔
      TerminalTransition parameters.setup parameters.machine
        parameters.terminalRelations statement proof.toGeneric := by
  rw [accepts_iff_fixedOne]
  letI : DecidableEq parameters.State := parameters.stateDecidableEq
  letI : DecidableEq parameters.Encoded := parameters.encodedDecidableEq
  exact
    Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.accepts_iff_transition
      parameters.setup parameters.machine parameters.terminalRelations
        parameters.terminalChecks statement proof

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal
