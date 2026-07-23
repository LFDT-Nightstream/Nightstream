import Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

/-!
Contract: intrinsic typed lowering of the paper-authoritative fixed-one
HyperNova `F'` step verifier.

Owns:
- exact addressed dataflow from the fixed-one step input schema;
- common state application before an intrinsic base/recursive branch;
- the base endpoint check and verifier-owned default running value;
- the recursive prior-public link and the sole partial NIFS verification;
- direct construction of the next public digest and exact exposed result;
- equivalence with the independent fixed-one canonical verifier.

Does not own: terminal verification, a whole-verifier opaque call, physical
rows or columns, an R1CS encoding, Rust behavior, or generated artifacts.

Every call receives its authoritative operands as separate typed references.
The branch condition is computed inside the program and the two private arms
export the same one-value running schema.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Step

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

namespace Canonical

abbrev Input :=
  Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Input

end Canonical

/-- A one-slot running vector is completely determined by its selected value.
This is what lets the N-ary hash calls consume one running operand without
smuggling a whole preimage record through the IR. -/
@[simp] theorem const_selected_eq
    {α : Type}
    (values : Fin 1 -> α) :
    (fun _ => values Vocabulary.Step.selected) = values := by
  funext slot
  simp [Vocabulary.Step.selected]

@[simp] theorem step_selected_eq_canonical :
    Vocabulary.Step.selected =
      Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected :=
  rfl

namespace InputRefs

def iteration (parameters : Parameters) :
    Ref (typeSystem parameters) (stepInputSchema parameters) (.data .nat) :=
  .here _

def z0 (parameters : Parameters) :
    Ref (typeSystem parameters) (stepInputSchema parameters) (.data .state) :=
  .there (.here _)

def zi (parameters : Parameters) :
    Ref (typeSystem parameters) (stepInputSchema parameters) (.data .state) :=
  .there (.there (.here _))

def running (parameters : Parameters) :
    Ref (typeSystem parameters) (stepInputSchema parameters) (.data .running) :=
  .there (.there (.there (.here _)))

def fresh (parameters : Parameters) :
    Ref (typeSystem parameters) (stepInputSchema parameters) (.data .fresh) :=
  .there (.there (.there (.there (.here _))))

def witness (parameters : Parameters) :
    Ref (typeSystem parameters) (stepInputSchema parameters) (.data .witness) :=
  .there (.there (.there (.there (.there (.here _)))))

def nifsProof (parameters : Parameters) :
    Ref (typeSystem parameters) (stepInputSchema parameters)
      (.data .nifsProof) :=
  .there (.there (.there (.there (.there (.there (.here _))))))

end InputRefs

/-- Context after the common state application. -/
def afterStepSchema (parameters : Parameters) :
    Schema (typeSystem parameters) :=
  Ports.committedState parameters :: stepInputSchema parameters

/-- Common context after computing `zNext` and the base-branch selector. -/
def commonSchema (parameters : Parameters) :
    Schema (typeSystem parameters) :=
  Ports.auxiliaryBit parameters ::
    afterStepSchema parameters

/-- Both private arms expose exactly the selected next running claim. -/
def joinedSchema (parameters : Parameters) :
    Schema (typeSystem parameters) :=
  [Ports.committedRunning parameters]

namespace CommonRefs

def iteration (parameters : Parameters) :
    Ref (typeSystem parameters) (commonSchema parameters) (.data .nat) :=
  .there (.there (InputRefs.iteration parameters))

def z0 (parameters : Parameters) :
    Ref (typeSystem parameters) (commonSchema parameters) (.data .state) :=
  .there (.there (InputRefs.z0 parameters))

def zi (parameters : Parameters) :
    Ref (typeSystem parameters) (commonSchema parameters) (.data .state) :=
  .there (.there (InputRefs.zi parameters))

def running (parameters : Parameters) :
    Ref (typeSystem parameters) (commonSchema parameters) (.data .running) :=
  .there (.there (InputRefs.running parameters))

def fresh (parameters : Parameters) :
    Ref (typeSystem parameters) (commonSchema parameters) (.data .fresh) :=
  .there (.there (InputRefs.fresh parameters))

def nifsProof (parameters : Parameters) :
    Ref (typeSystem parameters) (commonSchema parameters)
      (.data .nifsProof) :=
  .there (.there (InputRefs.nifsProof parameters))

def zNext (parameters : Parameters) :
    Ref (typeSystem parameters) (commonSchema parameters) (.data .state) :=
  .there (.here _)

def iterationZero (parameters : Parameters) :
    Ref (typeSystem parameters) (commonSchema parameters) .bit :=
  .here _

end CommonRefs

/-- The common state application as one separately checkable primitive. -/
def stepCall (parameters : Parameters) :
    Primitive (signature parameters)
      (stepInputSchema parameters) (afterStepSchema parameters) :=
  Primitive.invoke (signature := signature parameters)
    Vocabulary.Call.step
    (.cons (InputRefs.zi parameters)
      (.cons (InputRefs.witness parameters) .nil))

/-- The internally computed branch selector as one separately checkable
primitive. -/
def iterationZeroCall (parameters : Parameters) :
    Primitive (signature parameters)
      (afterStepSchema parameters) (commonSchema parameters) :=
  Primitive.invoke (signature := signature parameters)
    Vocabulary.Call.iterationZero
    (.cons
      (.there (InputRefs.iteration parameters))
      .nil)

def afterBaseEqualitySchema (parameters : Parameters) :
    Schema (typeSystem parameters) :=
  Ports.auxiliaryBit parameters :: commonSchema parameters

def afterBaseLiteralSchema (parameters : Parameters) :
    Schema (typeSystem parameters) :=
  Ports.committedRunning parameters :: afterBaseEqualitySchema parameters

def baseStateEqualCall (parameters : Parameters) :
    Primitive (signature parameters)
      (commonSchema parameters) (afterBaseEqualitySchema parameters) :=
  Primitive.invoke (signature := signature parameters)
    Vocabulary.Call.stateEqual
    (.cons (CommonRefs.z0 parameters)
      (.cons (CommonRefs.zi parameters) .nil))

def baseAssertion (parameters : Parameters) :
    Primitive (signature parameters)
      (afterBaseEqualitySchema parameters)
      (afterBaseEqualitySchema parameters) :=
  Primitive.assertTrue (signature := signature parameters)
    (.here (Ports.auxiliaryBit parameters))

def baseDefaultCall (parameters : Parameters) :
    Primitive (signature parameters)
      (afterBaseEqualitySchema parameters)
      (afterBaseLiteralSchema parameters) :=
  Primitive.literal (signature := signature parameters)
    (Ports.committedRunning parameters) (defaultRunning parameters)

/-- Base arm: enforce `z0 = zi`, install the verifier-owned default running
claim, and export that value. -/
def baseArm (parameters : Parameters) :
    Block (signature parameters)
      (commonSchema parameters) (joinedSchema parameters) :=
  .step
    (baseStateEqualCall parameters)
    (.step
      (baseAssertion parameters)
      (.step
        (baseDefaultCall parameters)
        (.yield (.cons (.here _) .nil))))

def afterHashSchema (parameters : Parameters) :
    Schema (typeSystem parameters) :=
  Ports.auxiliaryDigest parameters :: commonSchema parameters

def afterFreshPublicSchema (parameters : Parameters) :
    Schema (typeSystem parameters) :=
  Ports.auxiliaryEncoded parameters :: afterHashSchema parameters

def afterEncodeSchema (parameters : Parameters) :
    Schema (typeSystem parameters) :=
  Ports.auxiliaryEncoded parameters :: afterFreshPublicSchema parameters

def afterEncodedEqualitySchema (parameters : Parameters) :
    Schema (typeSystem parameters) :=
  Ports.auxiliaryBit parameters :: afterEncodeSchema parameters

def afterNifsSchema (parameters : Parameters) :
    Schema (typeSystem parameters) :=
  Ports.committedRunning parameters :: afterEncodedEqualitySchema parameters

namespace RecursiveRefs

def freshAfterHash (parameters : Parameters) :
    Ref (typeSystem parameters) (afterHashSchema parameters) (.data .fresh) :=
  .there (CommonRefs.fresh parameters)

def hashAfterFreshPublic (parameters : Parameters) :
    Ref (typeSystem parameters) (afterFreshPublicSchema parameters)
      (.data .digest) :=
  .there (.here (Ports.auxiliaryDigest parameters))

def encoded (parameters : Parameters) :
    Ref (typeSystem parameters) (afterEncodeSchema parameters)
      (.data .encoded) :=
  .here (Ports.auxiliaryEncoded parameters)

def freshPublic (parameters : Parameters) :
    Ref (typeSystem parameters) (afterEncodeSchema parameters)
      (.data .encoded) :=
  .there (.here (Ports.auxiliaryEncoded parameters))

def encodedEqual (parameters : Parameters) :
    Ref (typeSystem parameters) (afterEncodedEqualitySchema parameters) .bit :=
  .here (Ports.auxiliaryBit parameters)

def running (parameters : Parameters) :
    Ref (typeSystem parameters) (afterEncodedEqualitySchema parameters)
      (.data .running) :=
  .there (.there (.there (.there (CommonRefs.running parameters))))

def fresh (parameters : Parameters) :
    Ref (typeSystem parameters) (afterEncodedEqualitySchema parameters)
      (.data .fresh) :=
  .there (.there (.there (.there (CommonRefs.fresh parameters))))

def nifsProof (parameters : Parameters) :
    Ref (typeSystem parameters) (afterEncodedEqualitySchema parameters)
      (.data .nifsProof) :=
  .there (.there (.there (.there (CommonRefs.nifsProof parameters))))

end RecursiveRefs

def hashPriorCall (parameters : Parameters) :
    Primitive (signature parameters)
      (commonSchema parameters) (afterHashSchema parameters) :=
  Primitive.invoke (signature := signature parameters)
    Vocabulary.Call.hashPrior
    (.cons (CommonRefs.iteration parameters)
      (.cons (CommonRefs.z0 parameters)
        (.cons (CommonRefs.zi parameters)
          (.cons (CommonRefs.running parameters) .nil))))

def freshPublicCall (parameters : Parameters) :
    Primitive (signature parameters)
      (afterHashSchema parameters) (afterFreshPublicSchema parameters) :=
  Primitive.invoke (signature := signature parameters)
    Vocabulary.Call.freshPublic
    (.cons (RecursiveRefs.freshAfterHash parameters) .nil)

def encodeInstanceCall (parameters : Parameters) :
    Primitive (signature parameters)
      (afterFreshPublicSchema parameters) (afterEncodeSchema parameters) :=
  Primitive.invoke (signature := signature parameters)
    Vocabulary.Call.encodeInstance
    (.cons (RecursiveRefs.hashAfterFreshPublic parameters) .nil)

def encodedEqualCall (parameters : Parameters) :
    Primitive (signature parameters)
      (afterEncodeSchema parameters) (afterEncodedEqualitySchema parameters) :=
  Primitive.invoke (signature := signature parameters)
    Vocabulary.Call.encodedEqual
    (.cons (RecursiveRefs.encoded parameters)
      (.cons (RecursiveRefs.freshPublic parameters) .nil))

def encodedAssertion (parameters : Parameters) :
    Primitive (signature parameters)
      (afterEncodedEqualitySchema parameters)
      (afterEncodedEqualitySchema parameters) :=
  Primitive.assertTrue (signature := signature parameters)
    (RecursiveRefs.encodedEqual parameters)

def nifsVerifyCall (parameters : Parameters) :
    Primitive (signature parameters)
      (afterEncodedEqualitySchema parameters) (afterNifsSchema parameters) :=
  Primitive.invoke (signature := signature parameters)
    Vocabulary.Call.nifsVerify
    (.cons (RecursiveRefs.running parameters)
      (.cons (RecursiveRefs.fresh parameters)
        (.cons (RecursiveRefs.nifsProof parameters) .nil)))

/-- Recursive arm: compute the prior digest from the authoritative inputs,
enforce the public-link encoding, run the sole partial NIFS verifier, and
export its folded running claim. -/
def recursiveArm (parameters : Parameters) :
    Block (signature parameters)
      (commonSchema parameters) (joinedSchema parameters) :=
  .step
    (hashPriorCall parameters)
    (.step
      (freshPublicCall parameters)
      (.step
        (encodeInstanceCall parameters)
        (.step
          (encodedEqualCall parameters)
          (.step
            (encodedAssertion parameters)
            (.step
              (nifsVerifyCall parameters)
              (.yield (.cons (.here _) .nil)))))))

def continuationInputSchema (parameters : Parameters) :
    Schema (typeSystem parameters) :=
  joinedSchema parameters ++ commonSchema parameters

def afterHashNextSchema (parameters : Parameters) :
    Schema (typeSystem parameters) :=
  Ports.publicDigest parameters :: continuationInputSchema parameters

def hashNextCall (parameters : Parameters) :
    Primitive (signature parameters)
      (continuationInputSchema parameters) (afterHashNextSchema parameters) :=
  Primitive.invoke (signature := signature parameters)
    Vocabulary.Call.hashNext
    (.cons
      (.there (CommonRefs.iteration parameters))
      (.cons
        (.there (CommonRefs.z0 parameters))
        (.cons
          (.there (CommonRefs.zNext parameters))
          (.cons (.here (Ports.committedRunning parameters)) .nil))))

/-- Join continuation: compute the public next digest directly from
`iteration`, `z0`, the common-prefix `zNext`, and the arm-selected running
claim, then expose `zNext`, `runningNext`, and `x` in protocol order. -/
def continuation (parameters : Parameters) :
    Block (signature parameters)
      (continuationInputSchema parameters)
      (stepResultSchema parameters) :=
  .step
    (hashNextCall parameters)
    (.yield
      (.cons
        (.there (.there (CommonRefs.zNext parameters)))
        (.cons
          (.there (.here _))
          (.cons
            (.here _)
            .nil))))

/-- Intrinsic branch and join after the common prefix. -/
def branchBlock (parameters : Parameters) :
    Block (signature parameters)
      (commonSchema parameters) (stepResultSchema parameters) :=
  .branch
    (CommonRefs.iterationZero parameters)
    (baseArm parameters)
    (recursiveArm parameters)
    (continuation parameters)

/-- Exact intrinsic step block.  Both state application and the iteration-zero
selector are computed before branching, so neither arm is selected by an
external proposition. -/
def block (parameters : Parameters) :
    Block (signature parameters)
      (stepInputSchema parameters) (stepResultSchema parameters) :=
  .step
    (stepCall parameters)
    (.step
      (iterationZeroCall parameters)
      (branchBlock parameters))

/-- Complete typed lowering program for one fixed-one `F'` step. -/
def program (parameters : Parameters) :
    Program (signature parameters)
      (stepInputSchema parameters) (stepResultSchema parameters) :=
  ⟨block parameters⟩

/-- The exact lowering relation against the exact encoded paper result. -/
def Accepts
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof)
    (output :
      Output parameters.Digest parameters.State parameters.Running 1) : Prop :=
  (program parameters).Holds
    (stepInputValues parameters input)
    (stepResultValues parameters output)

/-- The independent executable checker with the vocabulary's fixed equality
procedures installed locally. -/
def fixedOneEval
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof) :
    Option
      (Output parameters.Digest parameters.State parameters.Running 1) :=
  letI : DecidableEq parameters.State := parameters.stateDecidableEq
  letI : DecidableEq parameters.Encoded := parameters.encodedDecidableEq
  Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.eval
    parameters.setup parameters.machine input

/-- Acceptance by the independent direct fixed-one executable checker. -/
def fixedOneAccepts
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof)
    (output :
      Output parameters.Digest parameters.State parameters.Running 1) : Prop :=
  fixedOneEval parameters input = some output

/-- Runtime value after the common state application. -/
def afterStepValues
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof) :
    Schema.Values (typeSystem parameters) (afterStepSchema parameters) :=
  .cons
    (parameters.machine.step Vocabulary.Step.selected input.zi input.witness)
    (stepInputValues parameters input)

/-- Runtime value at the intrinsic branch boundary for an explicit selector. -/
def commonValuesWith
    (parameters : Parameters)
    (iterationIsZero : Bool)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof) :
    Schema.Values (typeSystem parameters) (commonSchema parameters) :=
  .cons iterationIsZero (afterStepValues parameters input)

/-- Runtime value at the intrinsic branch boundary. -/
def commonValues
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof) :
    Schema.Values (typeSystem parameters) (commonSchema parameters) :=
  commonValuesWith parameters (decide (input.iteration = 0)) input

/-- Exact one-value export shared by both private arms. -/
def joinedValues
    (parameters : Parameters)
    (runningNext : parameters.Running) :
    Schema.Values (typeSystem parameters) (joinedSchema parameters) :=
  .cons runningNext .nil

def priorDigest
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof) : parameters.Digest :=
  parameters.machine.hash {
    verifierKeys := parameters.setup.verifierKeys
    iteration := input.iteration
    z0 := input.z0
    current := input.zi
    running := fun _ => input.running Vocabulary.Step.selected
    pc := oneBased Vocabulary.Step.selected
  }

def afterBaseEqualityValues
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof) :
    Schema.Values (typeSystem parameters) (afterBaseEqualitySchema parameters) :=
  .cons (stateEqual parameters input.z0 input.zi)
    (commonValues parameters input)

def afterBaseLiteralValues
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof) :
    Schema.Values (typeSystem parameters) (afterBaseLiteralSchema parameters) :=
  .cons (defaultRunning parameters) (afterBaseEqualityValues parameters input)

def afterHashValues
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof) :
    Schema.Values (typeSystem parameters) (afterHashSchema parameters) :=
  .cons (priorDigest parameters input) (commonValues parameters input)

def afterFreshPublicValues
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof) :
    Schema.Values (typeSystem parameters) (afterFreshPublicSchema parameters) :=
  .cons (parameters.machine.freshPublic input.fresh)
    (afterHashValues parameters input)

def afterEncodeValues
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof) :
    Schema.Values (typeSystem parameters) (afterEncodeSchema parameters) :=
  .cons (parameters.machine.encodeInstance (priorDigest parameters input))
    (afterFreshPublicValues parameters input)

def priorLinkAccepted
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof) : Bool :=
  encodedEqual parameters
    (parameters.machine.encodeInstance (priorDigest parameters input))
    (parameters.machine.freshPublic input.fresh)

def afterEncodedEqualityValues
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof) :
    Schema.Values (typeSystem parameters)
      (afterEncodedEqualitySchema parameters) :=
  .cons (priorLinkAccepted parameters input)
    (afterEncodeValues parameters input)

def afterNifsValues
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof)
    (folded : parameters.Running) :
    Schema.Values (typeSystem parameters) (afterNifsSchema parameters) :=
  .cons folded (afterEncodedEqualityValues parameters input)

def continuationInputValues
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof)
    (runningNext : parameters.Running) :
    Schema.Values (typeSystem parameters) (continuationInputSchema parameters) :=
  (joinedValues parameters runningNext).append (commonValues parameters input)

def nextDigest
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof)
    (runningNext : parameters.Running) : parameters.Digest :=
  parameters.machine.hash {
    verifierKeys := parameters.setup.verifierKeys
    iteration := input.iteration + 1
    z0 := input.z0
    current :=
      parameters.machine.step Vocabulary.Step.selected input.zi input.witness
    running := fun _ => runningNext
    pc := oneBased Vocabulary.Step.selected
  }

def afterHashNextValues
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof)
    (runningNext : parameters.Running) :
    Schema.Values (typeSystem parameters) (afterHashNextSchema parameters) :=
  .cons (nextDigest parameters input runningNext)
    (continuationInputValues parameters input runningNext)

def resultValuesFor
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof)
    (runningNext : parameters.Running) :
    Schema.Values (typeSystem parameters) (stepResultSchema parameters) :=
  stepResultValues parameters
    (Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.outputFor
      parameters.setup parameters.machine input (fun _ => runningNext))

/-- The state application reduces independently of every branch-local
obligation. -/
theorem stepCall_exec
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof) :
    (stepCall parameters).exec (stepInputValues parameters input) =
      some (afterStepValues parameters input) :=
  rfl

/-- The selector call reduces independently of either private arm. -/
theorem iterationZeroCall_exec
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof) :
    (iterationZeroCall parameters).exec (afterStepValues parameters input) =
      some (commonValues parameters input) :=
  rfl

theorem baseStateEqualCall_exec
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof) :
    (baseStateEqualCall parameters).exec (commonValues parameters input) =
      some (afterBaseEqualityValues parameters input) :=
  rfl

theorem baseAssertion_exec
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof) :
    (baseAssertion parameters).exec (afterBaseEqualityValues parameters input) =
      if stateEqual parameters input.z0 input.zi then
        some (afterBaseEqualityValues parameters input)
      else
        none := by
  change
    (match stateEqual parameters input.z0 input.zi with
      | false => none
      | true => some (afterBaseEqualityValues parameters input)) =
      if stateEqual parameters input.z0 input.zi then
        some (afterBaseEqualityValues parameters input)
      else
        none
  cases stateEqual parameters input.z0 input.zi <;> rfl

theorem baseDefaultCall_exec
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof) :
    (baseDefaultCall parameters).exec (afterBaseEqualityValues parameters input) =
      some (afterBaseLiteralValues parameters input) :=
  rfl

theorem baseExports_get
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof) :
    ((Exports.cons
        (.here (Ports.committedRunning parameters)) Exports.nil :
      Exports (typeSystem parameters) (afterBaseLiteralSchema parameters)
        (joinedSchema parameters)).get
        (afterBaseLiteralValues parameters input)) =
      joinedValues parameters (defaultRunning parameters) := by
  rfl

theorem hashPriorCall_exec
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof) :
    (hashPriorCall parameters).exec (commonValues parameters input) =
      some (afterHashValues parameters input) :=
  rfl

theorem freshPublicCall_exec
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof) :
    (freshPublicCall parameters).exec (afterHashValues parameters input) =
      some (afterFreshPublicValues parameters input) :=
  rfl

theorem encodeInstanceCall_exec
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof) :
    (encodeInstanceCall parameters).exec
        (afterFreshPublicValues parameters input) =
      some (afterEncodeValues parameters input) :=
  rfl

theorem encodedEqualCall_exec
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof) :
    (encodedEqualCall parameters).exec (afterEncodeValues parameters input) =
      some (afterEncodedEqualityValues parameters input) :=
  rfl

theorem encodedAssertion_exec
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof) :
    (encodedAssertion parameters).exec
        (afterEncodedEqualityValues parameters input) =
      if priorLinkAccepted parameters input then
        some (afterEncodedEqualityValues parameters input)
      else
        none := by
  change
    (match priorLinkAccepted parameters input with
      | false => none
      | true => some (afterEncodedEqualityValues parameters input)) =
      if priorLinkAccepted parameters input then
        some (afterEncodedEqualityValues parameters input)
      else
        none
  cases priorLinkAccepted parameters input <;> rfl

theorem recursiveRunning_get
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof) :
    (RecursiveRefs.running parameters).get
        (afterEncodedEqualityValues parameters input) =
      input.running Vocabulary.Step.selected := by
  rfl

theorem recursiveFresh_get
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof) :
    (RecursiveRefs.fresh parameters).get
        (afterEncodedEqualityValues parameters input) = input.fresh := by
  rfl

theorem recursiveNifsProof_get
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof) :
    (RecursiveRefs.nifsProof parameters).get
        (afterEncodedEqualityValues parameters input) = input.nifsProof := by
  rfl

theorem nifsOperands_get
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof) :
    (Refs.cons (RecursiveRefs.running parameters)
        (Refs.cons (RecursiveRefs.fresh parameters)
          (Refs.cons (RecursiveRefs.nifsProof parameters) Refs.nil))).get
        (afterEncodedEqualityValues parameters input) =
      .cons (input.running Vocabulary.Step.selected)
        (.cons input.fresh (.cons input.nifsProof .nil)) := by
  rfl

theorem nifsCallEval
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof) :
    callEval parameters Vocabulary.Call.nifsVerify
        ((Refs.cons (RecursiveRefs.running parameters)
          (Refs.cons (RecursiveRefs.fresh parameters)
            (Refs.cons (RecursiveRefs.nifsProof parameters) Refs.nil))).get
          (afterEncodedEqualityValues parameters input)) =
      match parameters.setup.nifs.verify
          (parameters.setup.verifierKeys Vocabulary.Step.selected)
          (input.running Vocabulary.Step.selected)
          input.fresh input.nifsProof with
      | none => none
      | some folded => some (.cons folded .nil) := by
  calc
    _ = callEval parameters Vocabulary.Call.nifsVerify
        (.cons (input.running Vocabulary.Step.selected)
          (.cons input.fresh (.cons input.nifsProof .nil))) :=
      congrArg (callEval parameters Vocabulary.Call.nifsVerify)
        (nifsOperands_get parameters input)
    _ = _ := rfl

theorem nifsVerifyCall_exec
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof) :
    (nifsVerifyCall parameters).exec
        (afterEncodedEqualityValues parameters input) =
      match parameters.setup.nifs.verify
          (parameters.setup.verifierKeys Vocabulary.Step.selected)
          (input.running Vocabulary.Step.selected)
          input.fresh input.nifsProof with
      | none => none
      | some folded => some (afterNifsValues parameters input folded) :=
  by
    cases verifierResult : parameters.setup.nifs.verify
        (parameters.setup.verifierKeys Vocabulary.Step.selected)
        (input.running Vocabulary.Step.selected)
        input.fresh input.nifsProof with
    | none =>
        have evaluated := nifsCallEval parameters input
        rw [verifierResult] at evaluated
        calc
          _ = none := by
            unfold nifsVerifyCall
            exact Primitive.invoke_exec_of_eq_none evaluated
          _ = _ := by
            simp [verifierResult]
    | some folded =>
        have evaluated := nifsCallEval parameters input
        rw [verifierResult] at evaluated
        calc
          _ = some
              ((HVec.cons folded HVec.nil).append
                (afterEncodedEqualityValues parameters input)) := by
            unfold nifsVerifyCall
            exact Primitive.invoke_exec_of_eq_some evaluated
          _ = some (afterNifsValues parameters input folded) := rfl
          _ = _ := by
            simp [verifierResult]
            rfl

theorem nifsExports_get
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof)
    (folded : parameters.Running) :
    ((Exports.cons
        (.here (Ports.committedRunning parameters)) Exports.nil :
      Exports (typeSystem parameters) (afterNifsSchema parameters)
        (joinedSchema parameters)).get
        (afterNifsValues parameters input folded)) =
      joinedValues parameters folded := by
  rfl

theorem hashNextCall_exec
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof)
    (runningNext : parameters.Running) :
    (hashNextCall parameters).exec
        (continuationInputValues parameters input runningNext) =
      some (afterHashNextValues parameters input runningNext) :=
  rfl

theorem continuationExports_get
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof)
    (runningNext : parameters.Running) :
    ((Exports.cons
        (.there (.there (CommonRefs.zNext parameters)))
        (Exports.cons
          (.there (.here (Ports.committedRunning parameters)))
          (Exports.cons
            (.here (Ports.publicDigest parameters)) Exports.nil)) :
      Exports (typeSystem parameters) (afterHashNextSchema parameters)
        (stepResultSchema parameters)).get
        (afterHashNextValues parameters input runningNext)) =
      resultValuesFor parameters input runningNext := by
  rfl

theorem priorDigest_eq_canonical
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof) :
    priorDigest parameters input =
      parameters.machine.hash
        (Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.priorHashPreimage
          parameters.setup input) := by
  unfold priorDigest
  unfold Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.priorHashPreimage
  rw [const_selected_eq]
  rw [step_selected_eq_canonical]

theorem priorLinkAccepted_eq_true_iff
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof) :
    priorLinkAccepted parameters input = true ↔
      parameters.machine.freshPublic input.fresh =
        parameters.machine.encodeInstance
          (parameters.machine.hash
            (Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.priorHashPreimage
              parameters.setup input)) := by
  simp [priorLinkAccepted, encodedEqual, priorDigest_eq_canonical, eq_comm]

/-- Isolated accepting base-arm execution. -/
theorem baseArm_exec_of_initial
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof)
    (initialState : input.z0 = input.zi) :
    (baseArm parameters).exec (commonValues parameters input) =
      some (joinedValues parameters (defaultRunning parameters)) := by
  simp only [baseArm, Block.exec]
  simp only [baseStateEqualCall_exec]
  rw [baseAssertion_exec]
  have accepted : stateEqual parameters input.z0 input.zi = true := by
    simp [stateEqual, initialState]
  simp only [accepted, if_pos, baseDefaultCall_exec]
  exact congrArg some (baseExports_get parameters input)

/-- Isolated rejecting base-arm execution. -/
theorem baseArm_exec_of_not_initial
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof)
    (initialState : ¬input.z0 = input.zi) :
    (baseArm parameters).exec (commonValues parameters input) = none := by
  simp only [baseArm, Block.exec]
  simp only [baseStateEqualCall_exec]
  rw [baseAssertion_exec]
  have rejected : stateEqual parameters input.z0 input.zi = false := by
    simp [stateEqual, initialState]
  simp [rejected]

/-- Isolated recursive-arm rejection at the prior public link. -/
theorem recursiveArm_exec_of_not_public
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof)
    (priorPublic :
      ¬parameters.machine.freshPublic input.fresh =
        parameters.machine.encodeInstance
          (parameters.machine.hash
            (Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.priorHashPreimage
              parameters.setup input))) :
    (recursiveArm parameters).exec (commonValues parameters input) = none := by
  simp only [recursiveArm, Block.exec]
  simp only [hashPriorCall_exec, freshPublicCall_exec,
    encodeInstanceCall_exec, encodedEqualCall_exec]
  rw [encodedAssertion_exec]
  have rejected : priorLinkAccepted parameters input = false := by
    cases accepted : priorLinkAccepted parameters input
    · rfl
    · exfalso
      exact priorPublic
        ((priorLinkAccepted_eq_true_iff parameters input).mp accepted)
  simp [rejected]

/-- Isolated recursive-arm rejection at the sole partial NIFS call. -/
theorem recursiveArm_exec_of_nifs_reject
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof)
    (priorPublic :
      parameters.machine.freshPublic input.fresh =
        parameters.machine.encodeInstance
          (parameters.machine.hash
            (Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.priorHashPreimage
              parameters.setup input)))
    (verifierResult :
      parameters.setup.nifs.verify
        (parameters.setup.verifierKeys Vocabulary.Step.selected)
        (input.running Vocabulary.Step.selected)
        input.fresh input.nifsProof = none) :
    (recursiveArm parameters).exec (commonValues parameters input) = none := by
  simp only [recursiveArm, Block.exec]
  simp only [hashPriorCall_exec, freshPublicCall_exec,
    encodeInstanceCall_exec, encodedEqualCall_exec]
  rw [encodedAssertion_exec]
  have accepted : priorLinkAccepted parameters input = true :=
    (priorLinkAccepted_eq_true_iff parameters input).mpr priorPublic
  simp only [accepted, if_pos]
  rw [nifsVerifyCall_exec]
  rw [verifierResult]

/-- Isolated accepting recursive-arm execution. -/
theorem recursiveArm_exec_of_nifs_accept
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof)
    (folded : parameters.Running)
    (priorPublic :
      parameters.machine.freshPublic input.fresh =
        parameters.machine.encodeInstance
          (parameters.machine.hash
            (Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.priorHashPreimage
              parameters.setup input)))
    (verifierResult :
      parameters.setup.nifs.verify
        (parameters.setup.verifierKeys Vocabulary.Step.selected)
        (input.running Vocabulary.Step.selected)
        input.fresh input.nifsProof = some folded) :
    (recursiveArm parameters).exec (commonValues parameters input) =
      some (joinedValues parameters folded) := by
  simp only [recursiveArm, Block.exec]
  simp only [hashPriorCall_exec, freshPublicCall_exec,
    encodeInstanceCall_exec, encodedEqualCall_exec]
  rw [encodedAssertion_exec]
  have accepted : priorLinkAccepted parameters input = true :=
    (priorLinkAccepted_eq_true_iff parameters input).mpr priorPublic
  simp only [accepted, if_pos]
  rw [nifsVerifyCall_exec]
  rw [verifierResult]
  exact congrArg some (nifsExports_get parameters input folded)

/-- Isolated common suffix execution. -/
theorem continuation_exec
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof)
    (runningNext : parameters.Running) :
    (continuation parameters).exec
        (continuationInputValues parameters input runningNext) =
      some (resultValuesFor parameters input runningNext) := by
  simp only [continuation, Block.exec]
  simp only [hashNextCall_exec]
  exact congrArg some
    (continuationExports_get parameters input runningNext)

theorem branchSelector_get
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof) :
    (signature parameters).types.bitValue
        ((CommonRefs.iterationZero parameters).get
          (commonValues parameters input)) =
      decide (input.iteration = 0) := by
  rfl

theorem branchBlock_exec_base_accept
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof)
    (iterationZero : input.iteration = 0)
    (initialState : input.z0 = input.zi) :
    (branchBlock parameters).exec (commonValues parameters input) =
      some (resultValuesFor parameters input (defaultRunning parameters)) := by
  unfold branchBlock
  have selected :
      (signature parameters).types.bitValue
          ((CommonRefs.iterationZero parameters).get
            (commonValues parameters input)) = true := by
    rw [branchSelector_get]
    simp [iterationZero]
  rw [Block.branch_exec_of_selector_true selected]
  rw [baseArm_exec_of_initial parameters input initialState]
  simp only
  change
    (continuation parameters).exec
        (continuationInputValues parameters input (defaultRunning parameters)) =
      some (resultValuesFor parameters input (defaultRunning parameters))
  exact continuation_exec parameters input (defaultRunning parameters)

theorem branchBlock_exec_base_reject
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof)
    (iterationZero : input.iteration = 0)
    (initialState : ¬input.z0 = input.zi) :
    (branchBlock parameters).exec (commonValues parameters input) = none := by
  unfold branchBlock
  have selected :
      (signature parameters).types.bitValue
          ((CommonRefs.iterationZero parameters).get
            (commonValues parameters input)) = true := by
    rw [branchSelector_get]
    simp [iterationZero]
  rw [Block.branch_exec_of_selector_true selected]
  rw [baseArm_exec_of_not_initial parameters input initialState]

theorem branchBlock_exec_recursive_not_public
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof)
    (iterationZero : ¬input.iteration = 0)
    (priorPublic :
      ¬parameters.machine.freshPublic input.fresh =
        parameters.machine.encodeInstance
          (parameters.machine.hash
            (Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.priorHashPreimage
              parameters.setup input))) :
    (branchBlock parameters).exec (commonValues parameters input) = none := by
  unfold branchBlock
  have selected :
      (signature parameters).types.bitValue
          ((CommonRefs.iterationZero parameters).get
            (commonValues parameters input)) = false := by
    rw [branchSelector_get]
    simp [iterationZero]
  rw [Block.branch_exec_of_selector_false selected]
  rw [recursiveArm_exec_of_not_public parameters input priorPublic]

theorem branchBlock_exec_recursive_nifs_reject
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof)
    (iterationZero : ¬input.iteration = 0)
    (priorPublic :
      parameters.machine.freshPublic input.fresh =
        parameters.machine.encodeInstance
          (parameters.machine.hash
            (Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.priorHashPreimage
              parameters.setup input)))
    (verifierResult :
      parameters.setup.nifs.verify
        (parameters.setup.verifierKeys Vocabulary.Step.selected)
        (input.running Vocabulary.Step.selected)
        input.fresh input.nifsProof = none) :
    (branchBlock parameters).exec (commonValues parameters input) = none := by
  unfold branchBlock
  have selected :
      (signature parameters).types.bitValue
          ((CommonRefs.iterationZero parameters).get
            (commonValues parameters input)) = false := by
    rw [branchSelector_get]
    simp [iterationZero]
  rw [Block.branch_exec_of_selector_false selected]
  rw [recursiveArm_exec_of_nifs_reject
    parameters input priorPublic verifierResult]

theorem branchBlock_exec_recursive_nifs_accept
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof)
    (folded : parameters.Running)
    (iterationZero : ¬input.iteration = 0)
    (priorPublic :
      parameters.machine.freshPublic input.fresh =
        parameters.machine.encodeInstance
          (parameters.machine.hash
            (Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.priorHashPreimage
              parameters.setup input)))
    (verifierResult :
      parameters.setup.nifs.verify
        (parameters.setup.verifierKeys Vocabulary.Step.selected)
        (input.running Vocabulary.Step.selected)
        input.fresh input.nifsProof = some folded) :
    (branchBlock parameters).exec (commonValues parameters input) =
      some (resultValuesFor parameters input folded) := by
  unfold branchBlock
  have selected :
      (signature parameters).types.bitValue
          ((CommonRefs.iterationZero parameters).get
            (commonValues parameters input)) = false := by
    rw [branchSelector_get]
    simp [iterationZero]
  rw [Block.branch_exec_of_selector_false selected]
  rw [recursiveArm_exec_of_nifs_accept
    parameters input folded priorPublic verifierResult]
  simp only
  change
    (continuation parameters).exec
        (continuationInputValues parameters input folded) =
      some (resultValuesFor parameters input folded)
  exact continuation_exec parameters input folded

/-- The exposed result schema is lossless for a one-slot paper output:
`pcNext` and all non-selected running entries are type-level singletons. -/
theorem stepResultValues_injective
    (parameters : Parameters) :
    Function.Injective (stepResultValues parameters) := by
  intro left right valuesEqual
  have zNextEqual : left.zNext = right.zNext :=
    congrArg HVec.head valuesEqual
  have runningEqual :
      left.runningNext Vocabulary.Step.selected =
        right.runningNext Vocabulary.Step.selected :=
    congrArg HVec.head (congrArg HVec.tail valuesEqual)
  have digestEqual : left.x = right.x :=
    congrArg HVec.head
      (congrArg HVec.tail (congrArg HVec.tail valuesEqual))
  have runningFunctionEqual : left.runningNext = right.runningNext := by
    funext slot
    have slotEqual :
        slot = Vocabulary.Step.selected :=
      Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.fin_eq_selected slot
    simpa only [slotEqual] using runningEqual
  have pcEqual : left.pcNext = right.pcNext :=
    Subsingleton.elim left.pcNext right.pcNext
  cases left
  cases right
  cases zNextEqual
  cases runningFunctionEqual
  cases pcEqual
  cases digestEqual
  rfl

/-- Program execution and the independent fixed-one evaluator compute exactly
the same exposed result. -/
theorem exec_eq_map_fixedOne
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof) :
    (program parameters).exec (stepInputValues parameters input) =
      Option.map (stepResultValues parameters)
        (fixedOneEval parameters input) := by
  letI : DecidableEq parameters.State := parameters.stateDecidableEq
  letI : DecidableEq parameters.Encoded := parameters.encodedDecidableEq
  unfold fixedOneEval
  by_cases iterationZero : input.iteration = 0
  · by_cases initialState : input.z0 = input.zi
    · simp only [program, Program.exec, block, Block.exec,
        stepCall_exec, iterationZeroCall_exec]
      rw [branchBlock_exec_base_accept
        parameters input iterationZero initialState]
      simp only [Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.eval,
        iterationZero, if_pos, initialState, Option.map_some]
      unfold resultValuesFor defaultRunning
      rfl
    · simp only [program, Program.exec, block, Block.exec,
        stepCall_exec, iterationZeroCall_exec]
      rw [branchBlock_exec_base_reject
        parameters input iterationZero initialState]
      simp only [Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.eval,
        iterationZero, if_pos, initialState, if_neg, Option.map_none]
      simp
      rfl
  · by_cases priorPublic :
        parameters.machine.freshPublic input.fresh =
          parameters.machine.encodeInstance
            (parameters.machine.hash
              (Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.priorHashPreimage
                parameters.setup input))
    · cases verifierResult :
        parameters.setup.nifs.verify
          (parameters.setup.verifierKeys Vocabulary.Step.selected)
          (input.running Vocabulary.Step.selected)
          input.fresh input.nifsProof with
      | none =>
          simp only [program, Program.exec, block, Block.exec,
            stepCall_exec, iterationZeroCall_exec]
          rw [branchBlock_exec_recursive_nifs_reject
            parameters input iterationZero priorPublic verifierResult]
          have verifierResultCanonical :
              parameters.setup.nifs.verify
                (parameters.setup.verifierKeys
                  Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
                (input.running
                  Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
                input.fresh input.nifsProof = none := by
            simpa only [step_selected_eq_canonical] using verifierResult
          simp only [Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.eval,
            iterationZero, if_neg, priorPublic, if_pos,
            verifierResultCanonical, Option.map_none]
          simp
          rfl
      | some folded =>
          simp only [program, Program.exec, block, Block.exec,
            stepCall_exec, iterationZeroCall_exec]
          rw [branchBlock_exec_recursive_nifs_accept
            parameters input folded iterationZero priorPublic verifierResult]
          have verifierResultCanonical :
              parameters.setup.nifs.verify
                (parameters.setup.verifierKeys
                  Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
                (input.running
                  Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
                input.fresh input.nifsProof = some folded := by
            simpa only [step_selected_eq_canonical] using verifierResult
          simp only [Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.eval,
            iterationZero, if_neg, priorPublic, if_pos,
            verifierResultCanonical, Option.map_some]
          rfl
    · simp only [program, Program.exec, block, Block.exec,
        stepCall_exec, iterationZeroCall_exec]
      rw [branchBlock_exec_recursive_not_public
        parameters input iterationZero priorPublic]
      simp only [Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.eval,
        iterationZero, if_neg, priorPublic, Option.map_none]
      simp
      rfl

/-- The intrinsic program accepts exactly the direct fixed-one canonical
verifier. -/
theorem accepts_iff_fixedOne
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof)
    (output :
      Output parameters.Digest parameters.State parameters.Running 1) :
    Accepts parameters input output ↔
      fixedOneAccepts parameters input output := by
  unfold Accepts
  rw [← Program.exec_eq_some_iff_holds]
  rw [exec_eq_map_fixedOne]
  unfold fixedOneAccepts
  cases evaluated : fixedOneEval parameters input with
  | none =>
      simp [evaluated]
  | some computed =>
      simp only [Option.map_some]
      constructor
      · intro exposedEqual
        exact
          congrArg some
            (stepResultValues_injective parameters
              (Option.some.inj exposedEqual))
      · intro outputEqual
        cases outputEqual
        rfl

/-- Consequently, the intrinsic lowering is exactly the frozen one-slot
Construction-2 transition. -/
theorem accepts_iff_transition
    (parameters : Parameters)
    (input :
      Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof)
    (output :
      Output parameters.Digest parameters.State parameters.Running 1) :
    Accepts parameters input output ↔
      Transition parameters.setup parameters.machine
        Vocabulary.Step.selected
        (input.toGeneric (Key := parameters.Key)) output := by
  rw [accepts_iff_fixedOne]
  letI : DecidableEq parameters.State := parameters.stateDecidableEq
  letI : DecidableEq parameters.Encoded := parameters.encodedDecidableEq
  simpa only [fixedOneAccepts, fixedOneEval] using
    Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.accepts_iff_transition
        parameters.setup parameters.machine input output

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Step
