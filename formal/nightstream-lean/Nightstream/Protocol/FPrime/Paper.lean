import Nightstream.SuperNeo.Folding.Nifs

/-!
Model-level HyperNova Construction-2 augmented relation over a keyed family of
typed SuperNeo NIFS relations.

Owns: the mathematical carrier of one fixed augmented function `F'_j`,
deterministic control and exact dispatch to `F_j`, the inductive
base/recursive split, exact prior and output hash preimages, and selection of
one running SuperNeo slot.

Does not own: Rust control flow, executable verifier callbacks, Poseidon2,
Fiat-Shamir encoding, R1CS rows, production public-input packing,
cryptographic hash binding, a refinement from `enc_str(F'_j)` to
`expectedStructure`, or permission to remove constraints.

Emits constraints: no.

Authority boundary: the same abstract `vk_fs` value absorbed by the hash
deterministically selects the per-slot NIFS semantics. Recursive folding is
accepted only through the resulting `Nifs.Accepted` attempt, which implies the
current candidate `Nifs.PaperNifsTransition`. The paper joint-Q to SplitNc
bridge, finite SumCheck authority, concrete key parsing, Fiat-Shamir schedules,
Poseidon2 binding, key-to-phase refinement, and the equation
`enc_str(F'_j) = expectedStructure` remain open. The hash and instance encoder
are mathematical functions; equality of outputs is not claimed to imply
equality of preimages.

| Protocol | Phase | Constraint family | Mathematical obligation | Lean owner |
|---|---|---|---|---|
| Construction 2 | carrier | program counter | recursive `pc` is 1-based and in `[1, ell]` before selection | `InRange`, `selectedIndex` |
| Construction 2 | carrier | running product | each of `ell` slots contains exactly `k` CE instances | `RunningProduct` |
| Construction 2 | carrier | fresh instance | exactly one CCS instance enters the selected fold | `selectedNifsInput` |
| Construction 2 | setup | keyed verifier family | the hashed `vk_fs` and selected slot choose both NIFS semantics and the expected relation structure | `NifsFamily`, `selectedVerifier` |
| Construction 2 | setup | default validity | every default CE child has a satisfying opening | `DefaultValid` |
| Construction 2 | setup | default structure binding | combine default validity with equality to every `(vk_fs, slot)` verifier structure | `SetupValid` |
| `F'` | control | `phi` | compute the next typed program counter from `(z_i, omega_i)` | `Machine.control` |
| `F'_j` | dispatch | fixed function identity | require `phi(z_i, omega_i)` to select this relation's fixed `j` | `ApplicationHolds.dispatch` |
| `F'_j` | application | `F_j` | evaluate exactly the fixed function selected by that dispatch equation | `Machine.step`, `ApplicationHolds.application` |
| `F'` | base | initial boundary | require `i = 0`, `z_0 = z_i`, the default running vector, and no fold | `BaseHolds` |
| `F'` | recursive | prior public input | bind fresh `u_i.x` to the exact prior typed hash preimage | `RecursiveHolds.priorPublicInput` |
| `F'` | recursive | selected structure | bind the selected fresh/running source structures to the verifier selected by `(vk_fs, pc_i)` | `SelectedInputStructuresBound` |
| `F'` | recursive | selected fold | fold `(U_i[pc_i], u_i)` through only the public independent SuperNeo transition | `RecursiveHolds.selectedNifs` |
| `F'` | recursive | inactive slots | copy every slot other than `pc_i` unchanged | `RecursiveHolds.unchanged` |
| `F'` | output | public hash | hash exactly `(vk, i+1, z_0, z_{i+1}, U_{i+1}, pc_{i+1})` | `nextHashPreimage`, `OutputHolds` |
| `F'` | branch | composition | accept exactly one base or recursive branch | `Holds` |
| `F'` | external relation | public output | hide every nondeterministic input and internal output except digest `x` | `PaperFPrimeStep` |
| assurance | structure | selected output | derive every selected output child's verifier-owned structure from the public transition and explicit input-structure binding | `NifsVerifier.Transition.outputStructure` |
| assurance | structure | running product | bind every slot to the structure selected by `(vk_fs, slot)` and preserve that binding across a recursive step | `RunningStructuresBound`, `RecursiveHolds.runningStructuresBound` |
| assurance | extraction | selected transition | expose the public NIFS transition from every non-base accepted step | `selected_nifs_transition` |
-/

namespace Nightstream.Protocol.FPrime.Paper

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment
  uScalar uChallenge uValue uState uWitness uVerifierKey uDigest

/-- The paper's explicit check `1 <= pc <= ell`. -/
def InRange (slotCount pc : Nat) : Prop :=
  1 <= pc /\ pc <= slotCount

/-- A program counter whose Construction-2 range check has already succeeded. -/
structure ProgramCounter (slotCount : Nat) where
  raw : Nat
  valid : InRange slotCount raw

namespace ProgramCounter

/-- Convert a checked 1-based program counter to Lean's 0-based slot index. -/
def index {slotCount : Nat} (pc : ProgramCounter slotCount) : Fin slotCount :=
  ⟨pc.raw - 1, by
    rcases pc.valid with ⟨lower, upper⟩
    omega⟩

/-- Canonical 1-based representation of a finite slot index. -/
def ofIndex {slotCount : Nat} (index : Fin slotCount) : ProgramCounter slotCount where
  raw := index.val + 1
  valid := by
    constructor <;> omega

@[simp] theorem ofIndex_raw {slotCount : Nat} (index : Fin slotCount) :
    (ofIndex index).raw = index.val + 1 := rfl

@[simp] theorem index_ofIndex {slotCount : Nat} (index : Fin slotCount) :
    (ofIndex index).index = index := by
  apply Fin.ext
  simp [ProgramCounter.index]

@[simp] theorem ofIndex_index {slotCount : Nat} (pc : ProgramCounter slotCount) :
    ofIndex pc.index = pc := by
  cases pc with
  | mk raw valid =>
      simp [ofIndex, index, Nat.sub_add_cancel valid.1]

end ProgramCounter

/-- Select a slot only after proving that a raw 1-based counter is valid. -/
def selectedIndex
    {slotCount pc : Nat}
    (valid : InRange slotCount pc) : Fin slotCount :=
  (ProgramCounter.mk pc valid).index

/-- One SuperNeo running accumulator: exactly the `k` Π_DEC children. -/
abbrev RunningSlot
    (Structure : Type uStructure)
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment)
    (params : GlobalParams) :=
  Fin params.k ->
    CE.Instance Structure PublicInput Point Evaluation Commitment

/-- HyperNova's `ell` independently selected SuperNeo running accumulators. -/
abbrev RunningProduct
    (Structure : Type uStructure)
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment)
    (params : GlobalParams)
    (slotCount : Nat) :=
  Fin slotCount ->
    RunningSlot Structure PublicInput Point Evaluation Commitment params

/-- Exact, typed preimage used for both the prior link and the next output. -/
structure HashPreimage
    (VerifierKey : Type uVerifierKey)
    (State : Type uState)
    (Structure : Type uStructure)
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment)
    (params : GlobalParams)
    (slotCount : Nat) where
  verifierKey : VerifierKey
  iteration : Nat
  z0 : State
  current : State
  running : RunningProduct
    Structure PublicInput Point Evaluation Commitment params slotCount
  /-- Stored as the paper's 1-based integer; recursive use proves its range. -/
  pc : Nat

/-- Nondeterministic inputs to one augmented `F'` invocation. -/
structure Input
    (VerifierKey : Type uVerifierKey)
    (State : Type uState)
    (Witness : Type uWitness)
    (Structure : Type uStructure)
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment)
    (params : GlobalParams)
    (slotCount : Nat) where
  verifierKey : VerifierKey
  iteration : Nat
  z0 : State
  zi : State
  running : RunningProduct
    Structure PublicInput Point Evaluation Commitment params slotCount
  fresh : CCS.Instance Structure PublicInput Commitment
  priorPc : Nat
  witness : Witness

/-- Advice/output values that the relation must determine exactly. -/
structure Output
    (Digest : Type uDigest)
    (State : Type uState)
    (Structure : Type uStructure)
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment)
    (params : GlobalParams)
    (slotCount : Nat) where
  zNext : State
  runningNext : RunningProduct
    Structure PublicInput Point Evaluation Commitment params slotCount
  pcNext : ProgramCounter slotCount
  x : Digest

/-- Independent paper NIFS operations consumed by Construction 2. -/
structure NifsVerifier
    (Structure : Type uStructure)
    (Assignment : Type uAssignment)
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment)
    (Scalar : Type uScalar)
    (Challenge : Type uChallenge)
    (Value : Type uValue)
    (relation : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : GlobalParams) where
  /-- Structure selected by this verifier key and HyperNova slot. -/
  expectedStructure : Structure
  /-- SuperNeo's running product is nonempty in the supported setup. -/
  kPositive : 0 < params.k
  /-- Construction 2 contributes exactly one fresh instance to the fold. -/
  oneFreshBound : 1 <= params.maxFresh
  sumcheckOps : SumCheck.Ops Challenge Value
  rlcAlgebra : PiRLC.Algebra
    Structure Assignment PublicInput Point Evaluation Commitment Scalar relation params
  decAlgebra : PiDEC.Algebra
    Structure Assignment PublicInput Point Evaluation Commitment relation params

namespace NifsVerifier

/-- The fixed one-fresh-plus-`k`-running shape of a recursive step. -/
def arity
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {relation : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    (verifier : NifsVerifier
      Structure Assignment PublicInput Point Evaluation Commitment
        Scalar Challenge Value relation params) : BatchArity params :=
  BatchArity.active params 1 (by omega) verifier.oneFreshBound

/-- Public transition accepted by the current independent phase composition.
This remains a candidate paper transition until the PiCCS joint-Q and finite
certificate bridges close. -/
def Transition
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {relation : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    (verifier : NifsVerifier
      Structure Assignment PublicInput Point Evaluation Commitment
        Scalar Challenge Value relation params)
    (input : PiCCS.InputProduct
      Structure PublicInput Point Evaluation Commitment params verifier.arity)
    (output : Fin params.k ->
      CE.Instance Structure PublicInput Point Evaluation Commitment) : Prop :=
  Nifs.PaperNifsTransition verifier.sumcheckOps verifier.rlcAlgebra
    verifier.decAlgebra input output

/-- Exact binding of every selected NIFS source to the relation structure
owned by its verifier. This is deliberately separate from transition
acceptance so structure authority cannot be inferred from self-consistent
phase messages. -/
structure InputStructuresBound
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {relation : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    (verifier : NifsVerifier
      Structure Assignment PublicInput Point Evaluation Commitment
        Scalar Challenge Value relation params)
    (input : PiCCS.InputProduct
      Structure PublicInput Point Evaluation Commitment params verifier.arity) :
    Prop where
  fresh : forall source,
    (input.fresh source).constraintSystem = verifier.expectedStructure
  running : forall child,
    (input.running child).constraintSystem = verifier.expectedStructure

/-- Every public NIFS transition preserves the verifier-owned relation
structure when its selected fresh and running inputs are explicitly bound to
that structure. The paper relation needs no retained verifier certificate for
this fact. -/
theorem Transition.outputStructure
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {relation : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {verifier : NifsVerifier
      Structure Assignment PublicInput Point Evaluation Commitment
        Scalar Challenge Value relation params}
    {input : PiCCS.InputProduct
      Structure PublicInput Point Evaluation Commitment params verifier.arity}
    {output : Fin params.k ->
      CE.Instance Structure PublicInput Point Evaluation Commitment}
    (transition : verifier.Transition input output)
    (structures : verifier.InputStructuresBound input)
    (child : Fin params.k) :
    (output child).constraintSystem = verifier.expectedStructure := by
  rcases transition with ⟨attempt, inputExact, outputExact, accepted⟩
  let first : Fin verifier.arity.total :=
    ⟨0, verifier.arity.totalPositive⟩
  have inputSourceStructure : forall index,
      (input.source index).constraintSystem = verifier.expectedStructure := by
    exact input.sourceCases
      (motive := fun source =>
        source.constraintSystem = verifier.expectedStructure)
      structures.fresh structures.running
  calc
    (output child).constraintSystem =
        (attempt.piDec.children child).constraintSystem := by
      rw [outputExact]
    _ = attempt.piDec.parent.constraintSystem :=
      accepted.piDec.sameStructure child
    _ = attempt.piRlc.output.constraintSystem := by
      rw [accepted.wiring.rlcToDec]
    _ = (attempt.piRlc.inputs first).constraintSystem :=
      (accepted.piRlc.sameStructure first).symm
    _ = (attempt.piCcs.outputs first).constraintSystem := by
      rw [accepted.wiring.ccsToRlc]
    _ = (attempt.piCcs.inputs.source first).constraintSystem :=
      accepted.piCcs.1.sameStructure first
    _ = (input.source first).constraintSystem := by
      rw [inputExact]
    _ = verifier.expectedStructure := inputSourceStructure first

end NifsVerifier

/--
Abstract verifier-key authority for the NIFS family.

This is a structural model-level binding: the exact key value included in the
hash preimage and the selected slot determine one verifier. It does not yet
refine a concrete key encoding or Fiat-Shamir transcript implementation.
-/
structure NifsFamily
    (VerifierKey : Type uVerifierKey)
    (Structure : Type uStructure)
    (Assignment : Type uAssignment)
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment)
    (Scalar : Type uScalar)
    (Challenge : Type uChallenge)
    (Value : Type uValue)
    (relation : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : GlobalParams)
    (slotCount : Nat) where
  slotPositive : 0 < slotCount
  select : VerifierKey -> Fin slotCount ->
    NifsVerifier Structure Assignment PublicInput Point Evaluation Commitment
      Scalar Challenge Value relation params

/-- The same key value hashed by `F'` selects the prior counter's verifier. -/
def selectedVerifier
    {VerifierKey : Type uVerifierKey}
    {State : Type uState}
    {Witness : Type uWitness}
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {relation : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {slotCount : Nat}
    (family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount)
    (input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount)
    (priorPcValid : InRange slotCount input.priorPc) :
    NifsVerifier Structure Assignment PublicInput Point Evaluation Commitment
      Scalar Challenge Value relation params :=
  family.select input.verifierKey (selectedIndex priorPcValid)

/-- Every running child uses the verifier-owned structure selected for its slot. -/
def RunningStructuresBound
    {VerifierKey : Type uVerifierKey}
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {relation : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {slotCount : Nat}
    (family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount)
    (verifierKey : VerifierKey)
    (running : RunningProduct Structure PublicInput Point Evaluation Commitment
      params slotCount) : Prop :=
  forall slot child,
    (running slot child).constraintSystem =
      (family.select verifierKey slot).expectedStructure

/-- Deterministic Construction-2 control, application, and hash functions. -/
structure Machine
    (VerifierKey : Type uVerifierKey)
    (Digest : Type uDigest)
    (State : Type uState)
    (Witness : Type uWitness)
    (Structure : Type uStructure)
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment)
    (params : GlobalParams)
    (slotCount : Nat) where
  /-- The deterministic control function `phi`. -/
  control : State -> Witness -> ProgramCounter slotCount
  /-- The family `(F_1, ..., F_ell)`, indexed by the selected next counter. -/
  step : Fin slotCount -> State -> Witness -> State
  /-- The paper default instance vector, specialized independently per slot. -/
  defaultRunning : RunningProduct
    Structure PublicInput Point Evaluation Commitment params slotCount
  hash : HashPreimage VerifierKey State Structure PublicInput Point Evaluation
    Commitment params slotCount -> Digest
  /-- `enc_inst` for the non-commitment public portion of the fresh instance. -/
  encodeInstance : Digest -> PublicInput

/-- Every child in a fixed running vector is a valid fresh `CE(b)` instance. -/
def RunningValid
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {relation : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {slotCount : Nat}
    (running : RunningProduct Structure PublicInput Point Evaluation Commitment
      params slotCount) : Prop :=
  exists assignments : Fin slotCount -> Fin params.k -> Assignment,
    forall slot child,
      (running slot child).stage = .fresh /\
        CE.Holds relation params (running slot child)
          (assignments slot child)

/--
Setup-level validity of the configured fixed-slot default vector.

This specializes Construction 2's universal `u_perp` assumption to the exact
`ell` structures selected for this machine. It is not a proof that one default
pair satisfies every possible structure or public parameter.
-/
def DefaultValid
    {VerifierKey : Type uVerifierKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {relation : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {slotCount : Nat}
    (machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount) : Prop :=
  RunningValid (relation := relation) machine.defaultRunning

/--
Complete fixed-slot setup premise for one verifier key and machine.

This combines semantic validity of the configured default children with their
exact binding to the structures selected by `(verifierKey, slot)`. It does not
prove that any selected `expectedStructure` is the encoding of the fixed
augmented function: `enc_str(F'_j) = expectedStructure` remains a separate
implementation-refinement obligation.
-/
def SetupValid
    {VerifierKey : Type uVerifierKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {relation : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {slotCount : Nat}
    (family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount)
    (machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount)
    (verifierKey : VerifierKey) : Prop :=
  DefaultValid (relation := relation) machine /\
    RunningStructuresBound family verifierKey machine.defaultRunning

/-- Exact prior-iteration hash preimage referenced by the fresh public input. -/
def priorHashPreimage
    {VerifierKey : Type uVerifierKey}
    {State : Type uState}
    {Witness : Type uWitness}
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {params : GlobalParams}
    {slotCount : Nat}
    (input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount) :
    HashPreimage VerifierKey State Structure PublicInput Point Evaluation
      Commitment params slotCount where
  verifierKey := input.verifierKey
  iteration := input.iteration
  z0 := input.z0
  current := input.zi
  running := input.running
  pc := input.priorPc

/-- Exact output preimage from Construction 2 step 5. -/
def nextHashPreimage
    {VerifierKey : Type uVerifierKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {params : GlobalParams}
    {slotCount : Nat}
    (input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount)
    (output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount) :
    HashPreimage VerifierKey State Structure PublicInput Point Evaluation
      Commitment params slotCount where
  verifierKey := input.verifierKey
  iteration := input.iteration + 1
  z0 := input.z0
  current := output.zNext
  running := output.runningNext
  pc := output.pcNext.raw

/-- Exactly one fresh CCS source and the `k` CE instances in the selected slot. -/
def selectedNifsInput
    {VerifierKey : Type uVerifierKey}
    {State : Type uState}
    {Witness : Type uWitness}
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {relation : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {slotCount : Nat}
    (family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount)
    (input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount)
    (priorPcValid : InRange slotCount input.priorPc) :
    PiCCS.InputProduct Structure PublicInput Point Evaluation Commitment params
      (selectedVerifier family input priorPcValid).arity where
  fresh := fun _ => input.fresh
  running := fun child => input.running (selectedIndex priorPcValid) child

/-- The selected public NIFS input is bound to the relation structure chosen
by the exact verifier key and checked prior program counter. This obligation
is separate from the NIFS transition itself and therefore independently
available to necessity analysis. -/
abbrev SelectedInputStructuresBound
    {VerifierKey : Type uVerifierKey}
    {State : Type uState}
    {Witness : Type uWitness}
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {relation : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {slotCount : Nat}
    (family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount)
    (input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount)
    (priorPcValid : InRange slotCount input.priorPc) :=
  (selectedVerifier family input priorPcValid).InputStructuresBound
    (selectedNifsInput family input priorPcValid)

/-- Deterministic `phi` and selected `F_j` evaluation shared by both branches. -/
structure ApplicationHolds
    {VerifierKey : Type uVerifierKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {params : GlobalParams}
    {slotCount : Nat}
    (machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount)
    (functionIndex : Fin slotCount)
    (input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount)
    (output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount) : Prop where
  control : output.pcNext = machine.control input.zi input.witness
  dispatch : output.pcNext = ProgramCounter.ofIndex functionIndex
  application : output.zNext =
    machine.step functionIndex input.zi input.witness

/-- Exact Construction-2 output hash equation shared by both branches. -/
def OutputHolds
    {VerifierKey : Type uVerifierKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {params : GlobalParams}
    {slotCount : Nat}
    (machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount)
    (input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount)
    (output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount) : Prop :=
  output.x = machine.hash (nextHashPreimage input output)

/-- Base branch: initial state, no NIFS fold, and the paper default vector. -/
structure BaseHolds
    {VerifierKey : Type uVerifierKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {params : GlobalParams}
    {slotCount : Nat}
    (machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount)
    (functionIndex : Fin slotCount)
    (input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount)
    (output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount) : Prop where
  iterationZero : input.iteration = 0
  initialState : input.z0 = input.zi
  application : ApplicationHolds machine functionIndex input output
  defaultRunning : output.runningNext = machine.defaultRunning
  outputHash : OutputHolds machine input output

/-- Recursive branch: prior-link check, selected NIFS fold, and exact copying. -/
structure RecursiveHolds
    {VerifierKey : Type uVerifierKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {relation : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {slotCount : Nat}
    (family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount)
    (machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount)
    (functionIndex : Fin slotCount)
    (input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount)
    (output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount) : Prop where
  iterationPositive : 0 < input.iteration
  priorPcValid : InRange slotCount input.priorPc
  priorPublicInput : input.fresh.publicInput =
    machine.encodeInstance (machine.hash (priorHashPreimage input))
  application : ApplicationHolds machine functionIndex input output
  selectedStructures :
    SelectedInputStructuresBound family input priorPcValid
  selectedNifs :
    (selectedVerifier family input priorPcValid).Transition
      (selectedNifsInput family input priorPcValid)
      (output.runningNext (selectedIndex priorPcValid))
  unchanged : forall slot, slot ≠ selectedIndex priorPcValid ->
    output.runningNext slot = input.running slot
  outputHash : OutputHolds machine input output

/-- Independent paper relation: exactly the base or recursive branch. -/
inductive Holds
    {VerifierKey : Type uVerifierKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {relation : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {slotCount : Nat}
    (family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount)
    (machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount)
    (functionIndex : Fin slotCount)
    (input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount)
    (output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount) : Prop where
  | base (accepted : BaseHolds machine functionIndex input output)
  | recursive (accepted : RecursiveHolds family machine functionIndex input output)

/-- Input-indexed projection used by later implementation refinement. -/
def InputIndexedStep
    {VerifierKey : Type uVerifierKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {relation : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {slotCount : Nat}
    (family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount)
    (machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount)
    (functionIndex : Fin slotCount)
    (input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount)
    (x : Digest) : Prop :=
  exists output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount,
    output.x = x /\ Holds family machine functionIndex input output

/--
The actual Construction-2 `F'` NP relation.

All function arguments are nondeterministic advice as required by the paper;
the sole public result is the digest `x`.
-/
def PaperFPrimeStep
    {VerifierKey : Type uVerifierKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {relation : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {slotCount : Nat}
    (family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount)
    (machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount)
    (functionIndex : Fin slotCount)
    (x : Digest) : Prop :=
  exists input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount,
    InputIndexedStep family machine functionIndex input x

/-- A rich accepted execution projects to the input-indexed public relation. -/
theorem inputIndexedStep_of_holds
    {VerifierKey : Type uVerifierKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {relation : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {slotCount : Nat}
    {family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount}
    {machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount}
    {functionIndex : Fin slotCount}
    {input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount}
    {output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount}
    (accepted : Holds family machine functionIndex input output) :
    InputIndexedStep family machine functionIndex input output.x := by
  exact ⟨output, rfl, accepted⟩

/-- A rich accepted execution projects to the public-output-only paper relation. -/
theorem paperFPrimeStep_of_holds
    {VerifierKey : Type uVerifierKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {relation : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {slotCount : Nat}
    {family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount}
    {machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount}
    {functionIndex : Fin slotCount}
    {input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount}
    {output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount}
    (accepted : Holds family machine functionIndex input output) :
    PaperFPrimeStep family machine functionIndex output.x := by
  exact ⟨input, inputIndexedStep_of_holds accepted⟩

/-- Inject an already established base branch into the rich execution relation. -/
theorem holds_of_base
    {VerifierKey : Type uVerifierKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {relation : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {slotCount : Nat}
    {family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount}
    {machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount}
    {functionIndex : Fin slotCount}
    {input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount}
    {output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount}
    (accepted : BaseHolds machine functionIndex input output) :
    Holds family machine functionIndex input output :=
  .base accepted

/-- Inject an already established recursive branch into the rich execution relation. -/
theorem holds_of_recursive
    {VerifierKey : Type uVerifierKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {relation : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {slotCount : Nat}
    {family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount}
    {machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount}
    {functionIndex : Fin slotCount}
    {input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount}
    {output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount}
    (accepted : RecursiveHolds family machine functionIndex input output) :
    Holds family machine functionIndex input output :=
  .recursive accepted

/-- Every accepted non-base `F'` step exposes its public paper NIFS transition. -/
theorem selected_nifs_transition
    {VerifierKey : Type uVerifierKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {relation : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {slotCount : Nat}
    {family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount}
    {machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount}
    {functionIndex : Fin slotCount}
    {input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount}
    {output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount}
    (accepted : Holds family machine functionIndex input output)
    (iterationPositive : 0 < input.iteration) :
    exists priorPcValid : InRange slotCount input.priorPc,
      (selectedVerifier family input priorPcValid).Transition
        (selectedNifsInput family input priorPcValid)
        (output.runningNext (selectedIndex priorPcValid)) := by
  cases accepted with
  | base baseAccepted =>
      have : False := by
        have iterationZero := baseAccepted.iterationZero
        omega
      exact this.elim
  | recursive recursiveAccepted =>
      exact ⟨recursiveAccepted.priorPcValid, recursiveAccepted.selectedNifs⟩

/-- A recursive step preserves verifier-owned structures in every running slot. -/
theorem RecursiveHolds.runningStructuresBound
    {VerifierKey : Type uVerifierKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {relation : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {slotCount : Nat}
    {family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount}
    {machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount}
    {functionIndex : Fin slotCount}
    {input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount}
    {output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount}
    (inputBound : RunningStructuresBound family input.verifierKey input.running)
    (accepted : RecursiveHolds family machine functionIndex input output) :
    RunningStructuresBound family input.verifierKey output.runningNext := by
  intro slot child
  by_cases selected : slot = selectedIndex accepted.priorPcValid
  · subst slot
    exact accepted.selectedNifs.outputStructure accepted.selectedStructures child
  · rw [accepted.unchanged slot selected]
    exact inputBound slot child

end Nightstream.Protocol.FPrime.Paper
