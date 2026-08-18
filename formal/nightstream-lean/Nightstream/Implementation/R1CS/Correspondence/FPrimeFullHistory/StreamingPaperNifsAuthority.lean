import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingLifecycleRelation
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Verifier

/-!
Contract: specialize the seven-part streaming lifecycle NIFS authority to the
paper SuperNeo verifier.

PiCCS and PiDEC are the verifier's two Boolean checks. PiRLC transcript,
parent, evaluation, and opening data are recomputed from the complete PiCCS
output. The output accumulator is the complete verifier-computed running
object. The theorem below proves exact equivalence with the executable verifier
on the lifecycle's one-element fresh batch.

Assurance tier: model-level.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPaperNifsAuthority

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation
open Nightstream.Protocol.FPrime
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

universe uExtension uCommitment uPublicInput uScalar uState uParams uStructure
  uNebulaOpen

section

variable
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {Nebula : Type}

local notation "PaperRunning" =>
  Running Extension Commitment PublicInput shape
local notation "PaperFresh" => Fresh Commitment PublicInput shape
local notation "PaperProof" => Proof Extension Commitment shape degreeBound

def WithFresh
    (call : NifsCall PaperRunning PaperFresh PaperProof Nebula)
    (predicate : PaperFresh -> Prop) : Prop :=
  exists fresh, call.latest = [fresh] /\ predicate fresh

def PiCcs
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (call : NifsCall PaperRunning PaperFresh PaperProof Nebula) : Prop :=
  WithFresh call fun fresh =>
    piCcsCheck key call.running fresh call.proof = true

def PiRlc
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (call : NifsCall PaperRunning PaperFresh PaperProof Nebula) : Prop :=
  WithFresh call fun fresh =>
    key.parent call.running fresh call.proof =
      PiRLC.combinedOutput key.piRlcAlgebra key.matrixSource
        (key.piCcsExecution call.running fresh call.proof).coins.roundPoint
        (key.piCcsOutputs call.running fresh call.proof)
        (key.piRlcChallenges call.running fresh call.proof)

def PiRlcTranscript
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (call : NifsCall PaperRunning PaperFresh PaperProof Nebula) : Prop :=
  WithFresh call fun fresh =>
    key.piRlcChallenges call.running fresh call.proof =
      key.piRlcResponse
        (key.absorbPiCcsOutput
          (key.piCcsExecution call.running fresh call.proof).coins.finalState
          call.proof.piCcsOutput)

def PiRlcEvaluation
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (call : NifsCall PaperRunning PaperFresh PaperProof Nebula) : Prop :=
  WithFresh call fun fresh =>
    (key.parent call.running fresh call.proof).evaluations =
      (PiRLC.combinedOutput key.piRlcAlgebra key.matrixSource
        (key.piCcsExecution call.running fresh call.proof).coins.roundPoint
        (key.piCcsOutputs call.running fresh call.proof)
        (key.piRlcChallenges call.running fresh call.proof)).evaluations

def PiRlcOpening
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (call : NifsCall PaperRunning PaperFresh PaperProof Nebula) : Prop :=
  WithFresh call fun fresh =>
    let combined := PiRLC.combinedOutput key.piRlcAlgebra key.matrixSource
      (key.piCcsExecution call.running fresh call.proof).coins.roundPoint
      (key.piCcsOutputs call.running fresh call.proof)
      (key.piRlcChallenges call.running fresh call.proof)
    (key.parent call.running fresh call.proof).commitment = combined.commitment /\
      (key.parent call.running fresh call.proof).publicInput =
        combined.publicInput

def PiDec
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (call : NifsCall PaperRunning PaperFresh PaperProof Nebula) : Prop :=
  WithFresh call fun fresh =>
    piDecCheck key call.running fresh call.proof = true

def OutputAccumulator
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (call : NifsCall PaperRunning PaperFresh PaperProof Nebula) : Prop :=
  WithFresh call fun fresh =>
    call.output = key.output call.running fresh call.proof

def authority
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound) :
    NifsAuthority PaperRunning PaperFresh PaperProof Nebula where
  piCcs := PiCcs key
  piRlc := PiRlc key
  piRlcTranscript := PiRlcTranscript key
  piRlcEvaluation := PiRlcEvaluation key
  piRlcOpening := PiRlcOpening key
  piDec := PiDec key
  outputAccumulator := OutputAccumulator key

def verifyList
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (_context : Step.NifsContext Digest Nebula)
    (running : PaperRunning) (latest : List PaperFresh)
    (proof : PaperProof) : Option PaperRunning :=
  match latest with
  | [fresh] => verify key running fresh proof
  | _ => none

theorem verifyList_eq_some_iff
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (context : Step.NifsContext Digest Nebula)
    (running : PaperRunning) (latest : List PaperFresh)
    (proof : PaperProof) (result : PaperRunning) :
    verifyList key context running latest proof = some result <->
      exists fresh,
        latest = [fresh] /\
        piCcsCheck key running fresh proof = true /\
        piDecCheck key running fresh proof = true /\
        result = key.output running fresh proof := by
  cases latest with
  | nil => simp [verifyList]
  | cons fresh tail =>
      cases tail with
      | nil =>
          simpa [verifyList] using
            (verify_eq_some_iff key running fresh proof result)
      | cons next rest => simp [verifyList]

theorem complete_iff_verifyList
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (call : NifsCall PaperRunning PaperFresh PaperProof Nebula) :
    (authority key).Complete call <->
      verifyList key call.context call.running call.latest call.proof =
        some call.output := by
  constructor
  . intro complete
    have ccs := complete.piCcs
    change PiCcs key call at ccs
    rcases ccs with ⟨fresh, latestExact, ccsAccepted⟩
    have dec := complete.piDec
    change PiDec key call at dec
    have decAccepted : piDecCheck key call.running fresh call.proof = true := by
      simpa [PiDec, WithFresh, latestExact] using dec
    have output := complete.outputAccumulator
    change OutputAccumulator key call at output
    have outputExact : call.output = key.output call.running fresh call.proof := by
      simpa [OutputAccumulator, WithFresh, latestExact] using output
    exact (verifyList_eq_some_iff key call.context call.running call.latest
      call.proof call.output).2
        ⟨fresh, latestExact, ccsAccepted, decAccepted, outputExact⟩
  . intro accepted
    rcases (verifyList_eq_some_iff key call.context call.running call.latest
      call.proof call.output).1 accepted with
      ⟨fresh, latestExact, ccsAccepted, decAccepted, outputExact⟩
    refine {
      piCcs := ⟨fresh, latestExact, ccsAccepted⟩
      piRlc := ⟨fresh, latestExact, rfl⟩
      piRlcTranscript := ⟨fresh, latestExact, rfl⟩
      piRlcEvaluation := ⟨fresh, latestExact, rfl⟩
      piRlcOpening := ⟨fresh, latestExact, rfl, rfl⟩
      piDec := ⟨fresh, latestExact, decAccepted⟩
      outputAccumulator := ⟨fresh, latestExact, outputExact⟩ }

/-- A lifecycle configuration is bound to the exact paper verifier function
and to the seven-part authority above. -/
structure BoundConfiguration
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {NebulaOpen : Type uNebulaOpen}
    {armCount : Nat}
    (configuration : Configuration Params StructureDigest PaperRunning
      PaperFresh PaperProof Nebula NebulaOpen armCount) : Prop where
  verifierExact : forall context running latest proof,
    configuration.stepSemantics.nifsVerify context running latest proof =
      verifyList key context running latest proof
  authorityExact : configuration.nifsAuthority = authority key

/-- Exact paper-verifier acceptance fixes the complete running value installed
by a recursive lifecycle step. -/
theorem recursive_output_exact
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {NebulaOpen : Type uNebulaOpen}
    {armCount : Nat}
    {configuration : Configuration Params StructureDigest PaperRunning
      PaperFresh PaperProof Nebula NebulaOpen armCount}
    (bound : BoundConfiguration key configuration)
    (recursive : Recursive configuration)
    (fresh : PaperFresh)
    (latestExact : recursive.latest = [fresh])
    (accepted : verify key recursive.running fresh recursive.nifsProof =
      some (key.output recursive.running fresh recursive.nifsProof)) :
    recursive.next = Step.advancedState configuration.stepSemantics
      recursive.prior
      (key.output recursive.running fresh recursive.nifsProof)
      recursive.input recursive.proof := by
  apply recursive.checked_fold_of_exact_verifier_output
  rw [bound.verifierExact]
  simpa [verifyList, latestExact] using accepted

end

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPaperNifsAuthority
