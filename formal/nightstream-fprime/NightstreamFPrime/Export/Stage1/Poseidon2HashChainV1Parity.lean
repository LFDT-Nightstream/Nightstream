import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Export.Stage1.PiDECParity
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Package
import NightstreamFPrime.Lifecycle.Stage1.Poseidon2HashChainV1

/-!
Owns one deterministic executable parity value for the approved Stage 1
application and its next-preimage handoff. It reuses the complete nonzero
PiCCS, PiRLC, and PiDEC fixture and changes only the application-owned current
state plus the NextPreimage-owned iteration word.
-/

namespace NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Parity

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec

def message : AppWitness :=
  [Poseidon2.ofNat 7, Poseidon2.ofNat 11,
    Poseidon2.ofNat 13, Poseidon2.ofNat 17]

def applicationOutput : AppState :=
  Lifecycle.Stage1.Poseidon2HashChainV1.step
    PiCCSNonzero.stateCurrent message

def finalOutputPreimage (fixture : PiDECNonzero.Fixture)
    (context : KeyDigest) : HashPreimage
    (logicalWidth := VerifierContext.candidateLogicalWidth)
    (publicFits := VerifierContext.candidatePublicFits) :=
  let transition := PiDECParity.transitionOutputPreimage fixture context
  { transition with
    iteration := transition.iteration + 1
    current := applicationOutput }

def finalOutputPreimageWords (fixture : PiDECNonzero.Fixture)
    (context : KeyDigest) : List F :=
  serializePreimage (publicFits := VerifierContext.candidatePublicFits)
    (finalOutputPreimage fixture context)

def finalOutputDigest (fixture : PiDECNonzero.Fixture)
    (context : KeyDigest) : Digest :=
  stateHash (publicFits := VerifierContext.candidatePublicFits)
    (finalOutputPreimage fixture context)

def inputValue (computed : PiCCSNonzero.Computed) : Value :=
  .array [
    PiCCSParity.fieldWordsValue
      Lifecycle.Stage1.Poseidon2HashChainV1.domainTag,
    PiCCSParity.fieldWordsValue computed.statement.preimageWords,
    PiCCSParity.fieldWordsValue message]

def terminalLayoutValue : Value :=
  Package.TerminalLayout.format.encode
    (PerApplicationCanonicalPackage.directTerminalLayout
      Poseidon2HashChainV1Package.application)

def resultValue (fixture : PiDECNonzero.Fixture) (context : KeyDigest) : Value :=
  .array [
    PiCCSParity.fieldWordsValue applicationOutput,
    PiCCSParity.fieldWordsValue (finalOutputPreimageWords fixture context),
    PiCCSParity.fieldWordsValue (finalOutputDigest fixture context),
    terminalLayoutValue]

def parityValueForFixture (computed : PiCCSNonzero.Computed)
    (fixture : PiDECNonzero.Fixture) : Value :=
  .array [.atom 2, inputValue computed,
    resultValue fixture computed.statement.stateKey]

def parityValueIO (context : VerifierContext.Digest4) : IO Value := do
  let computed ← PiCCSNonzero.computeIO context.toList
  match Transcript.PiRlcSampler.piRlcChallengesWithState
      computed.outgoingState PiRLCNonzero.SourceCount with
  | some batch =>
      pure (parityValueForFixture computed
        (PiDECParity.fixtureFromComputed computed batch))
  | none =>
      throw (IO.userError
        "PiRLC sampler shortfall before Poseidon2HashChainV1 fixture")

end NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Parity
