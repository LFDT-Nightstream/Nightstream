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

def resultValue (fixture : PiDECNonzero.Fixture) (context : KeyDigest)
    (terminal : Package.TerminalLayout) : Value :=
  .array [
    PiCCSParity.fieldWordsValue applicationOutput,
    PiCCSParity.fieldWordsValue (finalOutputPreimageWords fixture context),
    PiCCSParity.fieldWordsValue (finalOutputDigest fixture context),
    Package.TerminalLayout.format.encode terminal]

def parityValueForFixture (computed : PiCCSNonzero.Computed)
    (fixture : PiDECNonzero.Fixture) (terminal : Package.TerminalLayout) : Value :=
  .array [.atom 2, inputValue computed,
    resultValue fixture computed.statement.stateKey terminal]

def parityValueIO
    (sampler : Transcript.State → Transcript.PiRlcSampler.Batch PiRLCNonzero.SourceCount)
    (terminal : Package.TerminalLayout) (context : VerifierContext.Digest4) : IO Value := do
  let computed ← PiCCSNonzero.computeIO context.toList
  pure (parityValueForFixture computed
    (PiDECParity.fixtureFromComputed computed (sampler computed.outgoingState)) terminal)

end NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Parity
