import Nightstream.Assurance.FPrimeConcreteNifs
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalShellSound

/-! Executable shape checks for the production concrete NIFS assurance boundary. -/

namespace NightstreamTests.FPrimeConcreteNifs

open Nightstream.Assurance.FPrimeConcreteNifs
open Nightstream.SuperNeo.Folding

example : recursiveArity.mode = .bootstrap := by decide
example : recursiveArity.mode.count
    Nightstream.SuperNeo.Concrete.productionGlobalParams = 0 := by decide
example : recursiveArity.total = 1 := recursive_total

example : terminalArity.mode = .active := by decide
example : terminalArity.mode.count
    Nightstream.SuperNeo.Concrete.productionGlobalParams = 14 := by decide
example : terminalArity.total = 15 := terminal_total

example :
    Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.nativeVerifierOrder.length =
      31 := by native_decide

/-- A proof object is only field data; it cannot carry a success proposition.
The executable verifier recomputes acceptance and rejects the zero witness at
the distinguished constant-one check. -/
def zeroProof : Proof where
  witness := fun _ => ⟨0, by decide⟩

example : recursiveCheck zeroProof = false := by rfl
example : terminalCheck zeroProof = false := by rfl
example : recursiveNativeCheck zeroProof = false := by rfl
example : terminalNativeCheck zeroProof = false := by rfl

example (assignment : Nat → Nat) :
    ownersCheck recursiveResidualOwners assignment = true ↔
      OwnersAccepted recursiveResidualOwners assignment :=
  ownersCheck_eq_true_iff recursiveResidualOwners assignment

example (assignment : Nat → Nat) :
    ownersCheck terminalResidualOwners assignment = true ↔
      OwnersAccepted terminalResidualOwners assignment :=
  ownersCheck_eq_true_iff terminalResidualOwners assignment

def tamperedContext :
    Nightstream.Protocol.FPrime.Step.NifsContext Digest Unit :=
  { Nightstream.Implementation.R1CS.FPrimeFullHistoryTranscriptSound.decodedContext
      zeroProof.assignment with chunkCount := 1 }

example : recursiveContextCheck tamperedContext zeroProof = false := by
  native_decide

example : recursiveLatestCheck [] zeroProof = false := by
  native_decide

#check recursive_rows_complete
#check terminal_rows_complete
#check terminalAuthority_sound
#check TerminalSemanticAccepted.authority
#check RecursiveSemanticAccepted.residual
#check TerminalSemanticAccepted.residual
#check Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalShellSound.NifsCompilerWitness.authorityPiDec
#check Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalShellSound.NifsCompilerWitness.authorityTail
#check Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalShellSound.NifsCompilerWitness.residual
#check RecursiveSemanticAccepted.parentShapeAgrees
#check TerminalSemanticAccepted.parentShapeAgrees
#check RecursiveSemanticAccepted.parentSerializes
#check TerminalSemanticAccepted.parentSerializes
#check recursive_rows_nifsVerify_or_badRoot

end NightstreamTests.FPrimeConcreteNifs
