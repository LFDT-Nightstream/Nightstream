import Nightstream.Implementation.Rust.NifsProductionGolden.CertifiedDuplex

/-!
Exact `Pi_CCS` transcript replay with checked Poseidon2 round witnesses.

The executable replay fails if a permutation witness is absent or invalid.
Its soundness theorem identifies every returned challenge and the final state
with the canonical overwrite-duplex replay.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Rust.NifsProductionGolden.PiCcsReplay

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.SumCheck
open Nightstream.Implementation.Rust.NifsProductionGolden
open Nightstream.Implementation.Rust.NifsProductionGolden.CertifiedDuplex
open Nightstream.Implementation.Rust.PiCcsExecution

structure Result where
  alphaValues : List K
  gamma : K
  roundValues : List K
  finalTranscript : DuplexState

def reference (receipt : ProductionReceipt)
    (certificate : Finite.Certificate K) : Result :=
  let afterPublic := absorbFields receipt.piCcsStatement.publicFields
    (initialTranscript receipt.piCcsStatement)
  let afterStatement := absorbFields
    receipt.piCcsStatement.piCcsStatementFields afterPublic
  let alphaResult := deriveIndexed 42 0 6 afterStatement
  let gammaResult := squeezeSingle 43 alphaResult.2
  let roundResult := deriveRoundChallenges 0 certificate.rounds gammaResult.2
  { alphaValues := alphaResult.1
    gamma := gammaResult.1
    roundValues := roundResult.1
    finalTranscript := roundResult.2 }

def replay? (receipt : ProductionReceipt)
    (certificate : Finite.Certificate K) : Option Result := do
  let start := CertifiedDuplex.initial
    (initialTranscript receipt.piCcsStatement)
  let afterPublic <- CertifiedDuplex.absorbFields? receipt
    receipt.piCcsStatement.publicFields start
  let afterStatement <- CertifiedDuplex.absorbFields? receipt
    receipt.piCcsStatement.piCcsStatementFields afterPublic
  let alphaResult <- CertifiedDuplex.deriveIndexed? receipt 42 0 6
    afterStatement
  let gammaResult <- CertifiedDuplex.squeezeSingle? receipt 43 alphaResult.2
  let roundResult <- CertifiedDuplex.deriveRoundChallenges? receipt 0
    certificate.rounds gammaResult.2
  if roundResult.2.nextTrace = receipt.piCcsPermutationCount then
    some {
      alphaValues := alphaResult.1
      gamma := gammaResult.1
      roundValues := roundResult.1
      finalTranscript := roundResult.2.transcript }
  else
    none

theorem replay?_sound (receipt : ProductionReceipt)
    (certificate : Finite.Certificate K) (result : Result)
    (accepted : replay? receipt certificate = some result) :
    result = reference receipt certificate := by
  unfold replay? at accepted
  cases afterPublicEq : CertifiedDuplex.absorbFields? receipt
      receipt.piCcsStatement.publicFields
      (CertifiedDuplex.initial
        (initialTranscript receipt.piCcsStatement)) with
  | none => simp [afterPublicEq] at accepted
  | some afterPublic =>
    cases afterStatementEq : CertifiedDuplex.absorbFields? receipt
        receipt.piCcsStatement.piCcsStatementFields afterPublic with
    | none => simp [afterPublicEq, afterStatementEq] at accepted
    | some afterStatement =>
      cases alphaEq : CertifiedDuplex.deriveIndexed? receipt 42 0 6
          afterStatement with
      | none => simp [afterPublicEq, afterStatementEq, alphaEq] at accepted
      | some alphaResult =>
        cases gammaEq : CertifiedDuplex.squeezeSingle? receipt 43
            alphaResult.2 with
        | none =>
          simp [afterPublicEq, afterStatementEq, alphaEq, gammaEq] at accepted
        | some gammaResult =>
          cases roundEq : CertifiedDuplex.deriveRoundChallenges? receipt 0
              certificate.rounds gammaResult.2 with
          | none =>
            simp [afterPublicEq, afterStatementEq, alphaEq, gammaEq, roundEq]
              at accepted
          | some roundResult =>
            have publicSound := CertifiedDuplex.absorbFields?_sound receipt
              receipt.piCcsStatement.publicFields
              (CertifiedDuplex.initial
                (initialTranscript receipt.piCcsStatement))
              afterPublic afterPublicEq
            have statementSound := CertifiedDuplex.absorbFields?_sound receipt
              receipt.piCcsStatement.piCcsStatementFields afterPublic
              afterStatement afterStatementEq
            have alphaSound := CertifiedDuplex.deriveIndexed?_sound receipt
              42 0 6 afterStatement alphaResult.1 alphaResult.2
              (by simpa using alphaEq)
            have gammaSound := CertifiedDuplex.squeezeSingle?_sound receipt 43
              alphaResult.2 gammaResult.2 gammaResult.1
              (by simpa using gammaEq)
            have roundSound :=
              CertifiedDuplex.deriveRoundChallenges?_sound receipt 0
                certificate.rounds gammaResult.2 roundResult.1 roundResult.2
                (by simpa using roundEq)
            have accepted' :
                (if roundResult.2.nextTrace = receipt.piCcsPermutationCount then
                    some {
                      alphaValues := alphaResult.1
                      gamma := gammaResult.1
                      roundValues := roundResult.1
                      finalTranscript := roundResult.2.transcript }
                  else none) = some result := by
              simpa [afterPublicEq, afterStatementEq, alphaEq, gammaEq, roundEq]
                using accepted
            split at accepted'
            · cases Option.some.inj accepted'
              have publicSound' : afterPublic.transcript =
                  absorbFields receipt.piCcsStatement.publicFields
                    (initialTranscript receipt.piCcsStatement) := by
                simpa [absorbFields, CertifiedDuplex.initial] using publicSound
              have statementSound' : afterStatement.transcript =
                  absorbFields receipt.piCcsStatement.piCcsStatementFields
                    afterPublic.transcript := by
                simpa [absorbFields] using statementSound
              unfold reference
              dsimp only
              rw [<- publicSound']
              rw [<- statementSound']
              rw [<- alphaSound]
              dsimp only
              rw [<- gammaSound]
              dsimp only
              rw [<- roundSound]
            · contradiction

end Nightstream.Implementation.Rust.NifsProductionGolden.PiCcsReplay
