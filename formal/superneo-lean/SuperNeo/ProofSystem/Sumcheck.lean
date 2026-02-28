import SuperNeo.ProtocolBase
import SuperNeo.ProofSystem.Security

/-!
Paper-facing SumCheck API surface.

This module provides stable names (`Instance`, `Transcript`, `Accepted`,
`ProtocolAssumptions`, `soundness`, `completeness`) so protocol modules can
depend on SumCheck as a first-class component instead of internal check wiring.
-/

namespace SuperNeo.ProofSystem.Sumcheck

abbrev Instance := SuperNeo.SumcheckInstance
abbrev Transcript := SuperNeo.SumcheckTranscript
abbrev Output := SuperNeo.SumcheckOutput

def ClaimTrue (inst : Instance) : Prop :=
  SuperNeo.SumcheckClaimTrue inst

def Accepted (inst : Instance) (tr : Transcript) : Prop :=
  SuperNeo.SumcheckAcceptedProp inst tr

def AcceptedStrong (inst : Instance) (tr : Transcript) : Prop :=
  SuperNeo.SumcheckAcceptedStrongProp inst tr

def ResultValid (inst : Instance) (out : Output) : Prop :=
  SuperNeo.SumcheckResultValid inst out

abbrev ProtocolAssumptions := SuperNeo.SumcheckProtocolAssumption
abbrev StrongProtocolAssumptions :=
  SuperNeo.SumcheckStrongSoundnessAssumption ∧ SuperNeo.SumcheckCompletenessAssumption

abbrev ErrorFn := SuperNeo.ProofSystem.Security.ErrorFn

/--
Paper-facing SumCheck soundness error statement.

At this layer we require negligibility of the declared SumCheck error function.
-/
def SoundnessErrorStatement (ε : ErrorFn) : Prop :=
  SuperNeo.ProofSystem.Security.IsNegligible ε

/-- SumCheck soundness boundary bundled with an explicit error-function witness. -/
def SoundnessWithErrorAssumption : Prop :=
  ∃ ε : ErrorFn, SoundnessErrorStatement ε

theorem soundness
  (hProto : ProtocolAssumptions)
  {inst : Instance} {tr : Transcript}
  (hAccepted : Accepted inst tr) :
  ResultValid inst tr.toOutput :=
  SuperNeo.sumcheckAccepted_implies_result_of_assumption hProto.1 hAccepted

theorem completeness
  (hProto : ProtocolAssumptions)
  {inst : Instance}
  (hClaim : ClaimTrue inst) :
  ∃ tr : Transcript, Accepted inst tr :=
  SuperNeo.sumcheckCompleteness_of_assumption hProto.2 hClaim

theorem strong_soundness
  (hProto : StrongProtocolAssumptions)
  {inst : Instance} {tr : Transcript}
  (hAccepted : AcceptedStrong inst tr) :
  ResultValid inst tr.toOutput :=
  SuperNeo.sumcheckAcceptedStrong_implies_result_of_assumption hProto.1 hAccepted

theorem soundness_error_of_assumption
  (hErr : SoundnessWithErrorAssumption) :
  ∃ ε : ErrorFn, SoundnessErrorStatement ε := by
  exact hErr

end SuperNeo.ProofSystem.Sumcheck
