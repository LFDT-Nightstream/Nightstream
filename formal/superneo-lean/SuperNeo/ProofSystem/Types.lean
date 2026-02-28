import SuperNeo.ProofSystem.ConstraintSystem.CCS
import SuperNeo.ProofSystem.Sumcheck

/-!
Shared protocol-level type aliases used across theorem-construction layers.

These names provide a consistent proof-system-first surface while remaining
definitionally equal to the underlying `ProtocolRelations` structures.
-/

namespace SuperNeo

abbrev PSContext := SuperNeo.ProofSystem.ConstraintSystem.Context
abbrev PSClaim := SuperNeo.ProofSystem.ConstraintSystem.Claim
abbrev PSWitness := SuperNeo.ProofSystem.ConstraintSystem.Witness

abbrev PSSumcheckInstance := SuperNeo.ProofSystem.Sumcheck.Instance
abbrev PSSumcheckTranscript := SuperNeo.ProofSystem.Sumcheck.Transcript
abbrev PSSumcheckAccepted := SuperNeo.ProofSystem.Sumcheck.Accepted
abbrev PSSumcheckAcceptedStrong := SuperNeo.ProofSystem.Sumcheck.AcceptedStrong

abbrev PSCCSRelation := SuperNeo.ProofSystem.ConstraintSystem.CCS
abbrev PSCERelation := SuperNeo.ProofSystem.ConstraintSystem.CE
abbrev PSCERelaxedRelation := SuperNeo.ProofSystem.ConstraintSystem.CERelaxed

abbrev PSCEValid (ctx : PSContext) (claim : PSClaim) (wit : PSWitness) : Prop :=
  SuperNeo.CEValid ctx claim wit

end SuperNeo
