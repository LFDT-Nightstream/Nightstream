import Nightstream.Implementation.R1CS.Canonical.KSplitNcOperationalRows
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.CanonicalOpeningSplitNc.SelectedPhysicalRefinement

/-!
Contract: connect the selected operational Split-NC verifier rows to the
selected physical canonical-opening rows.

Assurance tier: security-reduced.

Owns: exact composition of Lean-owned verifier-row satisfaction, deterministic
Split-NC soundness, selected 61-column opening placement, and the 21-row
canonicality theorem.

Does not own: exclusion probabilities for the named output-binding and
algebraic bad events, or proof that an enclosing relation activates the
canonical-opening rows.

Emits constraints: no new rows.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningSplitNc.SelectedVerifierRefinement

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.ShiftedTernaryCompiler
open Nightstream.Implementation.R1CS.ShiftedTernarySound
open Nightstream.Implementation.R1CS.CenteredTernaryBorrowChunk
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningSplitNc
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningSplitNc.SelectedPhysicalRefinement
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

/-- The exact security branches retained after deterministic verifier
soundness. They are not converted into a success proposition. -/
def SecurityEvent
    {rows columns freshCount runningCount : Nat}
    {relationProfile : RelationProfile.Profile rows columns}
    {domains : Domains}
    (covers :
      domains.nc.Covers
        (ncShape relationProfile freshCount runningCount))
    (data : Data (ncShape relationProfile freshCount runningCount))
    (feProfile :
      Polynomial.Fe.SupportedProfile
        (ncShape relationProfile freshCount runningCount) domains.fe)
    (constants : Poseidon2Schedule.Constants)
    (input :
      KSplitNcOperationalRows.Input
        (PublicInput.ofSources data) domains)
    (message :
      OutputMessage (ncShape relationProfile freshCount runningCount))
    (assignment : Nat → Nat)
    (challengeSetSize : Nat) : Prop :=
  let schedule :=
    KSplitNcTranscriptSemantics.valueSchedule
      constants assignment input.transcript
  let priorState :=
    KSplitNcTranscriptSemantics.priorState
      assignment input.transcript
  let certificate :=
    KSplitNcOperational.certificate
      assignment input.transcript message
  let pre :=
    Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane.derivePreSumcheck
      schedule priorState KSplitNcTranscriptSemantics.unitStatement
  let execution :=
    Protocol.BlockLane.derive
      (fun _ : Unit => PublicInput.ofSources data)
      schedule priorState feProfile
      KSplitNcTranscriptSemantics.unitStatement certificate
  ¬ Protocol.BlockLane.OutputBound
      covers data execution certificate.output ∨
    Protocol.BlockLane.BadEvent
      feProfile covers data pre.challenges execution
      (Protocol.BlockLane.certificateAtSources data certificate rfl)
      challengeSetSize

/-- Satisfying the selected operational verifier rows yields NC truth or one
of the unchanged named security branches. -/
theorem ncTruth_or_securityEvent_of_selectedVerifierRows
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {rows columns freshCount runningCount : Nat}
    {relationProfile : RelationProfile.Profile rows columns}
    {domains : Domains}
    (covers :
      domains.nc.Covers
        (ncShape relationProfile freshCount runningCount))
    (data : Data (ncShape relationProfile freshCount runningCount))
    (feProfile :
      Polynomial.Fe.SupportedProfile
        (ncShape relationProfile freshCount runningCount) domains.fe)
    (constants : Poseidon2Schedule.Constants)
    (input :
      KSplitNcOperationalRows.Input
        (PublicInput.ofSources data) domains)
    (message :
      OutputMessage (ncShape relationProfile freshCount runningCount))
    (assignment : Nat → Nat)
    (residues : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (authority :
      KSplitNcEndpoints.DecodedAuthority
        (KSplitNcOperationalRows.endpointInput input)
        assignment message)
    (verifierRows :
      Satisfies
        (KSplitNcOperationalRows.rows constants input)
        assignment)
    (challengeSetSize : Nat) :
    Semantics.Nc.Truth data ∨
      SecurityEvent covers data feProfile constants input message
        assignment challengeSetSize := by
  have accepted :=
    KSplitNcOperationalRows.accepted_of_rows
      feProfile constants input message assignment residues
      constantWire authority verifierRows
  have soundness :=
    Protocol.BlockLane.accepted_implies_paperObligations_or_unbound_or_badEvent
      noZeroDivisors covers
      (fun _ : Unit => PublicInput.ofSources data)
      (KSplitNcTranscriptSemantics.valueSchedule
        constants assignment input.transcript)
      (KSplitNcTranscriptSemantics.priorState
        assignment input.transcript)
      feProfile KSplitNcTranscriptSemantics.unitStatement
      data rfl
      (KSplitNcOperational.certificate
        assignment input.transcript message)
      challengeSetSize accepted
  unfold SecurityEvent
  dsimp only at soundness
  rcases soundness with paper | unbound | badEvent
  · have truth := (Semantics.truth_iff_paperHolds data).mpr paper
    exact Or.inl truth.2
  · exact Or.inr (Or.inl unbound)
  · exact Or.inr (Or.inr badEvent)

/-- Headline end-to-end theorem. The selected verifier rows provide the
`b = 2` digit bound on their sound branch. The selected physical rows provide
the 21 canonical equations. The result is canonicality or an unchanged named
security event. -/
theorem selectedVerifierAndPhysicalRows_encoded_lt_modulus_or_securityEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {rows columns freshCount runningCount openingCount base : Nat}
    {relationProfile : RelationProfile.Profile rows columns}
    {domains : Domains}
    (fits : base + openingCount * openingWidth ≤ columns)
    (covers :
      domains.nc.Covers
        (ncShape relationProfile freshCount runningCount))
    (data : Data (ncShape relationProfile freshCount runningCount))
    (feProfile :
      Polynomial.Fe.SupportedProfile
        (ncShape relationProfile freshCount runningCount) domains.fe)
    (constants : Poseidon2Schedule.Constants)
    (input :
      KSplitNcOperationalRows.Input
        (PublicInput.ofSources data) domains)
    (message :
      OutputMessage (ncShape relationProfile freshCount runningCount))
    (assignment : Nat → Nat)
    (residues : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (authority :
      KSplitNcEndpoints.DecodedAuthority
        (KSplitNcOperationalRows.endpointInput input)
        assignment message)
    (verifierRows :
      Satisfies
        (KSplitNcOperationalRows.rows constants input)
        assignment)
    (challengeSetSize : Nat)
    (source :
      Fin (ncShape relationProfile freshCount runningCount).sourceCount)
    (opening : Fin openingCount)
    (canonicalRows :
      EmittedRowsHold
        (selectedLayout base fits) data source opening) :
    lowValue
        (assignmentTritMod
          (localAssignment
            (selectedLayout base fits) data source opening))
        digitCount <
      goldilocksP ∨
    SecurityEvent covers data feProfile constants input message
      assignment challengeSetSize := by
  rcases ncTruth_or_securityEvent_of_selectedVerifierRows
      noZeroDivisors covers data feProfile constants input message
      assignment residues constantWire authority verifierRows
      challengeSetSize with
    truth | securityEvent
  · apply Or.inl
    exact splitNc_and_canonicalRows_encoded_lt_modulus
      (selectedLayout base fits) data truth source opening
      ((emittedRowsHold_iff_canonicalRowsHold
        (selectedLayout base fits) data source opening).mp canonicalRows)
  · exact Or.inr securityEvent

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningSplitNc.SelectedVerifierRefinement
