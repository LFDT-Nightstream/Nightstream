import NightstreamFPrime.Layout.Stage1.PilotNifsCompleteness
import NightstreamFPrime.Lifecycle.VerifierContext

/-!
Owns source-word preservation through the constructed pilot and C/R/D
prefixes. Every prefix uses its existing allocation interval; D source
loading uses its existing bounded write interval. No global environment
equality or new source representation is assumed.
-/

namespace NightstreamFPrime.Layout.Stage1.NifsSourceReadback

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

private theorem pilot_before_c : PilotProduction.witnessOffset ≤ PiCCSInputs.phaseOffset := by
  rw [PilotProduction.witnessOffset_eq, PiCCSInputs.phaseOffset_eq]
  decide

private theorem c_before_r : PiCCSInputs.phaseOffset ≤ PiRLCInputs.phaseOffset := by
  have bound := PiRLCInputs.piCcsLogicalFreshBase_le_phaseOffset
  unfold PiCCSStarts.logicalFreshBase at bound
  exact Nat.le_trans (Nat.le_add_right _ _) bound

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

private theorem agrees_below_pilot
    {initial : Env}
    (proof : Proof 9)
    (parentPublic : PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    (p : Sequence.Prefix initial PilotProduction.witnessOffset)
    (c : Sequence.Prefix p.current PiCCSInputs.phaseOffset)
    (r : Sequence.Prefix c.current PiRLCInputs.phaseOffset)
    (d : Sequence.Prefix (PiDECProofInputs.load r.current proof parentPublic) PiDECInputs.phaseOffset)
    (index : Nat) (below : index < PilotProduction.witnessOffset) :
    d.current index = initial index := by
  have beforeC := Nat.lt_of_lt_of_le below pilot_before_c
  have beforeR := Nat.lt_of_lt_of_le beforeC c_before_r
  have beforeDInputs := Nat.lt_of_lt_of_le beforeR
    (Nat.le_trans (Nat.le_add_right _ _) PiDECProtocolCompleteness.rEnd_before_dInputs)
  have beforeD : index < PiDECInputs.phaseOffset :=
    Nat.lt_of_lt_of_le beforeDInputs (Nat.le_add_right _ _)
  exact (d.agrees index (Or.inl beforeD)).trans
    ((PiDECProofInputs.load_agreesOutside _ _ _ index (Or.inl beforeDInputs)).trans
      ((r.agrees index (Or.inl beforeR)).trans
        ((c.agrees index (Or.inl beforeC)).trans (p.agrees index (Or.inl below)))))

/-- The actual generated prefixes preserve every word of both canonical
state-preimage intervals. This derives the bounded source agreement consumed
by typed decoder readback; it adds no caller environment-equality premise. -/
theorem words_of_prefixes
    (prior advertised : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits))
    (priorPublic : PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    (digest : Digest)
    (priorFixed : PilotProduction.FixedPreimage prior)
    (advertisedFixed : PilotProduction.FixedPreimage advertised)
    (digestFixed : digest.length = PilotProduction.digestWords)
    (values : PiCCSProofInputs.ProofValues) (context : VerifierContext.Digest4)
    (proof : Proof 9)
    (parentPublic : PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    (p : Sequence.Prefix
      (PiCCSProtocolCompleteness.environment prior priorPublic advertised digest
        priorFixed advertisedFixed digestFixed values context) PilotProduction.witnessOffset)
    (c : Sequence.Prefix p.current PiCCSInputs.phaseOffset)
    (r : Sequence.Prefix c.current PiRLCInputs.phaseOffset)
    (d : Sequence.Prefix (PiDECProofInputs.load r.current proof parentPublic) PiDECInputs.phaseOffset) :
    (∀ index : Fin PilotProduction.stateHashWords,
      d.current (PilotProduction.priorPreimageStart + index.val) =
        (serializePreimage (publicFits := publicFits) prior).getD index.val 0) ∧
    (∀ index : Fin PilotProduction.stateHashWords,
      d.current (PilotProduction.outputPreimageStart + index.val) =
        (serializePreimage (publicFits := publicFits) advertised).getD index.val 0) := by
  constructor
  · intro index
    have below : PilotProduction.priorPreimageStart + index.val < PilotProduction.witnessOffset := by
      have bound := index.isLt
      unfold PilotProduction.witnessOffset PilotProduction.externalColumnCount PilotProduction.outputDigestStart
        PilotProduction.outputPreimageStart PilotProduction.priorPublicInputStart
      omega
    exact (agrees_below_pilot proof parentPublic p c r d _ below).trans
      (PiCCSProtocolCompleteness.prior_word prior priorPublic advertised digest
        priorFixed advertisedFixed digestFixed values context index)
  · intro index
    have below : PilotProduction.outputPreimageStart + index.val < PilotProduction.witnessOffset := by
      have bound := index.isLt
      unfold PilotProduction.witnessOffset PilotProduction.externalColumnCount PilotProduction.outputDigestStart
      omega
    exact (agrees_below_pilot proof parentPublic p c r d _ below).trans
      (PiCCSProtocolCompleteness.output_word prior priorPublic advertised digest
        priorFixed advertisedFixed digestFixed values context index)

end NightstreamFPrime.Layout.Stage1.NifsSourceReadback
