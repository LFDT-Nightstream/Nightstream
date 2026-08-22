import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ProtocolDataRefinement
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ProtocolVerifier

/-! Provenance: copied from `formal/nightstream-lean/Nightstream/SuperNeo/Folding/PiCCS/PaperJoint/UnifiedProtocolVerifier.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespaces renamed, otherwise unchanged. -/

/-!
Semantic soundness of the transcript-bound `Pi_CCS` verifier over one
authoritative source family.

Protocol: SuperNeo `Pi_CCS` (Section 7.3 / Appendix D.4).
Phase: verifier acceptance through independent CCS, norm, and carried-
evaluation semantics.
Constraint family: semantic verifier composition only; this file emits no
rows.

Owns: composition of the actual nonlinear protocol-polynomial checker, the
derived protocol-data refinement theorem, independent joint-table truth, and
the one-source semantic predicate. Acceptance reaches that predicate or one
of the exact algebraic/transcript/output bad events already exposed by the
finite verifier theorem.

Does not own: probability bounds for the named bad events, a concrete
Poseidon2 transcript instantiation, production ring/coefficient-matrix
refinement, output CE projection, Pi_RLC handoff, Rust, R1CS, row removal, or
constraint counts.

Emits constraints: no.

Authority boundary: the theorem accepts one `UnifiedInputs`, verifier context,
round/output certificate, and explicit algebraic laws. It does not accept a
semantic-truth premise, protocol image tables, residual tables, `JointData`,
challenge vector, terminal, outgoing transcript state, or refinement equality.
Executable acceptance sees only the derived `VerifierInput`; the richer
protocol data remains confined to this semantic reduction theorem.

| Protocol | Phase | Family | Result |
|---|---|---|---|
| `Pi_CCS` | source construction | all image families | `ProtocolDataRefinement.toProtocolData` |
| `Pi_CCS` | verifier replay | alpha, gamma, SumCheck challenges | `ProtocolVerifier.check` |
| assurance | Boolean restriction | actual polynomial to independent residuals | `toProtocolData_toJointData_eq` |
| assurance | semantic closure | CCS, strict norm, carried evaluations over one source family | `check_implies_semanticTruth_or_badEvent` |
| open security | bad-event discharge | mixing root, round collision, output mismatch | explicit disjunction, no probability claim |
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedProtocolVerifier

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.SumCheck
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

universe uExtension uState

/-- Deterministic semantic soundness for the actual transcript-bound
paper-polynomial verifier over the one authoritative `K+k` source family.

This is a model-level theorem. The three bad-event branches remain explicit;
turning them into a cryptographic soundness probability requires separate
degree/cardinality and concrete-transcript theorems. -/
theorem check_implies_semanticTruth_or_badEvent
    {Extension : Type uExtension}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : Shape}
    {columns : Nat}
    (oracle : ProtocolVerifier.Oracle Extension State shape)
    (priorState : State)
    (baseOps : InterpolationOps F)
    (baseZero : NormResidualTable.BaseZeroAgreement baseOps)
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (extensionOps : InterpolationOps Extension)
    (extensionLaws : InterpolationEvaluationLaws extensionOps)
    (extensionZeroLaws : InterpolationZeroLaws extensionOps)
    (lift : F -> Extension)
    (liftLaws : ProtocolDataRefinement.ProtocolLift
      baseOps extensionOps lift)
    (data : UnifiedSources.UnifiedInputs Extension shape columns)
    (challengeSetSize : Nat)
    (certificate : ProtocolVerifier.Certificate Extension shape)
    (checked : ProtocolVerifier.check oracle priorState extensionOps
      (ProtocolDataRefinement.toProtocolData baseOps lift data).toVerifierInput
      certificate = true) :
    let protocolData :=
      ProtocolDataRefinement.toProtocolData baseOps lift data
    let execution := ProtocolVerifier.derive oracle priorState
      protocolData.toVerifierInput certificate
    data.SemanticTruth baseOps extensionOps lift \/
      SignedCoefficientObject.MixingRoot extensionOps
        (protocolData.toJointData extensionOps)
        execution.coins.alpha execution.coins.gamma \/
      (exists round,
        SumCheck.BadChallenge
          (SumCheckInitial.symbolicInstance extensionOps
            (protocolData.toJointData extensionOps)
            execution.coins.alpha execution.coins.gamma
            protocolData.toVerifierInput.sumcheckDegreeBound
            challengeSetSize execution.coins.roundPoint.coordinates
            (ProtocolPolynomial.terminalFromMessage extensionOps
              protocolData.toVerifierInput
              execution.coins.alpha execution.coins.gamma
              execution.coins.roundPoint certificate.output)
            certificate.toFinite
            (ProtocolPolynomial.canonicalExpected extensionOps protocolData
              execution.coins.alpha execution.coins.gamma
              execution.coins.roundPoint.coordinates))
          round) \/
      ProtocolPolynomial.OutputMismatch extensionOps protocolData
        execution.coins.alpha execution.coins.gamma
        execution.coins.roundPoint certificate.output := by
  let protocolData :=
    ProtocolDataRefinement.toProtocolData baseOps lift data
  rcases ProtocolVerifier.check_implies_tableTruth_or_badEvent
      oracle priorState extensionOps extensionLaws extensionZeroLaws
      protocolData challengeSetSize certificate checked with
    tableTruth | badEvent
  · apply Or.inl
    have independentTableTruth :
        (TableResidualData.toTableObligations extensionOps
          (SignedCoefficientObject.toTableResidualData extensionOps
            (data.toIndependentInputs.toJointData baseOps lift))).AllHold := by
      rw [← ProtocolDataRefinement.toProtocolData_toJointData_eq
        baseOps extensionOps lift liftLaws data]
      exact tableTruth
    have independentSemantic :=
      (ConcreteJointData.jointTableTruth_iff_semanticTruth
        baseOps baseZero noZeroDivisors extensionOps extensionLaws lift
        liftLaws.toZeroReflectingLift data.toIndependentInputs).mp
          independentTableTruth
    exact (data.toIndependentInputs_semanticTruth_iff
      baseOps extensionOps lift).mp independentSemantic
  · exact Or.inr badEvent

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedProtocolVerifier
