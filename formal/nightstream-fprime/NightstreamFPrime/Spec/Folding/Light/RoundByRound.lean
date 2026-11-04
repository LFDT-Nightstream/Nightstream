import NightstreamFPrime.Spec.Folding.Light.Strong
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.RoundByRound

/-! SuperNeo v1.2 Appendix B.2 for the light Π_CCS. When the light Π_CCS accepts, the output
claims open at the ambient bound, and the source claims are false (`¬ SemanticTruth`), one
challenge hit its bad set: an `α` coordinate, `γ`, or one sum-check round challenge. The bad sets
are the ones of the full fold (`PaperJoint.RoundByRound`) on the light protocol data, so their
probability bounds apply unchanged. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Light.PiCCS

open NightstreamFPrime.Spec
open _root_.NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.MatrixCoefficientSource
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.RoundByRound
  (coordinate alphaBad gammaBad roundBad mixingRoot_splits)

universe uCommitment uPublicInput

variable {Commitment : Type uCommitment} {PublicInput : Type uPublicInput}
  {shape : Shape} {columns blockCount : Nat}

open ConcreteCarrier in
/-- SuperNeo v1.2 Appendix B.2, equation (14), for the light Π_CCS. -/
theorem falseAcceptance_splits
    (openingMaps : OpeningMaps Commitment PublicInput columns)
    (params : GlobalParams)
    (statement : Statement K Commitment PublicInput shape columns blockCount baseOps)
    (constantLaw : ConstantTermLaw baseOps statement.matrixSource.kernel)
    (width : Nat)
    (degreeCovers : (statement.verifierInput K.embed).sumcheckDegreeBound <= width)
    (probe : Probe K shape)
    (witness : OutputWitness shape columns)
    (invalid : ¬ SemanticTruth ((statement.sourceConnectedInputs witness).toUnifiedInputs baseOps)
      baseOps extensionOps K.embed)
    (ambient : AmbientOutputHolds extensionOps K.embed openingMaps params statement probe witness)
    (accepted : Accepted extensionOps K.embed statement width probe) :
    let data := protocolData baseOps K.embed
      ((statement.sourceConnectedInputs witness).toUnifiedInputs baseOps)
    (∃ index, coordinate probe.coins.alpha index ∈
        alphaBad (data.toJointData extensionOps) probe.coins.alpha index) ∨
      probe.coins.gamma ∈ gammaBad (data.toJointData extensionOps) probe.coins.alpha ∨
      ∃ (certificate : FixedPhase.Certificate K width) (round : Fin shape.cubeVariables)
        (message : FixedPolynomial K width),
        FixedPhase.RawCertificate.decode width probe.response.rounds = some certificate ∧
        certificate.rounds[round.val]? = some message ∧
        coordinate probe.coins.roundPoint round ∈
          roundBad data probe.coins.alpha probe.coins.gamma
            (probe.coins.roundPoint.coordinates.take round.val) message
            (shape.cubeVariables - round.val - 1) := by
  intro data
  rcases accepted_extracts_or_badEvent baseLaws baseZeroAgreement
      GoldilocksPrime.baseFieldNoZeroDivisors extensionOps extensionLaws extensionZeroLaws K.embed
      protocolLift openingMaps params statement constantLaw width degreeCovers
      (goldilocksModulus ^ 2) probe witness ambient accepted with
    truth | mixing | failure
  · exact absurd truth invalid
  · rcases mixingRoot_splits _ _ _ mixing with alpha | gamma
    · exact Or.inl alpha
    · exact Or.inr (Or.inl gamma)
  · right
    right
    obtain ⟨certificate, decoded, bad⟩ := failure
    obtain ⟨before, challenge, after, beforeMessages, message, afterMessages,
        challengesEqual, messagesEqual, length, different, equal⟩ :=
      FixedPhase.badChallenge_implies_causal_decomposition _ _ _ _ _ certificate bad
    have total : before.length + 1 + after.length = shape.cubeVariables := by
      have dimension := probe.coins.roundPoint.dimension
      rw [challengesEqual] at dimension
      simpa [Nat.add_assoc, Nat.add_comm 1] using dimension
    let round : Fin shape.cubeVariables := ⟨before.length, by omega⟩
    refine ⟨certificate, round, message, decoded, ?_, ?_⟩
    · show certificate.rounds[before.length]? = some message
      rw [messagesEqual, length]
      simp
    · have point : coordinate probe.coins.roundPoint round = challenge := by
        simp only [coordinate, round]
        simp [challengesEqual]
      have prefixEq : probe.coins.roundPoint.coordinates.take round.val = before := by
        simp [round, challengesEqual]
      have remaining : shape.cubeVariables - round.val - 1 = after.length := by
        simp only [round]
        omega
      rw [point, prefixEq, remaining]
      exact ⟨Function.ne_iff.mp different, equal⟩

end NightstreamFPrime.Spec.Folding.Light.PiCCS
