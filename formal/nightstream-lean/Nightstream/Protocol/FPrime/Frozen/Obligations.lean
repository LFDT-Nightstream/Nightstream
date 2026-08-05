import Nightstream.SuperNeo.InteractiveReduction.KnowledgeComposition
import Nightstream.HyperNova.NonInteractiveMultiFold

/-!
Frozen target propositions for the paper-authoritative F-prime proof program.

Owns: only the signatures of obligations 1--5, their exact quantitative or
named-event boundaries, and the definitionally linked
`Pi_DEC ∘ Pi_RLC ∘ Pi_CCS` game required by SuperNeo Theorem 6 and
Appendix D.3.

Does not own: a concrete discharge of obligations 1--3, a concrete SuperNeo
game, a concrete NIFS verifier, Construction 2, an implementation, Rust,
R1CS, or costs.

Emits constraints: no.

The final headline theorems must instantiate these targets from the permitted
primitive contracts.  Receiving one of these targets, an extractor callback,
an arithmetization, or semantic acceptance as a premise does not discharge an
obligation.  In particular, callers cannot select an unrelated final
knowledge game: it is computed from the two explicit operational couplings.
-/

namespace Nightstream.Protocol.FPrime.Frozen.Obligations

open Nightstream.SuperNeo.InteractiveReduction.Paper
open Nightstream.HyperNova.NonInteractiveMultiFold

universe uWeight uPiCcsAdversary uPiCcsExtractor
  uPiRlcAdversary uPiRlcPairedAdversary uPiRlcExtractor
  uStrongWeakAdversary uPiDecAdversary uPiDecExtractor uComposedAdversary
  uIntermediate uProjection

/-- Independently defined quantitative games and shared projection for the
three SuperNeo reductions.  This is game data, not evidence that a target
holds. -/
structure SuperNeoGames where
  Weight : Type uWeight
  scale : ProbabilityScale Weight

  PiCcsAdversary : Type uPiCcsAdversary
  PiCcsExtractor : Type uPiCcsExtractor
  piCcs : StrongGame Weight PiCcsAdversary PiCcsExtractor

  PiRlcAdversary : Type uPiRlcAdversary
  PiRlcPairedAdversary : Type uPiRlcPairedAdversary
  PiRlcExtractor : Type uPiRlcExtractor
  piRlc : WeakGame Weight PiRlcAdversary PiRlcPairedAdversary PiRlcExtractor

  StrongWeakAdversary : Type uStrongWeakAdversary
  strongWeakCoupling :
    Nightstream.SuperNeo.InteractiveReduction.StrongWeakComposition.Coupling
      scale piCcs piRlc StrongWeakAdversary

  PiDecAdversary : Type uPiDecAdversary
  PiDecExtractor : Type uPiDecExtractor
  piDec : KnowledgeGame Weight PiDecAdversary PiDecExtractor

  ComposedAdversary : Type uComposedAdversary
  piDecCoupling :
    Nightstream.SuperNeo.InteractiveReduction.KnowledgeComposition.Coupling
      scale
      (Nightstream.SuperNeo.InteractiveReduction.StrongWeakComposition.knowledgeGame
        scale piCcs piRlc strongWeakCoupling)
      piDec ComposedAdversary

  IntermediateInstance : Type uIntermediate
  Projection : Type uProjection
  piCcsProjection : IntermediateInstance -> Projection
  piRlcProjection : IntermediateInstance -> Projection

  errorBudget : InteractiveErrorBudget Weight
  scaleLaws :
    Nightstream.SuperNeo.InteractiveReduction.StrongWeakComposition.ScaleLaws
      scale

/-- Theorem-6 game is computed from the exact strong--weak operational
coupling. -/
def strongWeakKnowledgeGame (games : SuperNeoGames) :=
  Nightstream.SuperNeo.InteractiveReduction.StrongWeakComposition.knowledgeGame
    games.scale games.piCcs games.piRlc games.strongWeakCoupling

/-- The full SuperNeo game is computed by composing the Theorem-6 game with
the exact `Pi_DEC` operational coupling. -/
def superNeoCompositionGame (games : SuperNeoGames) :=
  Nightstream.SuperNeo.InteractiveReduction.KnowledgeComposition.knowledgeGame
    games.scale (strongWeakKnowledgeGame games) games.piDec
    games.piDecCoupling

/-- Obligation 1: Pi_CCS is success-gated strong with intrinsic SumCheck and
Schwartz--Zippel losses. Pi_RLC supplies the raw repeated-output witness
disagreement bound; Pi_CCS charges its root envelope once. -/
def PiCcsStrong (games : SuperNeoGames) : Prop :=
  SuccessGatedStrong games.scale games.piCcs
    (games.scale.add games.errorBudget.piCcsSumCheck
      games.errorBudget.piCcsSchwartzZippel)
    games.errorBudget.relaxedBindingRaw
    games.errorBudget.relaxedBindingRoot

/-- Obligation 2: Pi_RLC is weak with the fork-sampling loss and relaxed
binding as its exact witness-uniqueness boundary. -/
def PiRlcWeak (games : SuperNeoGames) : Prop :=
  Weak games.scale games.piRlc
    games.errorBudget.piRlcForkSampling
    games.errorBudget.relaxedBindingRaw

/-- Theorem 6 requires literal equality of the two commitment projections. -/
def SharedCommitmentProjection (games : SuperNeoGames) : Prop :=
  games.piCcsProjection = games.piRlcProjection

/-- Obligation 3: Pi_DEC is a zero-loss reduction of knowledge. -/
def PiDecReductionOfKnowledge (games : SuperNeoGames) : Prop :=
  ReductionOfKnowledge games.scale games.piDec games.scale.zero

/-- Obligation 4: the complete composition is a reduction of knowledge with
SumCheck, Schwartz--Zippel, fork-sampling, and the one adjusted binding loss.
The raw binding error is not charged a second time. -/
def SuperNeoCompositionReductionOfKnowledge (games : SuperNeoGames) : Prop :=
  ReductionOfKnowledge games.scale (superNeoCompositionGame games)
    (games.errorBudget.total games.scale)

/-- The linked final target follows from obligations 1--3.  There is no
independent final-game premise: both composed games are definitionally
computed from the couplings stored in `games`.  The operational same-`phi`
premise is already a field of `strongWeakCoupling`; literal equality of the
two declared projection functions remains a separate frozen obligation. -/
theorem superNeoCompositionReductionOfKnowledge
    (games : SuperNeoGames)
    (piCcs : PiCcsStrong games)
    (piRlc : PiRlcWeak games)
    (piDec : PiDecReductionOfKnowledge games) :
    SuperNeoCompositionReductionOfKnowledge games := by
  apply
    Nightstream.SuperNeo.InteractiveReduction.KnowledgeComposition.reductionOfKnowledge
      games.scale games.scaleLaws
      (strongWeakKnowledgeGame games) games.piDec games.piDecCoupling
      (games.errorBudget.strongWeakTotal games.scale) games.scale.zero
  · exact
      Nightstream.SuperNeo.InteractiveReduction.StrongWeakComposition.reductionOfKnowledge
        games.scale games.scaleLaws games.piCcs games.piRlc
        games.strongWeakCoupling
        (games.scale.add games.errorBudget.piCcsSumCheck
          games.errorBudget.piCcsSchwartzZippel)
        games.errorBudget.piRlcForkSampling
        games.errorBudget.relaxedBindingRaw
        games.errorBudget.relaxedBindingRoot piCcs piRlc
  · exact piDec

/-- Complete frozen SuperNeo target; no generic bad event or final `negl` term
is available. -/
def SuperNeoPaperObligations (games : SuperNeoGames) : Prop :=
  PiCcsStrong games /\
  PiRlcWeak games /\
  SharedCommitmentProjection games /\
  PiDecReductionOfKnowledge games /\
  SuperNeoCompositionReductionOfKnowledge games

/-- Assemble the complete frozen SuperNeo target once the three component
reductions and their literal shared projection have been established. -/
theorem superNeoPaperObligations_of_components
    (games : SuperNeoGames)
    (piCcs : PiCcsStrong games)
    (piRlc : PiRlcWeak games)
    (sharedProjection : SharedCommitmentProjection games)
    (piDec : PiDecReductionOfKnowledge games) :
    SuperNeoPaperObligations games :=
  ⟨piCcs, piRlc, sharedProjection, piDec,
    superNeoCompositionReductionOfKnowledge games piCcs piRlc piDec⟩

universe uKey uRunning uFresh uProof uNifsWeight uNifsOutcome

/-- Deterministic core of obligation 5 for a concrete one-message NIFS
verifier and independently defined paper transition.  This is a pointwise
cover only; it does not state the non-interactive probability bound. -/
def NifsSoundModulo
    {Key : Type uKey}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    (verifier : Verifier Key Running Fresh Proof)
    (transition : Key -> Running -> Fresh -> Running -> Prop)
    (badEvent : Key -> Running -> Fresh -> Proof -> Running -> Prop) : Prop :=
  forall key running fresh proof output,
    Accepts verifier key running fresh proof output ->
      transition key running fresh output \/
      badEvent key running fresh proof output

/-- Obligation-5 honest completeness target. -/
def NifsComplete
    {Key : Type uKey}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    (verifier : Verifier Key Running Fresh Proof)
    (transition : Key -> Running -> Fresh -> Running -> Prop) : Prop :=
  forall key running fresh output,
    transition key running fresh output ->
      exists proof, Accepts verifier key running fresh proof output

/-- Deterministic soundness-and-completeness core of obligation 5. The
bad-event predicate must be instantiated by a closed finite type of
protocol/security events, never `refinementFailure` or `outputUnbound`.

The quantitative non-interactive target remains
`NifsNonInteractiveSound`; this conjunction alone is not the full security
obligation. -/
def NifsSoundAndCompleteModulo
    {Key : Type uKey}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    (verifier : Verifier Key Running Fresh Proof)
    (transition : Key -> Running -> Fresh -> Running -> Prop)
    (badEvent : Key -> Running -> Fresh -> Proof -> Running -> Prop) : Prop :=
  NifsSoundModulo verifier transition badEvent /\
  NifsComplete verifier transition

/-- Quantitative non-interactive soundness target for obligation 5.

The accepted and transition predicates must be instantiated by the actual
protocol experiment.  This definition owns only the paper-shaped subtractive
inequality; choosing unrelated predicates or supplying this proposition as a
premise is not evidence that a concrete NIFS satisfies it. -/
def NifsNonInteractiveSound
    {Weight : Type uNifsWeight}
    {Outcome : Type uNifsOutcome}
    (scale : ProbabilityScale Weight)
    (experiment : ProbabilityExperiment scale Outcome)
    (accepted transition : Outcome -> Prop)
    (error : Weight) : Prop :=
  scale.le
    (scale.subtract (experiment.probability accepted) error)
    (experiment.probability transition)

end Nightstream.Protocol.FPrime.Frozen.Obligations
