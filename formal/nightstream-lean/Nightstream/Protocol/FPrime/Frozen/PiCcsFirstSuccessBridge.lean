import Nightstream.Protocol.FPrime.Frozen.Obligations
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.AsymptoticPaperStrong

/-!
Exact frozen-facade bridge for the unbounded PiCCS first-success reduction.

Owns: completion of the operational asymptotic PiCCS game into the frozen
`SuperNeoGames` carrier and the literal `PiCcsStrong family.games` theorem.

Does not own: proofs of PiRLC weakness, PiDEC knowledge reduction, either
composition coupling, Fiat--Shamir, Rust, R1CS, or constraints. Those objects
are carried only so `family.games` has the exact frozen type; they are not
used as premises of the PiCCS proof.

The game, sampler law, runtime predicate, success floor, raw mismatch event,
and conditioning adjustment are definitionally linked. In particular, no
free retry sequence or opaque runtime proposition is accepted.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.FPrime.Frozen.PiCcsFirstSuccessBridge

open Nightstream.Protocol.FPrime.Frozen.Obligations
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.AsymptoticPaperStrong
open Nightstream.SuperNeo.InteractiveReduction.Asymptotic
open Nightstream.SuperNeo.InteractiveReduction.Paper

universe uPiRlcAdversary uPiRlcPairedAdversary uPiRlcExtractor
  uStrongWeakAdversary uPiDecAdversary uPiDecExtractor
  uComposedAdversary uIntermediate uProjection

/-- Downstream game data needed only to inhabit the exact frozen
`SuperNeoGames` carrier. The PiCCS game is fixed by `core`; callers cannot
replace it with an unrelated strong game. -/
structure Completion (core : Family) where
  PiRlcAdversary : Type uPiRlcAdversary
  PiRlcPairedAdversary : Type uPiRlcPairedAdversary
  PiRlcExtractor : Type uPiRlcExtractor
  piRlc :
    WeakGame Weight PiRlcAdversary PiRlcPairedAdversary PiRlcExtractor

  StrongWeakAdversary : Type uStrongWeakAdversary
  strongWeakCoupling :
    Nightstream.SuperNeo.InteractiveReduction.StrongWeakComposition.Coupling
      Nightstream.SuperNeo.InteractiveReduction.Asymptotic.scale
      (strongGame core) piRlc StrongWeakAdversary

  PiDecAdversary : Type uPiDecAdversary
  PiDecExtractor : Type uPiDecExtractor
  piDec : KnowledgeGame Weight PiDecAdversary PiDecExtractor

  ComposedAdversary : Type uComposedAdversary
  piDecCoupling :
    Nightstream.SuperNeo.InteractiveReduction.KnowledgeComposition.Coupling
      Nightstream.SuperNeo.InteractiveReduction.Asymptotic.scale
      (Nightstream.SuperNeo.InteractiveReduction.StrongWeakComposition.knowledgeGame
        Nightstream.SuperNeo.InteractiveReduction.Asymptotic.scale
        (strongGame core) piRlc strongWeakCoupling)
      piDec ComposedAdversary

  IntermediateInstance : Type uIntermediate
  Projection : Type uProjection
  piCcsProjection : IntermediateInstance -> Projection
  piRlcProjection : IntermediateInstance -> Projection

/-- Complete security family whose PiCCS component and exact error terms are
owned by the operational first-success construction. -/
structure PiCcsSecurityFamily where
  core : Family
  sumCheckBudget : Weight
  schwartzZippelBudget : Weight
  piRlcForkSamplingBudget : Weight
  relaxedBindingRaw : Weight
  completion : Completion core

/-- Exact pointwise `delta / mu` conditioning adjustment. -/
def adjustUniqueness
    (raw floor : Weight) : Weight :=
  fun securityParameter =>
    raw securityParameter / floor securityParameter

/-- Frozen error owner with exact paper names and ordering. -/
def PiCcsSecurityFamily.errorBudget
    (family : PiCcsSecurityFamily) :
    InteractiveErrorBudget Weight where
  piCcsSumCheck := family.sumCheckBudget
  piCcsSchwartzZippel := family.schwartzZippelBudget
  piRlcForkSampling := family.piRlcForkSamplingBudget
  piCcsSuccessFloor := family.core.successFloor
  relaxedBindingRaw := family.relaxedBindingRaw
  adjustUniqueness := adjustUniqueness

/-- The exact frozen game carrier. PiCCS is definitionally the operational
unbounded first-success game. -/
noncomputable def PiCcsSecurityFamily.games
    (family : PiCcsSecurityFamily) :
    SuperNeoGames where
  Weight := Weight
  scale := Nightstream.SuperNeo.InteractiveReduction.Asymptotic.scale

  PiCcsAdversary := family.core.Adversary
  PiCcsExtractor := Extractor
  piCcs := strongGame family.core

  PiRlcAdversary := family.completion.PiRlcAdversary
  PiRlcPairedAdversary := family.completion.PiRlcPairedAdversary
  PiRlcExtractor := family.completion.PiRlcExtractor
  piRlc := family.completion.piRlc

  StrongWeakAdversary := family.completion.StrongWeakAdversary
  strongWeakCoupling := family.completion.strongWeakCoupling

  PiDecAdversary := family.completion.PiDecAdversary
  PiDecExtractor := family.completion.PiDecExtractor
  piDec := family.completion.piDec

  ComposedAdversary := family.completion.ComposedAdversary
  piDecCoupling := family.completion.piDecCoupling

  IntermediateInstance := family.completion.IntermediateInstance
  Projection := family.completion.Projection
  piCcsProjection := family.completion.piCcsProjection
  piRlcProjection := family.completion.piRlcProjection

  errorBudget := family.errorBudget
  scaleLaws :=
    Nightstream.SuperNeo.InteractiveReduction.Asymptotic.scaleLaws

/-- The exact frozen obligation follows from the operational unbounded
first-success construction and the two permitted fixed-witness algebraic
contracts.

Almost-sure termination, EPT, conditioned-law equality, fresh-second
independence, and the extraction inequality are derived facts about
`family.core`; none is a premise here. -/
theorem piCcsStrong_of_unboundedFirstSuccess
    (family : PiCcsSecurityFamily)
    (contracts :
      NamedSecurityContracts family.core
        family.sumCheckBudget family.schwartzZippelBudget) :
    PiCcsStrong family.games := by
  exact paperStrong family.core
    family.sumCheckBudget family.schwartzZippelBudget
    family.relaxedBindingRaw contracts

end Nightstream.Protocol.FPrime.Frozen.PiCcsFirstSuccessBridge
