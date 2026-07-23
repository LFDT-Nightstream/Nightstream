import Nightstream.SuperNeo.Folding.PiCCS.PaperCorrections
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcComposition.PiDec
import Nightstream.SuperNeo.Folding.PiDEC.PaperReduction
import Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections
import Nightstream.HyperNova.NIVCCompatibility
import Nightstream.HyperNova.Construction2.Paper
import Nightstream.Protocol.FPrime.CanonicalVerifier
import Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne
import Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Minimality
import Nightstream.Protocol.FPrime.CanonicalVerifier.NifsRefinement
import Nightstream.Protocol.FPrime.CanonicalVerifier.PaperNonInteractiveNifs
import Nightstream.Protocol.FPrime.CanonicalTerminalVerifier
import Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne
import Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.Minimality
import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol
import Nightstream.Protocol.FPrime.Frozen.Obligations

/-!
Frozen paper-authoritative facade for the F-prime verification program.

Authority:

- SuperNeo Sections 4--7 and D.3--D.6;
- HyperNova Sections 3--4, 6.2--6.3, H.1, and H.3.

Owns: curated access to the frozen target propositions, the proved Pi_CCS
formula obstructions/corrections, exact deterministic `NIFS.V` graph, exact
Construction-2 `F'_j` transition, and exact terminal verifier transition.
The separate `ProductionDeviations` namespace exposes the model-proved
one-fold delayed packed-`yZcol` production deviation without changing those
paper-authoritative statements.

Does not own: old candidate NIFS semantics, legacy implementation semantics,
Rust, R1CS, artifacts, costs, or any proof of the still-open security targets.

Emits constraints: no.

The `Option` result in `NIFS.V` is the frozen reject-totalization of the
paper's deterministic verifier notation: `none` means rejection and
`some U'` is the unique computed accepted output.

| Child path | Mathematical obligation | Excluded boundary |
|---|---|---|
| `Frozen.Obligations` | fixed signatures for the SuperNeo reduction and NIFS targets | no proof or implementation premise discharges a target |
| SuperNeo paper corrections/reductions | corrected quantitative boundaries and the finite `Pi_DEC o Pi_RLC o Pi_CCS` knowledge theorem | no Fiat--Shamir or concrete primitive bound |
| `CanonicalVerifier.PaperNonInteractiveNifs` | exact paper NIFS and selected Construction-2 recursive fold | no Rust/R1CS refinement |
| `CanonicalVerifier` | executable base/recursive `F'_j` graph equals Construction 2 | no concrete NIFS semantics by itself |
| `CanonicalTerminalVerifier` | explicit base/recursive terminal relation with no final fold | no concrete relation checker by itself |
| `FixedOne` | payload-minimal one-slot step/terminal exactness and model-level inclusion-minimality | no Rust/R1CS or global arithmetization lower bound |
| `ProductionDeviations` | model-proved block/lane combined-NC and one-fold delayed packed-`yZcol` trace | no claim of paper message identity or Rust/R1CS conformance |
-/

namespace Nightstream.Protocol.FPrime.Frozen

export Obligations
  (SuperNeoGames PiCcsStrong PiRlcWeak SharedCommitmentProjection
    PiDecReductionOfKnowledge SuperNeoCompositionReductionOfKnowledge
    SuperNeoPaperObligations NifsSoundModulo NifsComplete
    NifsSoundAndCompleteModulo)

namespace ProductionDeviations

export Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol
  (Holds holds
    stateBindingCheck stateBindingCheck_eq_true_iff
    piCcsMessageCheck piCcsMessageCheck_eq_true_iff_accepted
    paperOutputCheck paperOutputCheck_eq_true_iff
    canonicalParentOpeningCheck canonicalParentOpeningCheck_eq_true_iff
    check check_eq_true_iff_accepted
    BaseAccepted baseCheck baseCheck_eq_true_iff_accepted
    PaperStepAccepted PaperRefinement
    accepted_and_packed_implies_transition_or_yRingUnbound_or_badEvent
    of_piDec_and_stateBindings
    ProjectionOpeningAccepted
    projectionOpeningAccepted_implies_packedYZcolBound_or_badEvent
    acceptedPair_of_nextPacked_implies_previousClosed_or_failure
    SharedContext CheckedStep BaseStep OutputClosed TerminalClosure Tail
    TerminalFailure EdgeFailure AllOutputsClosed Failure ClosedTrace
    closedTrace_implies_baseAndAllClosed_or_failure)

end ProductionDeviations

namespace SuperNeo

open Nightstream.SuperNeo.InteractiveReduction.Paper

/- Concrete obligation-5 theorem for the typed paper SuperNeo NIFS. -/
export Nightstream.Protocol.FPrime.CanonicalVerifier.PaperNonInteractiveNifs
  (nifsSoundAndCompleteModulo)

/- Independent valid source claims construct an accepted paper NIFS fold. -/
export Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
  (sourceValid_exists_verifiedTransition)

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment
  uWeight

/-- Headline obligation-3 theorem: the exact paper Pi_DEC verifier has the
straight-line, zero-loss reduction of knowledge from Section 7.5 / D.6. -/
theorem piDec_reductionOfKnowledge
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Weight : Type uWeight}
    (context :
      Nightstream.SuperNeo.Folding.PiDEC.PaperReduction.Context
        Structure Assignment PublicInput Point Evaluation Commitment)
    (scale : ProbabilityScale Weight) :
    ReductionOfKnowledge scale
      (Nightstream.SuperNeo.Folding.PiDEC.PaperReduction.knowledgeGame
        context scale)
      scale.zero := by
  exact Nightstream.SuperNeo.Folding.PiDEC.PaperReduction.reductionOfKnowledge
    context scale

/- Headline obligation-4 theorem for the exact finite operational
`Pi_DEC ∘ Pi_RLC ∘ Pi_CCS` composition.  The adversary supplies only
`Pi_DEC` child messages and child assignments; the combined parent is
verifier-computed, and Theorem 7 adds zero loss to the finite Theorem-6
budget. -/
export Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcComposition.PiDec
  (finiteReductionOfKnowledge)

/-- Frozen paper obstruction: strict `q / 2` does not contain every centered
Goldilocks residue, contrary to Appendix D.5's universal-coverage step. -/
theorem piRlc_literalAmbientBound_obstruction :
    ¬ Nightstream.SuperNeo.Concrete.centeredMagnitude
        Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections.midpointResidue <
      Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections.literalAmbientBound := by
  exact Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections.midpointResidue_not_literalAmbientBounded

/-- The corrected strict ambient bound `floor(q / 2) + 1` contains every
production field residue. -/
theorem piRlc_correctedAmbientBound_covers
    (value : Nightstream.SuperNeo.Concrete.F) :
    Nightstream.SuperNeo.Concrete.centeredMagnitude value <
      Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections.correctedAmbientBound := by
  exact Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections.all_centeredMagnitude_lt_correctedAmbientBound
    value

end SuperNeo

namespace HyperNova

open Nightstream.HyperNova.NonInteractiveMultiFold
open Nightstream.HyperNova.Construction2.Paper

/- Concrete no-premise Construction-2 refinement using the paper SuperNeo
NIFS verifier. -/
export Nightstream.Protocol.FPrime.CanonicalVerifier.PaperNonInteractiveNifs
  (canonicalFprime_accepts_implies_paperTransition_or_nifsBadEvent
    canonicalFprime_paperTransition_implies_exists_nifsProof_accepts)

universe uKey uRunning uFresh uProof

/- Fixed-one paper-only executable surfaces.  These exports retain the
model-level boundary: they neither refine Rust/R1CS nor claim a global
arithmetization lower bound. -/
namespace FixedOne

namespace Step

export Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne
  (eval_eq_generic accepts_iff_transition)

end Step

namespace Terminal

export Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne
  (eval_eq_generic accepts_iff_transition)

end Terminal

namespace StepMinimality

export Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Minimality
  (inclusionMinimalSound obligation8_classification)

end StepMinimality

namespace TerminalMinimality

export Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.Minimality
  (accepts_iff_transition accepts_iff_fixedOne_eval inclusionMinimalSound)

end TerminalMinimality

end FixedOne

/-- Frozen exact graph of the deterministic, one-message NIFS verifier. -/
theorem nifsV_accepts_iff
    {Key : Type uKey}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    (verifier : Verifier Key Running Fresh Proof)
    (key : Key)
    (running : Running)
    (fresh : Fresh)
    (proof : Proof)
    (output : Running) :
    Accepts verifier key running fresh proof output <->
      verifier.verify key running fresh proof = some output := by
  exact accepts_iff_verify verifier key running fresh proof output

universe uDigest uState uWitness uEncoded

/-- Headline obligation-6 equation: `F'_j` accepts exactly the independently
expanded Construction-2 transition. -/
theorem fprime_accepts_iff_transition
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    (setup : Setup Key Running Fresh Proof slotCount)
    (machine : Machine Key Digest State Witness Running Fresh Encoded slotCount)
    (functionIndex : Fin slotCount)
    (input : Input Key State Witness Running Fresh Proof slotCount)
    (output : Output Digest State Running slotCount) :
    Holds setup machine functionIndex input output <->
      Transition setup machine functionIndex input output := by
  exact holds_iff_transition setup machine functionIndex input output

/-- Headline obligation-7 equation: the compact executable verifier accepts
exactly the independently expanded Construction-2 transition. -/
theorem canonicalFprime_accepts_iff_transition
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    [DecidableEq State]
    [DecidableEq Encoded]
    (setup : Setup Key Running Fresh Proof slotCount)
    (machine : Machine Key Digest State Witness Running Fresh Encoded slotCount)
    (functionIndex : Fin slotCount)
    (input : Input Key State Witness Running Fresh Proof slotCount)
    (output : Output Digest State Running slotCount) :
    Nightstream.Protocol.FPrime.CanonicalVerifier.Accepts
        setup machine functionIndex input output <->
      Transition setup machine functionIndex input output := by
  exact Nightstream.Protocol.FPrime.CanonicalVerifier.accepts_iff_transition
    setup machine functionIndex input output

/-- Canonical augmented-function soundness against the independent selected
NIFS transition. The only admitted failure is the bad event returned by that
selected NIFS verification. -/
theorem canonicalFprime_accepts_implies_semanticTransition_or_selectedNifsBadEvent
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    [DecidableEq State]
    [DecidableEq Encoded]
    (setup : Setup Key Running Fresh Proof slotCount)
    (machine : Machine Key Digest State Witness Running Fresh Encoded slotCount)
    (nifsTransition : Key -> Running -> Fresh -> Running -> Prop)
    (nifsBadEvent : Key -> Running -> Fresh -> Proof -> Running -> Prop)
    (nifsCorrect : NifsSoundAndCompleteModulo setup.nifs
      nifsTransition nifsBadEvent)
    (functionIndex : Fin slotCount)
    (input : Input Key State Witness Running Fresh Proof slotCount)
    (output : Output Digest State Running slotCount)
    (accepted : Nightstream.Protocol.FPrime.CanonicalVerifier.Accepts
      setup machine functionIndex input output) :
    Nightstream.Protocol.FPrime.CanonicalVerifier.NifsRefinement.SemanticTransition
        setup machine nifsTransition functionIndex input output \/
      Nightstream.Protocol.FPrime.CanonicalVerifier.NifsRefinement.SelectedNifsBadEvent
        setup nifsBadEvent input output := by
  exact Nightstream.Protocol.FPrime.CanonicalVerifier.NifsRefinement.accepts_implies_semanticTransition_or_selectedNifsBadEvent
      setup machine nifsTransition nifsBadEvent nifsCorrect functionIndex input
      output accepted

/-- Honest semantic Construction-2 execution is accepted after replacing
only the single recursive NIFS proof. Every other input field is preserved. -/
theorem canonicalFprime_semanticTransition_implies_exists_nifsProof_accepts
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    [DecidableEq State]
    [DecidableEq Encoded]
    (setup : Setup Key Running Fresh Proof slotCount)
    (machine : Machine Key Digest State Witness Running Fresh Encoded slotCount)
    (nifsTransition : Key -> Running -> Fresh -> Running -> Prop)
    (nifsBadEvent : Key -> Running -> Fresh -> Proof -> Running -> Prop)
    (nifsCorrect : NifsSoundAndCompleteModulo setup.nifs
      nifsTransition nifsBadEvent)
    (functionIndex : Fin slotCount)
    (input : Input Key State Witness Running Fresh Proof slotCount)
    (output : Output Digest State Running slotCount)
    (semantic :
      Nightstream.Protocol.FPrime.CanonicalVerifier.NifsRefinement.SemanticTransition
        setup machine nifsTransition functionIndex input output) :
    exists nifsProof : Proof,
      Nightstream.Protocol.FPrime.CanonicalVerifier.Accepts setup machine
        functionIndex
        (Nightstream.Protocol.FPrime.CanonicalVerifier.NifsRefinement.withNifsProof
          input nifsProof)
        output := by
  exact Nightstream.Protocol.FPrime.CanonicalVerifier.NifsRefinement.semanticTransition_implies_exists_nifsProof_accepts
      setup machine nifsTransition nifsBadEvent nifsCorrect functionIndex input
      output semantic

universe uRunningWitness uFreshWitness

/-- Headline terminal equation: base checks only the endpoint; recursive
terminal acceptance checks all instance/witness relations and performs no
additional NIFS fold. -/
theorem terminal_accepts_iff_transition
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {RunningWitness : Type uRunningWitness}
    {Fresh : Type uFresh}
    {FreshWitness : Type uFreshWitness}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    (setup : Setup Key Running Fresh Proof slotCount)
    (machine : Machine Key Digest State Witness Running Fresh Encoded slotCount)
    (relations : TerminalRelations Key Running RunningWitness Fresh FreshWitness
      slotCount)
    (statement : TerminalStatement State)
    (proof : TerminalProof Running RunningWitness Fresh FreshWitness slotCount) :
    TerminalHolds setup machine relations statement proof <->
      TerminalTransition setup machine relations statement proof := by
  exact terminalHolds_iff_transition setup machine relations statement proof

/-- Headline obligation-7 terminal equation: the compact executable terminal
checker accepts exactly the independent Construction-2 terminal relation and
performs no final NIFS fold. -/
theorem canonicalTerminal_accepts_iff_transition
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {RunningWitness : Type uRunningWitness}
    {Fresh : Type uFresh}
    {FreshWitness : Type uFreshWitness}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    [DecidableEq State]
    [DecidableEq Encoded]
    (setup : Setup Key Running Fresh Proof slotCount)
    (machine : Machine Key Digest State Witness Running Fresh Encoded slotCount)
    (relations : TerminalRelations Key Running RunningWitness Fresh FreshWitness
      slotCount)
    (checks :
      Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
        relations)
    (statement : TerminalStatement State)
    (proof : TerminalProof Running RunningWitness Fresh FreshWitness slotCount) :
    Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.eval
        setup machine relations checks statement proof = true <->
      TerminalTransition setup machine relations statement proof := by
  exact Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.eval_eq_true_iff_transition
    setup machine relations checks statement proof

/-- Executable terminal exactness is independent of NIFS soundness and
performs no final fold. -/
theorem canonicalTerminal_exact_without_nifs
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {RunningWitness : Type uRunningWitness}
    {Fresh : Type uFresh}
    {FreshWitness : Type uFreshWitness}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    [DecidableEq State]
    [DecidableEq Encoded]
    (setup : Setup Key Running Fresh Proof slotCount)
    (machine : Machine Key Digest State Witness Running Fresh Encoded slotCount)
    (relations : TerminalRelations Key Running RunningWitness Fresh FreshWitness
      slotCount)
    (checks :
      Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
        relations)
    (statement : TerminalStatement State)
    (proof : TerminalProof Running RunningWitness Fresh FreshWitness slotCount) :
    Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.eval
        setup machine relations checks statement proof = true <->
      TerminalTransition setup machine relations statement proof := by
  exact Nightstream.Protocol.FPrime.CanonicalVerifier.NifsRefinement.terminal_exact_without_nifs
    setup machine relations checks statement proof

end HyperNova

end Nightstream.Protocol.FPrime.Frozen
