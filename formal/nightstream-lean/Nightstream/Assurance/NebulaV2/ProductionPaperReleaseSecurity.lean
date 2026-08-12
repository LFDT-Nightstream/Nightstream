import Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline
import Nightstream.Assurance.NebulaV2.SecurityBudget

/-!
Contract: conditional end-to-end probability theorem for the exact
production-paper Nebula-on-SuperNeo V2 release path.

This module classifies every public `BadEvent`. Deterministic implementation
obligations must be impossible. Computational events are assigned to six
explicit game families. The proved union bound is exactly the rational total
from `SecurityBudget`; no event is placed in an unnamed remainder.

The final theorem is conditional. It does not prove a decoder, generated-row
refinement, application-port refinement, recursive-size closure, cryptographic
reduction, or terminal backend. Concrete release artifacts must supply those
premises. None of the premises contains `HasSoundExecution` or an execution
witness.

Assurance tier: security-reduction composition boundary.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Assurance.NebulaV2.ProductionPaperReleaseSecurity

open Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline
open Nightstream.Assurance.NebulaV2.SecurityBudget
open Nightstream.Assurance.NebulaV2.SeededSetupSecurity
open Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime
open Nightstream.Implementation.NebulaV2.SeedSchedule
open Nightstream.Protocol.NebulaV2.Soundness

/-- The probability laws used by the release union bound. Rationals are used
to match the executable security budget. A concrete security game must prove
these laws for its distribution. -/
structure ProbabilityModel (Outcome : Type) where
  probability : (Outcome → Prop) → ℚ
  nonnegative : ∀ event, 0 ≤ probability event
  impossible : probability (fun _ => False) = 0
  certain : probability (fun _ => True) = 1
  monotone : ∀ {left right : Outcome → Prop},
    (∀ outcome, left outcome → right outcome) →
      probability left ≤ probability right
  unionBound : ∀ (left right : Outcome → Prop),
    probability (fun outcome => left outcome ∨ right outcome) ≤
      probability left + probability right

/-- These six events are deterministic release obligations. Giving any of
them a negligible probability would hide a parser, relation, or lifecycle
defect. -/
structure DeterministicClosure
    {Outcome : Type} (occurs : BadEvent → Outcome → Prop) : Prop where
  decode : ∀ outcome, ¬ occurs .decode outcome
  circuitRefinement : ∀ outcome, ¬ occurs .circuitRefinement outcome
  applicationPortCoverage : ∀ outcome, ¬ occurs .applicationPortCoverage outcome
  fPrimeLifecycle : ∀ outcome, ¬ occurs .fPrimeLifecycle outcome
  recursiveSizeClosure : ∀ outcome, ¬ occurs .recursiveSizeClosure outcome
  bundlePropagation : ∀ outcome, ¬ occurs .bundlePropagation outcome

/-- One seeded-matrix game covers the full commitment, the compact-token
commitments, and replacement of verifier-key-seeded matrices by uniform
matrices. -/
def SetupFamily
    {Outcome : Type} (occurs : BadEvent → Outcome → Prop)
    (outcome : Outcome) : Prop :=
  occurs .commitmentBinding outcome ∨
    occurs .compactTokenBinding outcome ∨
    occurs .seededSetup outcome

/-- Complete computational failure family after deterministic obligations
are closed. The right-associated form matches the union-bound proof. -/
def ComputationalAny
    {Outcome : Type} (occurs : BadEvent → Outcome → Prop)
    (outcome : Outcome) : Prop :=
  SetupFamily occurs outcome ∨
    occurs .poseidonOrTranscript outcome ∨
    occurs .memoryFingerprint outcome ∨
    occurs .piRlcSampler outcome ∨
    occurs .foldExtraction outcome ∨
    occurs .terminalBackend outcome

/-- The six deterministic events and eight computational events exhaust the
public bad-event register. This proof is by constructor inspection. -/
theorem anyBad_implies_computationalAny
    {Outcome : Type} {occurs : BadEvent → Outcome → Prop}
    (closed : DeterministicClosure occurs) {outcome : Outcome}
    (failure : AnyBad (fun event => occurs event outcome)) :
    ComputationalAny occurs outcome := by
  rcases failure with ⟨event, evidence⟩
  cases event with
  | decode => exact False.elim (closed.decode outcome evidence)
  | circuitRefinement =>
      exact False.elim (closed.circuitRefinement outcome evidence)
  | applicationPortCoverage =>
      exact False.elim (closed.applicationPortCoverage outcome evidence)
  | fPrimeLifecycle =>
      exact False.elim (closed.fPrimeLifecycle outcome evidence)
  | recursiveSizeClosure =>
      exact False.elim (closed.recursiveSizeClosure outcome evidence)
  | bundlePropagation =>
      exact False.elim (closed.bundlePropagation outcome evidence)
  | commitmentBinding =>
      exact Or.inl (Or.inl evidence)
  | compactTokenBinding =>
      exact Or.inl (Or.inr (Or.inl evidence))
  | seededSetup =>
      exact Or.inl (Or.inr (Or.inr evidence))
  | poseidonOrTranscript =>
      exact Or.inr (Or.inl evidence)
  | memoryFingerprint =>
      exact Or.inr (Or.inr (Or.inl evidence))
  | piRlcSampler =>
      exact Or.inr (Or.inr (Or.inr (Or.inl evidence)))
  | foldExtraction =>
      exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl evidence))))
  | terminalBackend =>
      exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr evidence))))

/-- Every computational failure names one public `BadEvent`. -/
theorem computationalAny_implies_anyBad
    {Outcome : Type} {occurs : BadEvent → Outcome → Prop}
    {outcome : Outcome} (failure : ComputationalAny occurs outcome) :
    AnyBad (fun event => occurs event outcome) := by
  rcases failure with
    setup | poseidon | fingerprint | sampler | fold | terminal
  · rcases setup with commitment | token | seeded
    · exact ⟨.commitmentBinding, commitment⟩
    · exact ⟨.compactTokenBinding, token⟩
    · exact ⟨.seededSetup, seeded⟩
  · exact ⟨.poseidonOrTranscript, poseidon⟩
  · exact ⟨.memoryFingerprint, fingerprint⟩
  · exact ⟨.piRlcSampler, sampler⟩
  · exact ⟨.foldExtraction, fold⟩
  · exact ⟨.terminalBackend, terminal⟩

/-- Exact game-to-register obligations. Each field bounds one disjointly
named family in `ComputationalAny`. The setup field is the missing bridge from
the three setup-family events to the seven-role seeded Module-SIS hybrid.
The fold field combines the SuperNeo field, coordinate-fork, and NIFS
common-witness terms. The PiRLC field is a sampler-distribution or
native/circuit-agreement soundness game. It is not the honest-abort
completeness probability. -/
structure ComputationalBounds
    {Outcome : Type}
    (model : ProbabilityModel Outcome)
    (occurs : BadEvent → Outcome → Prop)
    {manifest : Manifest}
    (setup : HybridAssumption manifest)
    (additional : AdditionalBounds) : Prop where
  setupFamily :
    model.probability (SetupFamily occurs) ≤ setup.totalSeededAdvantage
  poseidonTranscriptAndFiatShamir :
    model.probability (occurs .poseidonOrTranscript) ≤
      additional.poseidonAndTranscript + additional.fiatShamirProgramming
  memoryFingerprint :
    model.probability (occurs .memoryFingerprint) ≤ fingerprintBound
  piRlcSampler :
    model.probability (occurs .piRlcSampler) ≤
      additional.piRlcSamplerDistribution
  foldExtraction :
    model.probability (occurs .foldExtraction) ≤
      fieldBound + coordinateForkBound + additional.nifsCommonWitness
  terminalBackend :
    model.probability (occurs .terminalBackend) ≤ additional.compactTerminal

/-- The probability of any computational event is at most the exact complete
budget. The equality at the end shows that no additional event term is
silently discarded. -/
theorem computationalAny_probability_le_total
    {Outcome : Type}
    {model : ProbabilityModel Outcome}
    {occurs : BadEvent → Outcome → Prop}
    {manifest : Manifest}
    {setup : HybridAssumption manifest}
    {additional : AdditionalBounds}
    (bounds : ComputationalBounds model occurs setup additional) :
    model.probability (ComputationalAny occurs) ≤
      total setup.totalSeededAdvantage additional := by
  have foldTerminal :
      model.probability (fun outcome =>
        occurs .foldExtraction outcome ∨ occurs .terminalBackend outcome) ≤
      (fieldBound + coordinateForkBound + additional.nifsCommonWitness) +
        additional.compactTerminal :=
    (model.unionBound (occurs .foldExtraction)
      (occurs .terminalBackend)).trans
      (add_le_add bounds.foldExtraction bounds.terminalBackend)
  have samplerTail :
      model.probability (fun outcome =>
        occurs .piRlcSampler outcome ∨
          occurs .foldExtraction outcome ∨ occurs .terminalBackend outcome) ≤
      additional.piRlcSamplerDistribution +
        ((fieldBound + coordinateForkBound + additional.nifsCommonWitness) +
          additional.compactTerminal) :=
    (model.unionBound (occurs .piRlcSampler)
      (fun outcome =>
        occurs .foldExtraction outcome ∨
          occurs .terminalBackend outcome)).trans
      (add_le_add bounds.piRlcSampler foldTerminal)
  have fingerprintTail :
      model.probability (fun outcome =>
        occurs .memoryFingerprint outcome ∨
          occurs .piRlcSampler outcome ∨
          occurs .foldExtraction outcome ∨ occurs .terminalBackend outcome) ≤
      fingerprintBound +
        (additional.piRlcSamplerDistribution +
          ((fieldBound + coordinateForkBound + additional.nifsCommonWitness) +
            additional.compactTerminal)) :=
    (model.unionBound (occurs .memoryFingerprint)
      (fun outcome =>
        occurs .piRlcSampler outcome ∨
          occurs .foldExtraction outcome ∨
          occurs .terminalBackend outcome)).trans
      (add_le_add bounds.memoryFingerprint samplerTail)
  have poseidonTail :
      model.probability (fun outcome =>
        occurs .poseidonOrTranscript outcome ∨
          occurs .memoryFingerprint outcome ∨
          occurs .piRlcSampler outcome ∨
          occurs .foldExtraction outcome ∨ occurs .terminalBackend outcome) ≤
      (additional.poseidonAndTranscript +
          additional.fiatShamirProgramming) +
        (fingerprintBound +
          (additional.piRlcSamplerDistribution +
            ((fieldBound + coordinateForkBound + additional.nifsCommonWitness) +
              additional.compactTerminal))) :=
    (model.unionBound (occurs .poseidonOrTranscript)
      (fun outcome =>
        occurs .memoryFingerprint outcome ∨
          occurs .piRlcSampler outcome ∨
          occurs .foldExtraction outcome ∨
          occurs .terminalBackend outcome)).trans
      (add_le_add bounds.poseidonTranscriptAndFiatShamir fingerprintTail)
  have allFamilies :
      model.probability (ComputationalAny occurs) ≤
      setup.totalSeededAdvantage +
        ((additional.poseidonAndTranscript +
            additional.fiatShamirProgramming) +
          (fingerprintBound +
            (additional.piRlcSamplerDistribution +
              ((fieldBound + coordinateForkBound +
                  additional.nifsCommonWitness) +
                additional.compactTerminal)))) := by
    unfold ComputationalAny
    exact
      (model.unionBound (SetupFamily occurs)
        (fun outcome =>
          occurs .poseidonOrTranscript outcome ∨
            occurs .memoryFingerprint outcome ∨
            occurs .piRlcSampler outcome ∨
            occurs .foldExtraction outcome ∨
            occurs .terminalBackend outcome)).trans
        (add_le_add bounds.setupFamily poseidonTail)
  calc
    model.probability (ComputationalAny occurs) ≤
        setup.totalSeededAdvantage +
          ((additional.poseidonAndTranscript +
              additional.fiatShamirProgramming) +
            (fingerprintBound +
              (additional.piRlcSamplerDistribution +
                ((fieldBound + coordinateForkBound +
                    additional.nifsCommonWitness) +
                  additional.compactTerminal)))) := allFamilies
    _ = total setup.totalSeededAdvantage additional := by
      unfold total AdditionalBounds.total
      ring

/-- False acceptance for one security-game outcome. The semantic negation is
part of the event being bounded; it is not an extraction premise. -/
def FalseAcceptance
    {Outcome Bytes Parsed Program : Type}
    (proof : Outcome → Bytes)
    (decode : Bytes → Option Parsed)
    (terminalAccepts : Parsed → Prop)
    (context : Context Program)
    (outcome : Outcome) : Prop :=
  Accepts decode terminalAccepts (proof outcome) ∧
    ¬ HasSoundExecution context.machine.semantics context.statement.base
      context.snapshotRoot

/-- Pointwise release reduction. A false acceptance can only enter the
computational family after all deterministic obligations are proved. -/
theorem falseAcceptance_implies_computationalAny
    {Outcome Bytes Parsed Program : Type}
    {proof : Outcome → Bytes}
    {decode : Bytes → Option Parsed}
    {terminalAccepts : Parsed → Prop}
    {occurs : BadEvent → Outcome → Prop}
    {context : Context Program}
    (boundary : ∀ outcome,
      StagedExtraction decode terminalAccepts
        (fun event => occurs event outcome) context)
    (accounting : ∀ outcome,
      LifetimeFailureAccounting (fun event => occurs event outcome) context)
    (closed : DeterministicClosure occurs)
    {outcome : Outcome}
    (falseAcceptance :
      FalseAcceptance proof decode terminalAccepts context outcome) :
    ComputationalAny occurs outcome := by
  rcases falseAcceptance with ⟨accepted, noExecution⟩
  rcases
      acceptance_under_staged_refinement_implies_any_bad_or_execution
        (boundary outcome) (accounting outcome) accepted with
    failure | execution
  · exact anyBad_implies_computationalAny closed failure
  · exact False.elim (noExecution execution)

/-- Conditional false-acceptance bound for one exact production context. All
implementation refinement and computational game premises remain visible. -/
theorem falseAcceptance_probability_lt_target96
    {Outcome Bytes Parsed Program : Type}
    {model : ProbabilityModel Outcome}
    {proof : Outcome → Bytes}
    {decode : Bytes → Option Parsed}
    {terminalAccepts : Parsed → Prop}
    {occurs : BadEvent → Outcome → Prop}
    {context : Context Program}
    {manifest : Manifest}
    {setup : HybridAssumption manifest}
    {additional : AdditionalBounds}
    (boundary : ∀ outcome,
      StagedExtraction decode terminalAccepts
        (fun event => occurs event outcome) context)
    (accounting : ∀ outcome,
      LifetimeFailureAccounting (fun event => occurs event outcome) context)
    (closed : DeterministicClosure occurs)
    (bounds : ComputationalBounds model occurs setup additional) :
    model.probability
        (FalseAcceptance proof decode terminalAccepts context) < target96 := by
  have reduced :
      model.probability
          (FalseAcceptance proof decode terminalAccepts context) ≤
        model.probability (ComputationalAny occurs) :=
    model.monotone fun outcome failure =>
      falseAcceptance_implies_computationalAny boundary accounting closed failure
  have budgeted :
      model.probability (ComputationalAny occurs) ≤
        total setup.totalSeededAdvantage additional :=
    computationalAny_probability_le_total bounds
  have setupBound :
      setup.totalSeededAdvantage < setupRequirement127 := by
    simpa [setupRequirement127, postUnionBits] using setup.total_lt_post_union
  have target :=
    (total_lt_target96 setup.total_nonnegative setupBound additional).2
  exact lt_of_le_of_lt (reduced.trans budgeted) target

/-- Generated-artifact specialization. The context, verifier-key identity,
relation programs, and terminal program come from one `GeneratedContext`. -/
theorem generated_falseAcceptance_probability_lt_target96
    {Outcome Bytes Parsed Program : Type}
    {model : ProbabilityModel Outcome}
    {proof : Outcome → Bytes}
    {decode : Bytes → Option Parsed}
    {terminalAccepts : Parsed → Prop}
    {occurs : BadEvent → Outcome → Prop}
    {generated : GeneratedContext Program}
    {manifest : Manifest}
    {setup : HybridAssumption manifest}
    {additional : AdditionalBounds}
    (boundary : ∀ outcome,
      StagedExtraction decode terminalAccepts
        (fun event => occurs event outcome) generated.context)
    (accounting : ∀ outcome,
      LifetimeFailureAccounting (fun event => occurs event outcome)
        generated.context)
    (closed : DeterministicClosure occurs)
    (bounds : ComputationalBounds model occurs setup additional) :
    model.probability
        (FalseAcceptance proof decode terminalAccepts generated.context) <
      target96 :=
  falseAcceptance_probability_lt_target96 boundary accounting closed bounds

end Nightstream.Assurance.NebulaV2.ProductionPaperReleaseSecurity
