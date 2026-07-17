import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.DerivedPiDec.Checker
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Evaluator
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Canonical.RunningAuthority
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Checker

/-!
Executable raw-certificate checker for the canonical fixed-active NIFS.

Protocol: fixed SuperNeo NIFS `CE^14 x CCS -> CE^14`.
Phase: incoming checked parent, Split-NC `Pi_CCS`, `Pi_RLC` sampling, and
outgoing `Pi_DEC` recomposition.
Constraint family: logical composition only; this file emits no rows.

Owns: one fail-closed Boolean composition of all retained physical verifier
families; exact equivalence with independent `ConcretePhi81.Accepted`; and the
concrete checker instance consumed by the canonical evaluator.

Does not own: semantic source openings, output truth, bad-event probability,
Poseidon2/Rust transcript refinement, R1CS decoding, physical rows, costs,
necessity, or row removal.

Emits constraints: no.

Authority boundary: the checker receives only the canonical public context
and raw phase messages. Incoming parent structure/stages/presence and outgoing
`Pi_RLC` structure/stages/parent fields are computed or proved from that
carrier. Every retained finite comparison is delegated to its exact child
checker; no digest is promoted to authority.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.fixed_active.running_authority` | four-family incoming checked-parent authority | checked | `Canonical.RunningAuthority.check` |
| `nifs.fixed_active.pi_ccs` | exact FE-to-NC raw transcript acceptance | checked | `Protocol.check` |
| `nifs.fixed_active.pi_rlc.sampler` | exact 54-of-64 challenge derivation and binding | checked/computed | `Sampler.Checker.certificateCheck` |
| `nifs.fixed_active.pi_rlc.structure` | all sources use the verifier-owned relation structure | derived/eliminated | `Canonical.Context.sourceStructures` |
| `nifs.fixed_active.pi_dec` | three-family outgoing recomposition | checked | `DerivedPiDec.Checker.check` |
| `nifs.fixed_active.checker.exact` | Boolean composition iff independent physical NIFS acceptance | exact model theorem | `check_eq_true_iff_accepted` |
| `nifs.fixed_active.evaluator` | instantiate the fail-closed evaluator with the proved checker | refinement instance | `evaluatorChecker` |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Canonical.Checker

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uState

/-- Execute all and only the retained physical verifier families. -/
def check
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context :
      Canonical.Context shape domain State publicRingColumns publicFits
        verifierRows)
    (certificate : FixedActive.Certificate context.materialize) : Bool :=
  Canonical.RunningAuthority.check context &&
    (Protocol.check context.materialize.feMachine context.materialize.ncMachine
        context.materialize.initialState context.profile context.piCcsInput
        context.materialize.feCoins context.materialize.ncCoins
        certificate.piCcs &&
      (Sampler.Checker.certificateCheck context.materialize certificate &&
        DerivedPiDec.Checker.check context.materialize certificate))

/-- The raw Boolean composition accepts exactly the independently defined
physical NIFS verifier predicate. -/
theorem check_eq_true_iff_accepted
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context :
      Canonical.Context shape domain State publicRingColumns publicFits
        verifierRows)
    (certificate : FixedActive.Certificate context.materialize) :
    check context certificate = true <->
      ConcretePhi81.Accepted context.materialize certificate := by
  simp only [check, Bool.and_eq_true,
    Canonical.RunningAuthority.check_eq_true_iff_accepted,
    Protocol.check_eq_true_iff_accepted,
    Sampler.Checker.certificateCheck_eq_true_iff_accepted,
    DerivedPiDec.Checker.check_eq_true_iff_recomposition]
  constructor
  · rintro ⟨running, piCcs, sampler, piDec⟩
    exact {
      running := running
      piCcs := piCcs
      sampler := sampler
      tail := {
        sourceStructures := Canonical.Context.sourceStructures context
        piDecRecomposition := piDec
      }
    }
  · intro accepted
    exact ⟨accepted.running, accepted.piCcs, accepted.sampler,
      accepted.tail.piDecRecomposition⟩

/-- Concrete evaluator instance whose exactness is now discharged rather than
left as an abstract backend promise. -/
def evaluatorChecker
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context :
      Canonical.Context shape domain State publicRingColumns publicFits
        verifierRows) :
    FixedActive.Evaluator.Checker context.materialize where
  check := check context
  exact := check_eq_true_iff_accepted context

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Canonical.Checker
