import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionNifs
import Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.DelayedPending
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.DerivedPiDec.Checker
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Canonical.RunningAuthority
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Checker

/-!
Executable production checker for the delayed packed-`yZcol` path.

Assurance tier: model-level executable semantics; concrete assignment decoding
is supplied by the fixed-profile correspondence leaf.

Owns: the claims-only Boolean `Pi_CCS` check whose delayed NC terminal is
computed from the public output message; the post-extraction raw-source
checker; Boolean composition with the canonical incoming-parent, sampler, and
outgoing `Pi_DEC` checks; and exact equivalence with their respective logical
predicates.

Does not own: physical assignment decoding, R1CS rows, transcript-machine
agreement with Rust, one-fold state continuity, terminal openings, commitment
binding, costs, or row-removal permission.

Emits constraints: none; canonical executable verifier semantics only.

| Stable stage path | Mathematical obligation | Authority class |
|---|---|---|
| `nifs.production.pi_ccs.check.fe` | execute the existing statement-derived FE checker | checked |
| `nifs.production.pi_ccs.check.nc.message` | execute the combined NC chain with the public verifier terminal | checked/computed |
| `nifs.production.pi_ccs.check.nc.raw` | validate the same chain against extracted raw sources | checked/post-extraction |
| `nifs.production.check.tail` | execute incoming parent, sampler, and outgoing recomposition checks | checked |
| `nifs.production.check.exact` | each Boolean checker is exact to its claims or raw predicate | derived |

`messageCheck` is the executable verifier surface and contains no private
source table. `check` is deliberately post-extraction: its terminal is
`ProductionPiCcs.rawPolynomial context data`. Neither checker promotes a
digest or caller-provided scalar to authority.
-/

namespace Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionChecker

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.SumCheck.Finite
open Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.DelayedPending

private abbrev ops := ConcreteCarrier.extensionOps

universe uState uEncoding uDigest

variable
  {shape : SemanticShape}
  {State : Type uState}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Fail-closed equality check between the public accumulator coordinate and
the digest recomputed from the complete parent, ordered children, and optional
pending projection. The digest remains compression, not authority; later
soundness keeps `BindingFailure` explicit. -/
def stateBindingCheck
    {Encoding : Type uEncoding}
    {Digest : Type uDigest}
    [DecidableEq Digest]
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (stateDigest : Digest)
    (parent : CEStatement
      (RelationShape shape publicRingColumns publicFits)
      (CommitmentValue verifierRows))
    (children : Fin productionGlobalParams.k -> CEStatement
      (RelationShape shape publicRingColumns publicFits)
      (CommitmentValue verifierRows))
    (pending : Option ProductionDelayedBlockLane) : Bool :=
  decide (stateDigest = pendingFamilyDigest scheme parent children pending)

/-- Executing the accumulator-coordinate equality derives `StateBinds`; it
is not admitted as a semantic premise. -/
theorem stateBindingCheck_eq_true_iff
    {Encoding : Type uEncoding}
    {Digest : Type uDigest}
    [DecidableEq Digest]
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (stateDigest : Digest)
    (parent : CEStatement
      (RelationShape shape publicRingColumns publicFits)
      (CommitmentValue verifierRows))
    (children : Fin productionGlobalParams.k -> CEStatement
      (RelationShape shape publicRingColumns publicFits)
      (CommitmentValue verifierRows))
    (pending : Option ProductionDelayedBlockLane) :
    stateBindingCheck scheme stateDigest parent children pending = true <->
      StateBinds scheme stateDigest parent children pending := by
  simp [stateBindingCheck, StateBinds]

/-- Execute the production FE phase and the claims-only combined NC phase.
The NC point and incoming state are transcript-derived; the terminal is
computed from the public output message. -/
def piCcsMessageCheck
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context) : Bool :=
  Fe.check context.feMachine context.initialState context.profile
      context.piCcsInput context.feCoins certificate.piCcs.output
      certificate.piCcs.fe &&
    Transcript.Nc.BlockLane.check context.ncMachine
      (ProductionPiCcs.ncTranscriptState context certificate)
      (ProductionPiCcs.rawInitial context)
      (ProductionPiCcs.messageTerminal context certificate)
      certificate.piCcs.nc

/-- The claims-only Π_CCS Boolean checker is exact to the public protocol
predicate. -/
theorem piCcsMessageCheck_eq_true_iff_accepted
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context) :
    piCcsMessageCheck context certificate = true <->
      ProductionPiCcs.MessageAccepted context certificate := by
  simp only [piCcsMessageCheck, Bool.and_eq_true,
    Fe.check_eq_true_iff_accepted,
    Transcript.Nc.BlockLane.check_eq_true_iff_accepted]
  constructor
  · rintro ⟨fe, nc⟩
    exact { fe := fe, nc := nc }
  · intro accepted
    exact ⟨accepted.fe, accepted.nc⟩

/-- Execute the production FE phase and the raw-source combined NC phase.
The final NC scalar is recomputed from `data`; the output message supplies no
NC terminal value. -/
def piCcsCheck
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context) : Bool :=
  Fe.check context.feMachine context.initialState context.profile
      context.piCcsInput context.feCoins certificate.piCcs.output
      certificate.piCcs.fe &&
    FixedPhase.check ops.toOps
      (ProductionPiCcs.rawPolynomial context data)
      (ProductionPiCcs.rawInitial context)
      (ProductionPiCcs.ncPoint context certificate).coordinates
      certificate.piCcs.nc.toSumCheck

/-- The production `Pi_CCS` Boolean checker is exact to the logical prefix
accepted by delayed NIFS soundness. -/
theorem piCcsCheck_eq_true_iff_accepted
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context) :
    piCcsCheck context data certificate = true <->
      ProductionPiCcs.Accepted context data certificate := by
  simp only [piCcsCheck, Bool.and_eq_true,
    Fe.check_eq_true_iff_accepted,
    FixedPhase.check_eq_true_iff_accepted]
  constructor
  · rintro ⟨fe, nc⟩
    exact { fe := fe, nc := nc }
  · intro accepted
    exact ⟨accepted.fe, accepted.nc⟩

/-- Execute every retained production NIFS family without reading a private
assignment. This is the canonical claims-level checker intended for Rust/R1CS
refinement. -/
def messageCheck
    (context : FixedActive.Canonical.Context shape State publicRingColumns
      publicFits verifierRows)
    (certificate : FixedActive.Certificate context.materialize) : Bool :=
  FixedActive.Canonical.RunningAuthority.check context &&
    (piCcsMessageCheck context.materialize certificate &&
      (Sampler.Checker.certificateCheck context.materialize certificate &&
        DerivedPiDec.Checker.check context.materialize certificate))

/-- Claims-level Boolean production acceptance contains exactly the public
delayed NIFS predicate. -/
theorem messageCheck_eq_true_iff_accepted
    (context : FixedActive.Canonical.Context shape State publicRingColumns
      publicFits verifierRows)
    (certificate : FixedActive.Certificate context.materialize) :
    messageCheck context certificate = true <->
      ProductionNifs.MessageAccepted context.materialize certificate := by
  simp only [messageCheck, Bool.and_eq_true,
    FixedActive.Canonical.RunningAuthority.check_eq_true_iff_accepted,
    piCcsMessageCheck_eq_true_iff_accepted,
    Sampler.Checker.certificateCheck_eq_true_iff_accepted,
    DerivedPiDec.Checker.check_eq_true_iff_recomposition]
  constructor
  · rintro ⟨running, piCcs, sampler, piDec⟩
    exact {
      running := running
      piCcs := piCcs
      sampler := sampler
      tail := {
        sourceStructures := FixedActive.Canonical.Context.sourceStructures
          context
        piDecRecomposition := piDec
      }
    }
  · intro accepted
    exact ⟨accepted.running, accepted.piCcs, accepted.sampler,
      accepted.tail.piDecRecomposition⟩

/-- Execute all retained production NIFS families on one canonical context.
Unlike the ordinary canonical checker, the `Pi_CCS` child is the raw-source
combined checker above. -/
def check
    (context : FixedActive.Canonical.Context shape State publicRingColumns
      publicFits verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context.materialize) : Bool :=
  FixedActive.Canonical.RunningAuthority.check context &&
    (piCcsCheck context.materialize data certificate &&
      (Sampler.Checker.certificateCheck context.materialize certificate &&
        DerivedPiDec.Checker.check context.materialize certificate))

/-- Boolean production acceptance contains exactly the independently stated
delayed NIFS predicate. In particular, `ProductionNifs.Accepted` is derived
from execution rather than accepted as a theorem premise. -/
theorem check_eq_true_iff_accepted
    (context : FixedActive.Canonical.Context shape State publicRingColumns
      publicFits verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context.materialize) :
    check context data certificate = true <->
      ProductionNifs.Accepted context.materialize data certificate := by
  simp only [check, Bool.and_eq_true,
    FixedActive.Canonical.RunningAuthority.check_eq_true_iff_accepted,
    piCcsCheck_eq_true_iff_accepted,
    Sampler.Checker.certificateCheck_eq_true_iff_accepted,
    DerivedPiDec.Checker.check_eq_true_iff_recomposition]
  constructor
  · rintro ⟨running, piCcs, sampler, piDec⟩
    exact {
      running := running
      piCcs := piCcs
      sampler := sampler
      tail := {
        sourceStructures := FixedActive.Canonical.Context.sourceStructures
          context
        piDecRecomposition := piDec
      }
    }
  · intro accepted
    exact ⟨accepted.running, accepted.piCcs, accepted.sampler,
      accepted.tail.piDecRecomposition⟩

end Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionChecker
