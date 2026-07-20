import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPiCcs

/-!
Production NIFS refinement with packed-`yZcol` deliberately delayed.

Assurance tier: model-level, pending concrete decoder refinement.

Owns: composition of the raw-source `Pi_CCS` prefix with the unchanged
running-parent, sampler, `Pi_RLC`, and `Pi_DEC` checks; a positive refinement
record containing paper truth and exact `yRing` authority; and promotion to
the existing independent semantic fold once the adjacent-step packed
projection theorem supplies the missing old output equation.

Does not own: derivation of source data/input binding from final assignment
columns, child-opening extraction, the adjacent or terminal delayed check,
Rust/R1CS rows, primitive security, costs, or row removal.

Emits constraints: none.

Authority boundary: `DelayedRefinement` is intentionally insufficient to
construct `CertificateRefinement`. Only `toCertificateRefinement` accepts the
separately derived `PackedYZcolBoundAtBlock`; no generic `OutputBound`,
`SourceProjectionMatches`, or child sidecar premise is admitted.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.production.running` | validate the exact incoming parent against all running children | checked | `Accepted.running` |
| `nifs.production.pi_ccs` | FE plus raw-source ordinary/combined NC acceptance | checked | `Accepted.piCcs` |
| `nifs.production.y_ring` | retain the exact source-derived `yRing` equation | derived | `DelayedRefinement.yRing` |
| `nifs.production.tail` | replay sampler and outgoing recomposition checks | checked | `Accepted.sampler`, `Accepted.tail` |
| `nifs.production.y_zcol.delay` | require the independently derived adjacent packed equation before semantic promotion | delayed/derived | `toCertificateRefinement` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionNifs

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane

universe uState

variable
  {shape : SemanticShape}
  {State : Type uState}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Complete claims-level production acceptance. This matches the executable
protocol boundary: public Π_CCS messages, running-parent checks, sampler
replay, and Π_RLC/Π_DEC tail checks. No private `Sources.Data` value appears
in this predicate. -/
structure MessageAccepted
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context) : Prop where
  running : RunningAuthority.Accepted context
  piCcs : CombinedNc.ProductionPiCcs.MessageAccepted context certificate
  sampler : Sampler.CertificateAccepted context certificate
  tail : TailAccepted context certificate

/-- Complete production physical acceptance except for concrete decoding of
the private `Sources.Data` value. The existing tail checks remain unchanged;
only `Pi_CCS` uses the raw-assignment NC terminal. -/
structure Accepted
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context) : Prop where
  running : RunningAuthority.Accepted context
  piCcs : CombinedNc.ProductionPiCcs.Accepted context data certificate
  sampler : Sampler.CertificateAccepted context certificate
  tail : TailAccepted context certificate

/-- Claims-level NIFS acceptance refines to the post-extraction raw-source
predicate or the exact current Π_CCS packed-output opening failure. -/
theorem messageAccepted_implies_accepted_or_outputBindingFailure
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context)
    (accepted : MessageAccepted context certificate) :
    Accepted context data certificate ∨
      CombinedNc.ProductionPiCcs.OutputBindingFailure
        context data certificate := by
  rcases
      CombinedNc.ProductionPiCcs.messageAccepted_implies_accepted_or_outputBindingFailure
        context data certificate accepted.piCcs with raw | failure
  · exact Or.inl {
      running := accepted.running
      piCcs := raw
      sampler := accepted.sampler
      tail := accepted.tail
    }
  · exact Or.inr failure

/-- Terminal- or successor-derived packed authority closes the only
claims-to-raw extraction branch.  All other NIFS checks are copied from the
same public-message acceptance object. -/
theorem accepted_of_messageAccepted_and_packed
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context)
    (accepted : MessageAccepted context certificate)
    (packed : Terminal.PackedYZcolBoundAtBlock context.covers data
      (CombinedNc.ProductionPiCcs.ncPoint context certificate).block
      certificate.piCcs.output) :
    Accepted context data certificate where
  running := accepted.running
  piCcs := CombinedNc.ProductionPiCcs.accepted_of_messageAccepted_and_packed
    context data certificate accepted.piCcs packed
  sampler := accepted.sampler
  tail := accepted.tail

/-- Certificate-indexed evidence before the one-fold-delayed packed output is
closed. This record contains no asserted `yZcol` equality. -/
structure DelayedRefinement
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context) : Prop where
  paper : Semantics.Paper.Holds data
  input : SemanticInput context data
  running : RunningAuthority.Accepted context
  yRing : CombinedNc.ProductionPiCcs.YRingBound context data certificate
  sampler : Sampler.CertificateAccepted context certificate
  children : ChildOpenings context data certificate

namespace DelayedRefinement

/-- Once the successor or terminal raw check derives the packed equation, the
ordinary two-component output binding follows exactly. -/
def outputBound
    {context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows}
    {data : Data shape}
    {certificate : FixedActive.Certificate context}
    (refinement : DelayedRefinement context data certificate)
    (packed : Terminal.PackedYZcolBoundAtBlock context.covers data
      (derive context certificate).piCcs.ncPoint.block
      certificate.piCcs.output) :
    OutputBound context data certificate :=
  ⟨refinement.yRing, packed⟩

/-- Promotion uses the existing independent refinement owner after filling
the packed half of `OutputBound`; no semantic relation is restated here. -/
def toCertificateRefinement
    {context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows}
    {data : Data shape}
    {certificate : FixedActive.Certificate context}
    (refinement : DelayedRefinement context data certificate)
    (packed : Terminal.PackedYZcolBoundAtBlock context.covers data
      (derive context certificate).piCcs.ncPoint.block
      certificate.piCcs.output) :
    CertificateRefinement context data certificate where
  paper := refinement.paper
  input := refinement.input
  running := refinement.running
  output := refinement.outputBound packed
  sampler := refinement.sampler
  children := refinement.children

/-- The delayed production execution refines the certificate-independent
semantic fold as soon as its old packed output is closed. -/
theorem toSemanticFold
    {context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows}
    {data : Data shape}
    {certificate : FixedActive.Certificate context}
    (refinement : DelayedRefinement context data certificate)
    (packed : Terminal.PackedYZcolBoundAtBlock context.covers data
      (derive context certificate).piCcs.ncPoint.block
      certificate.piCcs.output) :
    SemanticFold.Holds context data
      (derive context certificate).piRlcOutput
      (outputChildren context certificate) :=
  (refinement.toCertificateRefinement packed).toSemanticFold

end DelayedRefinement

/-- Full physical production acceptance yields the positive delayed
refinement, a specifically named `yRing` failure, or a raw `Pi_CCS` algebraic
event. There is no generic output-unbound branch. -/
theorem accepted_implies_delayedRefinement_or_yRingUnbound_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context)
    (input : SemanticInput context data)
    (children : ChildOpenings context data certificate)
    (accepted : Accepted context data certificate) :
    DelayedRefinement context data certificate ∨
      CombinedNc.ProductionPiCcs.YRingUnbound context data certificate ∨
      CombinedNc.ProductionPiCcs.BadEvent context data certificate := by
  rcases
      CombinedNc.ProductionPiCcs.accepted_implies_paper_and_yRingBound_or_yRingUnbound_or_badEvent
        noZeroDivisors context data certificate input.publicInput
        accepted.piCcs with
    paper | yRingUnbound | bad
  · exact Or.inl {
      paper := paper.1
      input := input
      running := accepted.running
      yRing := paper.2
      sampler := accepted.sampler
      children := children
    }
  · exact Or.inr (Or.inl yRingUnbound)
  · exact Or.inr (Or.inr bad)

end Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionNifs
