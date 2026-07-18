import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.DerivedPiRlc
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.OutputRefinement

/-!
Canonical opening authority for the concrete Π_RLC parent.

Assurance tier: model-level.

Owns: the theorem that source-authorized, paper-valid Π_CCS outputs with a
verifier-derived valid challenge batch open the one computed Π_RLC parent at
the canonical combined assignment.

Does not own: Π_DEC extraction, Ajtai binding security, transcript
probability, Poseidon2 refinement, Rust, R1CS, costs, or row removal.

Emits constraints: no.

Authority boundary: this file proves an opening for the parent already
computed by `derive`; it never accepts a prover-supplied parent. CE authority
uses only the source-derived `yRing` equality. Packed `yZcol` and its combined
source binding belongs to the Split-NC terminal proof; it is not a paper CE
field and therefore is not carried into the Π_RLC parent.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_rlc.authority.parent.inputs` | every materialized Π_CCS output is a genuine fresh CE opening | derived | `canonicalParent_holds_of_yRingBound` |
| `nifs.pi_rlc.authority.parent.challenge` | every Π_RLC scalar belongs to the production strong set | derived from sampler replay | `canonicalParent_holds_of_yRingBound` |
| `nifs.pi_rlc.authority.parent.opening` | the computed parent opens at the canonical challenge-folded assignment | derived | `canonicalParent_holds_of_yRingBound` |
| `nifs.pi_rlc.authority.parent.y_zcol` | packed `yZcol` closes Split-NC only and is absent from paper CE/Π_RLC | eliminated by type/dataflow | `canonicalParent_holds_of_outputBound` |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.PiRlcParentOpening

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uState

variable
  {shape : SemanticShape}
  {State : Type uState}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}
  {arity : BatchArity productionGlobalParams}

/-- Row-evaluation authority required by CE materialization. Delayed NC and
packed-sidecar data are intentionally absent. -/
def YRingBound
    (context :
      Context shape State publicRingColumns publicFits verifierRows
        arity)
    (data : Data shape)
    (certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput) : Prop :=
  certificate.piCcs.output.yRing =
    Polynomial.Fe.sourceYRingAt data
      (derive context certificate).piCcs.fePoint.row

/-- The unique derived Π_RLC parent is a genuine CE opening at the exact
challenge-folded source assignment. -/
theorem canonicalParent_holds_of_yRingBound
    {context :
      Context shape State publicRingColumns publicFits verifierRows
        arity}
    {data : Data shape}
    {certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput}
    (paper : Semantics.Paper.Holds data)
    (inputBound :
      InputAuthority.BoundToSources publicRingColumns publicFits
        (commit context.key) data context.alignment context.input)
    (yRingBound : YRingBound context data certificate)
    (sampler : Sampler.CertificateAccepted context certificate)
    (structures : DerivedPiRlc.SourceStructuresBound context) :
    CE.Holds (semantics context.key) productionGlobalParams
      (derive context certificate).piRlcOutput
      (PiRLC.combinedWitness (rlcAlgebra context.key)
        certificate.piRlcChallenges
        (InputAuthority.productAssignments data context.alignment)) := by
  let execution := derive context certificate
  let assignments := InputAuthority.productAssignments data context.alignment
  have outputsHold :
      ProductHolds publicRingColumns publicFits (commit context.key)
        execution.piCcsOutputs assignments := by
    simpa [execution, derive] using
      (Protocol.OutputRefinement.materializedOutputsHold_of_yRingEq
        publicRingColumns publicFits (commit context.key) data
        context.alignment context.input execution.piCcs.fePoint.row
        certificate.piCcs.output production_norm_stages.1 paper inputBound
        yRingBound)
  have parentHolds :=
    PiRLC.combinedOutput_holds (semantics context.key) productionGlobalParams
      (rlcAlgebra context.key) arity context.system
      execution.piCcs.fePoint.row execution.piCcsOutputs
      certificate.piRlcChallenges assignments
      (fun _ => rfl) structures (fun _ => rfl)
      (Sampler.certificateAccepted_challengesValid sampler)
      outputsHold
      (Phi81Relation.evaluationPointValid_holds context.system
        execution.piCcs.fePoint.row)
  simpa [execution, derive, assignments] using parentHolds

/-- The complete canonical Split-NC output boundary contains exactly the
`yRing` authority needed for the paper CE parent. Its packed `yZcol` branch
remains consumed by Split-NC soundness and is not copied into the parent. -/
theorem canonicalParent_holds_of_outputBound
    {context :
      Context shape State publicRingColumns publicFits verifierRows
        arity}
    {data : Data shape}
    {certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput}
    (paper : Semantics.Paper.Holds data)
    (inputBound :
      InputAuthority.BoundToSources publicRingColumns publicFits
        (commit context.key) data context.alignment context.input)
    (outputBound :
      Protocol.BlockLane.OutputBound context.covers data
        (derive context certificate).piCcs certificate.piCcs.output)
    (sampler : Sampler.CertificateAccepted context certificate)
    (structures : DerivedPiRlc.SourceStructuresBound context) :
    CE.Holds (semantics context.key) productionGlobalParams
      (derive context certificate).piRlcOutput
      (PiRLC.combinedWitness (rlcAlgebra context.key)
        certificate.piRlcChallenges
        (InputAuthority.productAssignments data context.alignment)) := by
  exact canonicalParent_holds_of_yRingBound paper inputBound outputBound.1
    sampler structures

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.PiRlcParentOpening
