import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.ActiveSemantics
import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.Profile
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.RelationProfile
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive

/-!
Fixed-active NIFS profile bridge for the shape-indexed `Pi_CCS` output codec.

Assurance tier: model-level profile refinement.

Owns: construction of the canonical active Split-NC shape from the independent
thirteen-port selective relation profile and fixed NIFS arity; equality of its
batch-invariant Phi81 relation shape with the independent profile; derivation
of the exact 15-source count from an arbitrary fixed-active context; and
specialization of the active serializer length.

Does not own: proof that a production context/Rust relation instantiates this
canonical shape; output authority; SIS/Poseidon2; Rust/R1CS columns; costs;
necessity; or row removal.

Emits constraints: no.

Authority boundary: source and matrix counts are derived from two independent
semantic owners, never from a diagnostic artifact or profiler. The generic
context theorem keeps matrix count explicit; the canonical selective shape
needs no such premise. The legacy three-matrix projection is proved
incompatible with this shape. Production must still prove it selected the
active shape.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.output_digest.profile.shape` | combine fixed arity with the independent 13-port relation | computed | `selectiveShape` |
| `nifs.pi_ccs.output_digest.profile.relation` | forgetting batch counts recovers the exact independent relation shape | derived | `relationShape_eq` |
| `nifs.pi_ccs.output_digest.profile.sources` | fixed active arity implies exactly 15 sources | derived | `context_sourceCount_eq_15` |
| `nifs.pi_ccs.output_digest.profile.matrices` | canonical selective shape has exactly 13 matrices | computed | `selectiveShape_matrixCount_eq_13` |
| `nifs.pi_ccs.output_digest.profile.legacy_mismatch` | the active shape differs from the diagnostic three-matrix profile | derived/quarantine | `selectiveShape_not_diagnosticProfile` |
| `nifs.pi_ccs.output_digest.profile.fields` | complete canonical active pre-SIS message has 23,033 fields | derived representation length | `selective_serialize_length` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveProfile

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

namespace Selective

abbrev Profile :=
  Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.RelationProfile.Profile

abbrev matrixCount :=
  Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.RelationProfile.matrixCount

namespace Profile

abbrev shape {rows columns : Nat} (profile : Selective.Profile rows columns) :=
  Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.RelationProfile.Profile.shape
    profile

end Profile
end Selective

/-- Canonical active batch shape: relation dimensions come from the
independent selective profile, while source counts come from fixed NIFS arity. -/
def selectiveShape
    {rows columns : Nat}
    (profile : Selective.Profile rows columns) : SemanticShape where
  rowVariables := profile.rowVariables
  logicalWidth := columns
  freshCount := ConcretePhi81.FixedActive.arity.freshCount
  runningCount :=
    ConcretePhi81.FixedActive.arity.mode.count productionGlobalParams
  matrixCount := Selective.matrixCount

@[simp] theorem selectiveShape_sourceCount_eq_15
    {rows columns : Nat}
    (profile : Selective.Profile rows columns) :
    (selectiveShape profile).sourceCount = 15 := by
  rfl

@[simp] theorem selectiveShape_matrixCount_eq_13
    {rows columns : Nat}
    (profile : Selective.Profile rows columns) :
    (selectiveShape profile).matrixCount = 13 := by
  rfl

/-- Forgetting the fixed batch counts recovers exactly the independent
batch-invariant selective relation profile. -/
theorem relationShape_eq
    {rows columns : Nat}
    (profile : Selective.Profile rows columns) :
    Shape.ofSemantic (selectiveShape profile) publicRingColumns
        profile.publicFits =
      Selective.Profile.shape profile := by
  rfl

/-- The active serializer profile is exactly the independent thirteen-matrix
specialization. -/
theorem selectiveShape_profile_eq_steadyFixedPoint
    {rows columns : Nat}
    (profile : Selective.Profile rows columns) :
    PiCcsOutputDigest.Profile.ofSemanticShape (selectiveShape profile) =
      PiCcsOutputDigest.Profile.steadyFixedPointThirteenMatrix := by
  rfl

/-- The independently specified active relation cannot be mistaken for the
historical three-matrix diagnostic profile. No diagnostic projection module
is imported to establish this separation. -/
theorem selectiveShape_not_diagnosticProfile
    {rows columns : Nat}
    (profile : Selective.Profile rows columns) :
    PiCcsOutputDigest.Profile.ofSemanticShape (selectiveShape profile) ≠
      PiCcsOutputDigest.Profile.diagnosticThreeMatrix := by
  rw [selectiveShape_profile_eq_steadyFixedPoint]
  exact PiCcsOutputDigest.Profile.diagnosticThreeMatrix_ne_steadyFixedPointThirteenMatrix.symm

/-- The fixed active alignment derives one fresh plus fourteen running
sources; no independent source-count profile is accepted. -/
theorem context_sourceCount_eq_15
    {shape : SemanticShape}
    {State : Type}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context :
      ConcretePhi81.FixedActive.Context shape State publicRingColumns
        publicFits verifierRows) :
    shape.sourceCount = 15 :=
  context.alignment.total_eq_sourceCount.symm.trans
    ConcretePhi81.FixedActive.arity_total

/-- Exact fixed-active representation length. The matrix-count premise is
the still-open relation-shape refinement, not a trusted physical count. -/
theorem serialize_length
    {shape : SemanticShape}
    {State : Type}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context :
      ConcretePhi81.FixedActive.Context shape State publicRingColumns
        publicFits verifierRows)
    (matrixCount : shape.matrixCount = 13)
    (message : OutputMessage shape) :
    (ActiveSemantics.serialize message).length = 23033 :=
  ActiveSemantics.serialize_length_15_sources_13_matrices
    (context_sourceCount_eq_15 context) matrixCount message

/-- The independently constructed active selective shape fixes the exact
pre-SIS field length without a caller-supplied source or matrix count. -/
theorem selective_serialize_length
    {rows columns : Nat}
    (profile : Selective.Profile rows columns)
    (message : OutputMessage (selectiveShape profile)) :
    (ActiveSemantics.serialize message).length = 23033 :=
  ActiveSemantics.serialize_length_15_sources_13_matrices
    (selectiveShape_sourceCount_eq_15 profile)
    (selectiveShape_matrixCount_eq_13 profile) message

end Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveProfile
