import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.WitnessProjection

/-!
Owns the source-return event: a returned source witness satisfies the source
relation after its fresh prefixes are restored from the verifier statement
through Phi81Relation's actual map. `HyperNovaFirstFailure.MarkedSourceFailure`
reads this event.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CheckedWitnessExtraction

open NightstreamFPrime.Spec
open StrongReduction ConcreteCarrier WitnessProjection UnifiedSources
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork (Result)

/-- These are the existing production opening maps at the selected Phi81 prefix. -/
def openingMaps {Commitment : Type*} {carrier : Phi81Relation.Shape}
    (commit : Phi81Relation.Assignment carrier → Commitment) :
    OpeningMaps Commitment (Phi81Relation.PublicInput carrier) carrier.carrierWidth where
  commit := commit
  projectPublicInput := Phi81Relation.projectPublicInput

variable {Commitment : Type*} {shape : Shape} {carrier : Phi81Relation.Shape}
  {blockCount : Nat} (commit : Phi81Relation.Assignment carrier → Commitment)
  (params : GlobalParams)
  (statement : Statement K Commitment (Phi81Relation.PublicInput carrier)
    shape carrier.carrierWidth blockCount baseOps)

/-- A valid returned source witness is interpreted through the existing source
relation and the verifier-owned public prefix. -/
def SourceReturned (result : Option (SourceWitness shape carrier)) : Prop :=
  ∃ values, result = some values ∧
    SourceHolds extensionOps K.embed (openingMaps commit) params statement
      (reconstruct statement.publicInputs values)

/-- The returned tails and full running values satisfy the existing source
CCS/CE products. The selected fresh profile has b=2. -/
theorem sourceReturned_iff_memberships (freshBound : params.b = 2)
    (result : Option (SourceWitness shape carrier)) :
    SourceReturned commit params statement result ↔
      ∃ values, result = some values ∧
        (∀ fresh, CCS.Holds (paperRelationSemantics baseOps extensionOps K.embed (openingMaps commit)) params
          (SourceMembership.freshInstance statement fresh)
          ((reconstruct statement.publicInputs values).assignments (freshSourceIndex fresh))) ∧
        (∀ running, CE.Holds (paperRelationSemantics baseOps extensionOps K.embed (openingMaps commit)) params
          (SourceMembership.runningInstance statement running)
          ((reconstruct statement.publicInputs values).assignments (runningSourceIndex running))) := by
  unfold SourceReturned
  simp only [SourceMembership.sourceHolds_iff_memberships extensionOps extensionLaws K.embed
    (openingMaps commit) params freshBound]

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CheckedWitnessExtraction
