import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.WitnessProjection

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CheckedWitnessExtraction

open NightstreamFPrime.Spec
open StrongReduction ConcreteCarrier WitnessProjection UnifiedSources
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork (Result)

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

def SourceReturned (result : Option (SourceWitness shape carrier)) : Prop :=
  ∃ values, result = some values ∧
    SourceHolds extensionOps K.embed (openingMaps commit) params statement
      (reconstruct statement.publicInputs values)

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
