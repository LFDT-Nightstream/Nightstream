import Nightstream.Implementation.NebulaV2.ProductionFreshClaimProducerFor

/-!
Regression surface for the exact source-R1CS to SuperNeo-CCS compiler.

The hostile interface below records the defect that this compiler removes:
putting the desired conclusion in a witness is logically equivalent to
assuming that conclusion. It is not a soundness reduction.
-/

set_option autoImplicit false

namespace tests.NebulaV2ProductionFreshRelationCompilerFor

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.ProductionFreshRelationCompilerFor
open Nightstream.Implementation.R1CS.CenteredTernaryField

#check SourceProgram.system_degree_exact
#check SourceProgram.ccsSatisfied_iff_application
#check SourceProgram.ccsSatisfied_iff_loweredRows
#check SourceProgram.decodedSourceAssignment_canonical
#check SourceProgram.decodedSourceAssignment_direct
#check SourceProgram.ccsSatisfied_iff_decodedSourceRows
#check SourceProgram.encoded_loweredRows_iff_sourceRows
#check SourceProgram.projectPublicInput_encodeCarrier_for
#check SourceProgram.encoded_ccsSatisfied_iff_sourceRows
#check ProductionFreshClaimProducerFor.RelationAuthority
#check ProductionFreshClaimProducerFor.FreshRelationWitness.publicOutput
#check ProductionFreshClaimProducerFor.FreshRelationWitness.norm
#check ProductionFreshClaimProducerFor.FreshRelationWitness.relation

/-- Countermodel for the removed direct-assertion interface. Such a witness
contains no evidence beyond the result that the caller wants to obtain. -/
structure CircularCertificate (conclusion : Prop) : Prop where
  assertedConclusion : conclusion

theorem circular_certificate_equivalent_to_conclusion
    (conclusion : Prop) :
    Nonempty (CircularCertificate conclusion) ↔ conclusion := by
  constructor
  · rintro ⟨certificate⟩
    exact certificate.assertedConclusion
  · intro asserted
    exact ⟨⟨asserted⟩⟩

/-! A relation artifact need not determine a unique source program. If each
claim selects its own preimage of the artifact, two claims can use different
source programs while the verifier sees the same artifact. -/

def collapsedArtifact (_program : Bool) : Unit := ()

structure LoosePerClaimAuthority where
  program : Bool
  artifactExact : collapsedArtifact program = ()

def looseFalseProgram : LoosePerClaimAuthority where
  program := false
  artifactExact := rfl

def looseTrueProgram : LoosePerClaimAuthority where
  program := true
  artifactExact := rfl

theorem per_claim_preimages_can_disagree :
    looseFalseProgram.program ≠ looseTrueProgram.program := by
  decide

/-- An accepted 41-trit carrier is not necessarily the deterministic word
chosen by the honest encoder. Reverse soundness must decode the exact word
that the commitment opens. -/
theorem accepted_words_can_be_distinct_with_equal_decode :
    FiniteAlphabetWord (rawTargetWord 0) /\
      FiniteAlphabetWord
        (rawTargetWord Nightstream.Implementation.R1CS.goldilocksP) /\
      rawTargetWord 0 ≠
        rawTargetWord Nightstream.Implementation.R1CS.goldilocksP /\
      decodeFiniteWord (rawTargetWord 0) =
        decodeFiniteWord
          (rawTargetWord Nightstream.Implementation.R1CS.goldilocksP) := by
  exact ⟨duplicate_words_accepted.1, duplicate_words_accepted.2,
    duplicate_words_differ, duplicate_words_decode_same⟩

end tests.NebulaV2ProductionFreshRelationCompilerFor
