import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcQuotientHandoff

/-!
Focused mutation regression for the public `Pi_RLC` challenge carrier.

The negative control replaces every challenge coefficient with the constant
wire.  The selected quotient source must instead use the physical centered
sampler outputs, and binding must erase the caller mutation completely.
-/

set_option autoImplicit false

namespace NightstreamTests.PaperNifsPiRlcQuotientChallengeBinding

open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open PaperNifsPiRlcQuotientHandoff

def firstSource : Fin Arity.total :=
  ⟨0, by decide⟩

def firstPosition : Fin PiRlcCanonicalSelector.outputCount :=
  ⟨0, by decide⟩

def zeroChallengeMutation
    (source : KPiRlcSemanticBinding.SourceColumns Params Arity 13) :
    KPiRlcSemanticBinding.SourceColumns Params Arity 13 :=
  { source with
    challenges := fun _ => List.replicate
      Nightstream.SuperNeo.Concrete.ringDegree 0 }

/-- The zero-column mutation is observably different from the physical
sampler carrier at its first coefficient. -/
theorem zeroChallengeMutation_differs_from_sampler
    (source : KPiRlcSemanticBinding.SourceColumns Params Arity 13) :
    (zeroChallengeMutation source).challenges firstSource ≠
      (samplerBoundSource 1 source).challenges firstSource := by
  intro equal
  have physicalMember :
      PiRlcCanonicalSelector.outputColumn
          (PiRlcCanonicalSamplerProgram.selectorBase 1)
          (samplerCoordinate firstSource) firstPosition ∈
        challengeColumns 1 firstSource := by
    unfold challengeColumns
    exact List.mem_ofFn.mpr ⟨firstPosition, rfl⟩
  have boundMember :
      PiRlcCanonicalSelector.outputColumn
          (PiRlcCanonicalSamplerProgram.selectorBase 1)
          (samplerCoordinate firstSource) firstPosition ∈
        (samplerBoundSource 1 source).challenges firstSource := by
    simpa using physicalMember
  have mutatedMember :
      PiRlcCanonicalSelector.outputColumn
          (PiRlcCanonicalSamplerProgram.selectorBase 1)
          (samplerCoordinate firstSource) firstPosition ∈
        (zeroChallengeMutation source).challenges firstSource := by
    rw [equal]
    exact boundMember
  have zeroMember :
      PiRlcCanonicalSelector.outputColumn
          (PiRlcCanonicalSamplerProgram.selectorBase 1)
          (samplerCoordinate firstSource) firstPosition ∈
        List.replicate Nightstream.SuperNeo.Concrete.ringDegree 0 := by
    simpa [zeroChallengeMutation] using mutatedMember
  have equalsZero : PiRlcCanonicalSelector.outputColumn
      (PiRlcCanonicalSamplerProgram.selectorBase 1)
      (samplerCoordinate firstSource) firstPosition = 0 :=
    List.eq_of_mem_replicate zeroMember
  unfold PiRlcCanonicalSelector.outputColumn
    PiRlcCanonicalSelector.positionBase at equalsZero
  omega

/-- Binding is authoritative: changing only caller challenge lists cannot
change the selected quotient source. -/
theorem samplerBinding_erases_zeroChallengeMutation
    (source : KPiRlcSemanticBinding.SourceColumns Params Arity 13) :
    samplerBoundSource 1 (zeroChallengeMutation source) =
      samplerBoundSource 1 source := by
  cases source
  rfl

end NightstreamTests.PaperNifsPiRlcQuotientChallengeBinding
