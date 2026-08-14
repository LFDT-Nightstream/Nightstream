import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.CenteredSeptenaryFreshCcsAuthority

set_option autoImplicit false

namespace NightstreamTests.CenteredSeptenaryFreshCcsAuthority

open Nightstream.Implementation.R1CS.CenteredSeptenaryField
open Nightstream.Implementation.R1CS.CenteredSeptenaryFreshCcsAuthority
open Nightstream.Implementation.R1CS.CenteredSeptenaryLayout
open Nightstream.Implementation.R1CS.CenteredSeptenaryNormDischarged
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation

universe uCommitment

example {fieldCount : Nat}
    (layout : Layout fieldCount)
    {shape : Shape}
    (widthExact : layout.encodedColumnCount = shape.carrierWidth)
    {Commitment : Type uCommitment}
    (commit : Assignment shape → Commitment)
    (statement : CCSStatement shape Commitment)
    (fresh : statement.stage = .fresh)
    (assignment : Assignment shape)
    (holds : CCS.Holds (relationSemantics commit)
      Radix4Candidate.globalParams statement assignment) :
    ∀ field, FiniteAlphabetWord
      (finiteWordOfField
        (typedWordDigits layout widthExact assignment field)) := by
  exact radixFourCandidate_every_word_has_septenary_alphabet layout widthExact
    commit statement fresh assignment holds

end NightstreamTests.CenteredSeptenaryFreshCcsAuthority
