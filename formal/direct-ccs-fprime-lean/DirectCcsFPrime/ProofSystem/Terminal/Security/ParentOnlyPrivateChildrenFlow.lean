import DirectCcsFPrime.Commitment.Parent.Security.ParentCEBHashBinding
import DirectCcsFPrime.ProofSystem.Terminal.Security.ParentOnlyAccumulatorStep

/-!
Parent-only private-child flow.

This module states the optimization boundary directly: post-DEC `CE(b)^14`
children are still private proof values, still checked by pointwise `Pi_DEC`,
and still wired into the next `Pi_CCS`; they are just not public child-hash
inputs. Public binding comes from the parent handle plus the ordinary prior
`F'` authority path.
-/

namespace DirectCcsFPrime

namespace ParentOnlyPrivateChildrenFlow

open DecDigitUniqueness

/--
One accepted parent-only step exposes the hidden child table flow:

* the prior parent handle authorizes private `Pi_DEC` children,
* the authorization is pointwise, not aggregate,
* the same private children feed the `Pi_CCS -> Pi_RLC` stage computation.

No public hash of the child table is an input to this theorem.
-/
theorem private_children_flow_of_parent_only_step
    {Digest PiCCSOut : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (parentHash : ParentCEBHashBinding.ParentCEBHash Digest)
    {ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment}
    {commitmentOfParent :
      ParentEncoding.SomeParentCEB →
        SuperNeo.ProofSystem.Commitment}
    {PiCCS :
      Nat →
        ReducedAccumulatorStep.AccumulatorHandle
          (DigestParentBinding.Source Digest)
          n →
        PiCCSOut →
          Prop}
    {PiRLC :
      Nat →
        PiCCSOut →
        DigestParentBinding.Source Digest →
          Prop}
    {i : Nat}
    {prior next :
      ParentOnlyAccumulatorStep.AccumulatorHandle
        (DigestParentBinding.Source Digest)}
    (hStep :
      ParentOnlyAccumulatorStep.Step
        (CanonicalPrivatePiDecVerifier.AuthorizedNextPiCCSInputs
          (n := n)
          (hashEncoded := parentHash.hashEncoded)
          (params := params)
          (ce := ce)
          (StatementEncodes :=
            ParentOpeningAuthorization.StatementEncodesByCommitment
              commitmentOfParent))
        (ParentOnlyAccumulatorStep.ParentSourceFromPiStages
          (n := n)
          PiCCS
          PiRLC)
        i
        prior
        next) :
    ∃ priorInputs,
      ParentOnlyAccumulatorStep.PointwisePrivateDecRequirements
        (n := n)
        (hashEncoded := parentHash.hashEncoded)
        (params := params)
        (ce := ce)
        (StatementEncodes :=
          ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent)
        prior.parentSource
        priorInputs ∧
      ParentOnlyAccumulatorStep.ParentSourceFromPiStages
        (n := n)
        PiCCS
        PiRLC
        i
        prior
        priorInputs
        next.parentSource :=
  ParentOnlyAccumulatorStep.pointwise_prior_dec_requirements_of_step hStep

/--
Two pointwise private DEC witnesses for the same public parent handle authorize
the same private child inputs. This is the local statement that aggregate
checks are insufficient: the proof uses binary length-14 digit tables,
coordinate-wise recomposition, CE witness-table identity, and next-wire
identity.

No public hash of `CE(b)^14` children is required.
-/
theorem same_private_child_inputs_without_public_child_hashes
    {Digest : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (parentHash : ParentCEBHashBinding.ParentCEBHash Digest)
    {ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment}
    {commitmentOfParent :
      ParentEncoding.SomeParentCEB →
        SuperNeo.ProofSystem.Commitment}
    {source : DigestParentBinding.Source Digest}
    {nextA nextB : ColumnDigits n}
    (hNoCollision :
      AjtaiResidueBinding.NoAjtaiBindingCollision params)
    (adapter :
      AjtaiResidueBinding.CEOpeningAdapter
        n
        params
        ce)
    (hA :
      ParentOnlyAccumulatorStep.PointwisePrivateDecRequirements
        (n := n)
        (hashEncoded := parentHash.hashEncoded)
        (params := params)
        (ce := ce)
        (StatementEncodes :=
          ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent)
        source
        nextA)
    (hB :
      ParentOnlyAccumulatorStep.PointwisePrivateDecRequirements
        (n := n)
        (hashEncoded := parentHash.hashEncoded)
        (params := params)
        (ce := ce)
        (StatementEncodes :=
          ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent)
        source
        nextB) :
    nextA = nextB :=
  ParentOnlyAccumulatorStep.pointwise_private_dec_requirements_functional_of_statementCommitment_and_ajtaiCEOpening
    (ParentCEBHashBinding.encodedParentCEBDigestBinding parentHash)
    hNoCollision
    adapter
    hA
    hB

/--
Two accepted parent-only latest steps from the same prior handle must compute
the same next parent source, even though the post-DEC children are private.

The witness extracted by the theorem is the precise runtime flow: one private
child table is pointwise DEC-authorized from the prior parent source and reused
by both `Pi_CCS -> Pi_RLC` computations.
-/
theorem same_next_parent_source_without_public_child_hashes
    {Digest PiCCSOut : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (parentHash : ParentCEBHashBinding.ParentCEBHash Digest)
    {ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment}
    {commitmentOfParent :
      ParentEncoding.SomeParentCEB →
        SuperNeo.ProofSystem.Commitment}
    {PiCCS :
      Nat →
        ReducedAccumulatorStep.AccumulatorHandle
          (DigestParentBinding.Source Digest)
          n →
        PiCCSOut →
          Prop}
    {PiRLC :
      Nat →
        PiCCSOut →
        DigestParentBinding.Source Digest →
          Prop}
    {i : Nat}
    {prior nextA nextB :
      ParentOnlyAccumulatorStep.AccumulatorHandle
        (DigestParentBinding.Source Digest)}
    (hPiCCS : ParentSourceStep.PiCCSFunctional PiCCS)
    (hPiRLC : ParentSourceStep.PiRLCFunctional PiRLC)
    (hNoCollision :
      AjtaiResidueBinding.NoAjtaiBindingCollision params)
    (adapter :
      AjtaiResidueBinding.CEOpeningAdapter
        n
        params
        ce)
    (hA :
      ParentOnlyAccumulatorStep.Step
        (CanonicalPrivatePiDecVerifier.AuthorizedNextPiCCSInputs
          (n := n)
          (hashEncoded := parentHash.hashEncoded)
          (params := params)
          (ce := ce)
          (StatementEncodes :=
            ParentOpeningAuthorization.StatementEncodesByCommitment
              commitmentOfParent))
        (ParentOnlyAccumulatorStep.ParentSourceFromPiStages
          (n := n)
          PiCCS
          PiRLC)
        i
        prior
        nextA)
    (hB :
      ParentOnlyAccumulatorStep.Step
        (CanonicalPrivatePiDecVerifier.AuthorizedNextPiCCSInputs
          (n := n)
          (hashEncoded := parentHash.hashEncoded)
          (params := params)
          (ce := ce)
          (StatementEncodes :=
            ParentOpeningAuthorization.StatementEncodesByCommitment
              commitmentOfParent))
        (ParentOnlyAccumulatorStep.ParentSourceFromPiStages
          (n := n)
          PiCCS
          PiRLC)
        i
        prior
        nextB) :
    (∃ priorInputs,
      ParentOnlyAccumulatorStep.PointwisePrivateDecRequirements
        (n := n)
        (hashEncoded := parentHash.hashEncoded)
        (params := params)
        (ce := ce)
        (StatementEncodes :=
          ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent)
        prior.parentSource
        priorInputs ∧
      ParentOnlyAccumulatorStep.ParentSourceFromPiStages
        (n := n)
        PiCCS
        PiRLC
        i
        prior
        priorInputs
        nextA.parentSource ∧
      ParentOnlyAccumulatorStep.ParentSourceFromPiStages
        (n := n)
        PiCCS
        PiRLC
        i
        prior
        priorInputs
        nextB.parentSource) ∧
      nextA.parentSource = nextB.parentSource := by
  constructor
  · exact
      ParentOnlyAccumulatorStep.pointwise_common_prior_dec_requirements_of_steps
        (ReducedAccumulatorStep.canonical_authorized_functional_of_statementCommitment_and_ajtaiCEOpening
          (ParentCEBHashBinding.encodedParentCEBDigestBinding parentHash)
          hNoCollision
          adapter)
        hA
        hB
  · exact
      ParentOnlyAccumulatorStep.step_parentSource_functional_of_statementCommitment_stages_and_ajtaiCEOpening
        hPiCCS
        hPiRLC
        (ParentCEBHashBinding.encodedParentCEBDigestBinding parentHash)
        hNoCollision
        adapter
        hA
        hB

end ParentOnlyPrivateChildrenFlow

end DirectCcsFPrime
