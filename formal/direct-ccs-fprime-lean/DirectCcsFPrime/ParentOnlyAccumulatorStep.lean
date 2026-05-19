import DirectCcsFPrime.ParentSourceStep
import DirectCcsFPrime.ReducedAccumulatorStep

/-!
Parent-only accumulator step for the reduced `CE(B)` handle.

This module states the optimized latest-step shape where the public accumulator
handle carries only the parent `CE(B)` source. The child `CE(b)^k` table is
private advice, but it must be authorized by the canonical private `Pi_DEC`
relation before it may feed the next `Pi_CCS`/`Pi_RLC` parent-source
computation.
-/

namespace DirectCcsFPrime

namespace ParentOnlyAccumulatorStep

open DecDigitUniqueness

/--
Parent-only accumulator handle exposed by the optimized direct `F'` public
image.

The post-DEC children are deliberately absent from this handle.
-/
structure AccumulatorHandle (Source : Type) where
  parentSource : Source

/--
The parent-source computation used after privately decoding the prior parent.

`priorInputs` is the hidden `CE(b)^k` child table authorized from
`prior.parentSource`; the output source is the new parent `CE(B)` produced by
the latest `Pi_CCS -> Pi_RLC` stages.
-/
def ParentSourceFunctional
    {Source : Type}
    {n : Nat}
    (ParentSourceStep :
      Nat →
        AccumulatorHandle Source →
        ColumnDigits n →
        Source →
          Prop) : Prop :=
  ∀ i prior priorInputs sourceA sourceB,
    ParentSourceStep i prior priorInputs sourceA →
    ParentSourceStep i prior priorInputs sourceB →
      sourceA = sourceB

/--
Optimized parent-only accumulator update.

The verifier may keep the post-DEC child table private, but every accepted step
must provide a child table authorized by the prior parent source and must use
that exact table in the parent-source computation.
-/
def Step
    {Source : Type}
    {n : Nat}
    (AuthorizedPrior : Source → ColumnDigits n → Prop)
    (ParentSourceStep :
      Nat →
        AccumulatorHandle Source →
        ColumnDigits n →
        Source →
          Prop)
    (i : Nat)
    (prior next : AccumulatorHandle Source) : Prop :=
  ∃ priorInputs,
    AuthorizedPrior prior.parentSource priorInputs ∧
      ParentSourceStep i prior priorInputs next.parentSource

/--
If private DEC authorization is functional and the parent-source computation is
functional for a fixed authorized child table, then a parent-only step cannot
produce two different next parent sources from the same prior handle.
-/
theorem step_parentSource_functional
    {Source : Type}
    {n : Nat}
    {AuthorizedPrior : Source → ColumnDigits n → Prop}
    {ParentSourceStep :
      Nat →
        AccumulatorHandle Source →
        ColumnDigits n →
        Source →
          Prop}
    {i : Nat}
    {prior nextA nextB : AccumulatorHandle Source}
    (hAuthorized :
      ReducedAccumulatorStep.AuthorizedFunctional AuthorizedPrior)
    (hParent : ParentSourceFunctional ParentSourceStep)
    (hA : Step AuthorizedPrior ParentSourceStep i prior nextA)
    (hB : Step AuthorizedPrior ParentSourceStep i prior nextB) :
    nextA.parentSource = nextB.parentSource := by
  rcases hA with ⟨inputsA, hAuthA, hParentA⟩
  rcases hB with ⟨inputsB, hAuthB, hParentB⟩
  have hInputs : inputsA = inputsB :=
    hAuthorized prior.parentSource inputsA inputsB hAuthA hAuthB
  subst inputsB
  exact hParent i prior inputsA nextA.parentSource nextB.parentSource hParentA hParentB

/--
Two accepted parent-only steps from the same prior handle must use one common
authorized hidden child table.

This is the anti-substitution statement behind the parent-only public handle:
the children are private, but they are not free advice. Any alternate accepted
latest step must reuse the same children authorized by the prior parent source.
-/
theorem step_common_authorized_inputs
    {Source : Type}
    {n : Nat}
    {AuthorizedPrior : Source → ColumnDigits n → Prop}
    {ParentSourceStep :
      Nat →
        AccumulatorHandle Source →
        ColumnDigits n →
        Source →
          Prop}
    {i : Nat}
    {prior nextA nextB : AccumulatorHandle Source}
    (hAuthorized :
      ReducedAccumulatorStep.AuthorizedFunctional AuthorizedPrior)
    (hA : Step AuthorizedPrior ParentSourceStep i prior nextA)
    (hB : Step AuthorizedPrior ParentSourceStep i prior nextB) :
    ∃ priorInputs,
      AuthorizedPrior prior.parentSource priorInputs ∧
        ParentSourceStep i prior priorInputs nextA.parentSource ∧
        ParentSourceStep i prior priorInputs nextB.parentSource := by
  rcases hA with ⟨inputsA, hAuthA, hParentA⟩
  rcases hB with ⟨inputsB, hAuthB, hParentB⟩
  have hInputs : inputsA = inputsB :=
    hAuthorized prior.parentSource inputsA inputsB hAuthA hAuthB
  subst inputsB
  exact ⟨inputsA, hAuthA, hParentA, hParentB⟩

/--
Adapter from the existing child-carrying stage interface to the parent-only
step. The privately authorized child table is placed into the child-carrying
handle used by the existing `Pi_CCS` relation, but it is not part of the
parent-only public handle.
-/
def ParentSourceFromPiStages
    {Source PiCCSOut : Type}
    {n : Nat}
    (PiCCS :
      Nat →
        ReducedAccumulatorStep.AccumulatorHandle Source n →
        PiCCSOut →
          Prop)
    (PiRLC : Nat → PiCCSOut → Source → Prop)
    (i : Nat)
    (prior : AccumulatorHandle Source)
    (priorInputs : ColumnDigits n)
    (source : Source) : Prop :=
  ∃ piCCSOutput,
    PiCCS
        i
        { parentSource := prior.parentSource
          nextPiCCSInputs := priorInputs }
        piCCSOutput ∧
      PiRLC i piCCSOutput source

/--
Functional `Pi_CCS` and `Pi_RLC` stage relations induce a functional
parent-source computation for the parent-only step.
-/
theorem parentSourceFromPiStages_functional
    {Source PiCCSOut : Type}
    {n : Nat}
    {PiCCS :
      Nat →
        ReducedAccumulatorStep.AccumulatorHandle Source n →
        PiCCSOut →
          Prop}
    {PiRLC : Nat → PiCCSOut → Source → Prop}
    (hPiCCS : ParentSourceStep.PiCCSFunctional PiCCS)
    (hPiRLC : ParentSourceStep.PiRLCFunctional PiRLC) :
    ParentSourceFunctional
      (ParentSourceFromPiStages (n := n) PiCCS PiRLC) := by
  intro i prior priorInputs sourceA sourceB hA hB
  rcases hA with ⟨outA, hPiCCSA, hPiRLCA⟩
  rcases hB with ⟨outB, hPiCCSB, hPiRLCB⟩
  have hOut : outA = outB :=
    hPiCCS
      i
      { parentSource := prior.parentSource
        nextPiCCSInputs := priorInputs }
      outA
      outB
      hPiCCSA
      hPiCCSB
  subst outB
  exact hPiRLC i outA sourceA sourceB hPiRLCA hPiRLCB

/--
Named certificate for pointwise private-DEC authorization.

The parent-only public handle hides the post-DEC children, so this certificate
is the private evidence that those children are not arbitrary advice. It keeps
the arithmetic DEC facts, CE witness-table identity, and next-`Pi_CCS` wire
identity as named fields for auditability.
-/
structure PointwisePrivateDecCertificate
    {n : Nat}
    {Digest : Type}
    {hashEncoded : List Nat → Digest}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment}
    {StatementEncodes :
      ParentOpeningAuthorization.StatementEncodesParentCEB
        SuperNeo.ProofSystem.Commitment}
    (source : DigestParentBinding.Source Digest)
    (nextInputs : ColumnDigits n) where
  parentResidues : Fin n → Nat
  bundle : SuperNeoBridge.AjtaiCEChildBundle 14 n
  proof : Unit
  accepted :
    PrivatePiDecSoundness.AcceptedPrivatePiDec
          (ParentOpeningAuthorization.BindsOpenedParentCEBResiduesFor
            (n := n)
            hashEncoded
            ce
            StatementEncodes)
          (CanonicalPrivatePiDecVerifier.Verify
            (n := n)
            (Source := DigestParentBinding.Source Digest))
          source
          parentResidues
          bundle
          proof
  fixedCE : bundle.ce = ce
  fixedAjtaiParams : bundle.ajtaiParams = params
  binaryDigits : DecDigitUniqueness.binaryColumnDigits bundle.digitTable
  length14 : BinaryChildTableAuthorization.fixedColumnLength 14 bundle.digitTable
  recomposesToParent :
    ∀ j,
      DecDigitUniqueness.recomposeNatDigits (bundle.digitTable j) %
          SuperNeo.Goldilocks.q =
        parentResidues j % SuperNeo.Goldilocks.q
  witnessTable :
    bundle.digitTable =
      SuperNeoBridge.childWitnessDigitTable
        (k := 14)
        (n := n)
        bundle.witness
  wireIdentity : bundle.nextPiCCSInputs = bundle.digitTable
  nextInputIdentity : nextInputs = bundle.nextPiCCSInputs

/--
Pointwise private-DEC requirements exposed by an authorized prior child table.

This is intentionally stronger than any aggregate norm or checksum statement:
it requires a named certificate with binary digits, exact length, per-column
recomposition, child witness-table identity, and wire identity into the next
`Pi_CCS` inputs.
-/
def PointwisePrivateDecRequirements
    {n : Nat}
    {Digest : Type}
    {hashEncoded : List Nat → Digest}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment}
    {StatementEncodes :
      ParentOpeningAuthorization.StatementEncodesParentCEB
        SuperNeo.ProofSystem.Commitment}
    (source : DigestParentBinding.Source Digest)
    (nextInputs : ColumnDigits n) : Prop :=
  Nonempty
    (PointwisePrivateDecCertificate
      (n := n)
      (hashEncoded := hashEncoded)
      (params := params)
      (ce := ce)
      (StatementEncodes := StatementEncodes)
      source
      nextInputs)

/--
Named certificate for fixed-CE membership of the private DEC children.

This packages the real SuperNeo child-membership facts carried by the private
child bundle: the child CE relation and Ajtai parameters are the fixed
production objects, every child satisfies that CE relation, every child opening
opens under the fixed Ajtai parameters, and the `nextPiCCSInputs` table is the
CE witness-derived child table.
-/
structure FixedCEChildMembershipCertificate
    {n : Nat}
    (params : SuperNeo.ProofSystem.AjtaiParams)
    (ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment)
    (nextInputs : ColumnDigits n) where
  bundle : SuperNeoBridge.AjtaiCEChildBundle 14 n
  fixedCE : bundle.ce = ce
  fixedAjtaiParams : bundle.ajtaiParams = params
  nextInputIdentity : nextInputs = bundle.nextPiCCSInputs
  witnessTable :
    bundle.digitTable =
      SuperNeoBridge.childWitnessDigitTable
        (k := 14)
        (n := n)
        bundle.witness
  wireIdentity : bundle.nextPiCCSInputs = bundle.digitTable
  ceHolds :
    ∀ i,
      SuperNeo.ProofSystem.ConstraintSystem.CE.Holds
        ce
        (bundle.statement i)
        (bundle.witness i)
  ajtaiOpens :
    ∀ i,
      SuperNeo.ProofSystem.opensTo
        params
        (bundle.statement i).commitment
        (bundle.opening i)

/--
Fixed-CE membership for the private DEC children wired into the next `Pi_CCS`.
-/
def FixedCEChildMembership
    {n : Nat}
    (params : SuperNeo.ProofSystem.AjtaiParams)
    (ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment)
    (nextInputs : ColumnDigits n) : Prop :=
  Nonempty (FixedCEChildMembershipCertificate params ce nextInputs)

/--
Every canonical private-DEC authorization exposes the pointwise requirements
needed by the parent-only step.
-/
theorem pointwise_private_dec_requirements_of_authorized
    {n : Nat}
    {Digest : Type}
    {hashEncoded : List Nat → Digest}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment}
    {StatementEncodes :
      ParentOpeningAuthorization.StatementEncodesParentCEB
        SuperNeo.ProofSystem.Commitment}
    {source : DigestParentBinding.Source Digest}
    {nextInputs : ColumnDigits n}
    (hAuthorized :
      CanonicalPrivatePiDecVerifier.AuthorizedNextPiCCSInputs
        (n := n)
        (hashEncoded := hashEncoded)
        (params := params)
        (ce := ce)
        (StatementEncodes := StatementEncodes)
        source
        nextInputs) :
    PointwisePrivateDecRequirements
      (n := n)
      (hashEncoded := hashEncoded)
      (params := params)
      (ce := ce)
      (StatementEncodes := StatementEncodes)
      source
      nextInputs := by
  rcases hAuthorized with
    ⟨parentResidues, bundle, proof, hAccepted, hCE, hParams, hNextInputs⟩
  exact ⟨{
    parentResidues := parentResidues
    bundle := bundle
    proof := proof
    accepted := hAccepted
    fixedCE := hCE
    fixedAjtaiParams := hParams
    binaryDigits := hAccepted.proofVerified.1
    length14 := hAccepted.proofVerified.2.1
    recomposesToParent := hAccepted.proofVerified.2.2
    witnessTable := bundle.digitTableMatchesWitnesses
    wireIdentity := hAccepted.wireIdentity
    nextInputIdentity := hNextInputs
  }⟩

/--
Pointwise private-DEC requirements expose fixed-CE membership for the private
children wired into the next `Pi_CCS`.
-/
theorem fixedCEChildMembership_of_pointwise_private_dec_requirements
    {n : Nat}
    {Digest : Type}
    {hashEncoded : List Nat → Digest}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment}
    {StatementEncodes :
      ParentOpeningAuthorization.StatementEncodesParentCEB
        SuperNeo.ProofSystem.Commitment}
    {source : DigestParentBinding.Source Digest}
    {nextInputs : ColumnDigits n}
    (hPointwise :
      PointwisePrivateDecRequirements
        (n := n)
        (hashEncoded := hashEncoded)
        (params := params)
        (ce := ce)
        (StatementEncodes := StatementEncodes)
        source
        nextInputs) :
    FixedCEChildMembership params ce nextInputs := by
  rcases hPointwise with ⟨cert⟩
  refine ⟨{
    bundle := cert.bundle
    fixedCE := cert.fixedCE
    fixedAjtaiParams := cert.fixedAjtaiParams
    nextInputIdentity := cert.nextInputIdentity
    witnessTable := cert.witnessTable
    wireIdentity := cert.wireIdentity
    ceHolds := ?_
    ajtaiOpens := ?_
  }⟩
  · intro i
    simpa [cert.fixedCE] using cert.bundle.ceHolds i
  · intro i
    simpa [cert.fixedAjtaiParams] using cert.bundle.ajtaiOpens i

/--
Canonical private `Pi_DEC` authorization exposes fixed-CE membership for the
children it wires into the next `Pi_CCS`.
-/
theorem fixedCEChildMembership_of_authorized
    {n : Nat}
    {Digest : Type}
    {hashEncoded : List Nat → Digest}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment}
    {StatementEncodes :
      ParentOpeningAuthorization.StatementEncodesParentCEB
        SuperNeo.ProofSystem.Commitment}
    {source : DigestParentBinding.Source Digest}
    {nextInputs : ColumnDigits n}
    (hAuthorized :
      CanonicalPrivatePiDecVerifier.AuthorizedNextPiCCSInputs
        (n := n)
        (hashEncoded := hashEncoded)
        (params := params)
        (ce := ce)
        (StatementEncodes := StatementEncodes)
        source
        nextInputs) :
    FixedCEChildMembership params ce nextInputs :=
    fixedCEChildMembership_of_pointwise_private_dec_requirements
    (pointwise_private_dec_requirements_of_authorized hAuthorized)

/--
Named audit for the pointwise private-DEC no-swap theorem.

The final `requestedInputs` equality is the theorem used by accumulator
functionality. The preceding equalities record why that equality is meaningful:
the opened parent residues agree, the full hidden DEC digit table agrees
pointwise, the table extracted from CE witnesses agrees, and the bundle wires
feeding the next `Pi_CCS` agree.
-/
structure PrivateDecNoSwapAudit
    {n : Nat}
    {Digest : Type}
    {hashEncoded : List Nat → Digest}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment}
    {StatementEncodes :
      ParentOpeningAuthorization.StatementEncodesParentCEB
        SuperNeo.ProofSystem.Commitment}
    {source : DigestParentBinding.Source Digest}
    {nextA nextB : ColumnDigits n}
    (certA :
      PointwisePrivateDecCertificate
        (n := n)
        (hashEncoded := hashEncoded)
        (params := params)
        (ce := ce)
        (StatementEncodes := StatementEncodes)
        source
        nextA)
    (certB :
      PointwisePrivateDecCertificate
        (n := n)
        (hashEncoded := hashEncoded)
        (params := params)
        (ce := ce)
        (StatementEncodes := StatementEncodes)
        source
        nextB) : Prop where
  parentResidues :
    certA.parentResidues = certB.parentResidues
  digitTable :
    certA.bundle.digitTable = certB.bundle.digitTable
  witnessDigitTable :
    SuperNeoBridge.childWitnessDigitTable
        (k := 14)
        (n := n)
        certA.bundle.witness =
      SuperNeoBridge.childWitnessDigitTable
        (k := 14)
        (n := n)
        certB.bundle.witness
  bundleInputs :
    certA.bundle.nextPiCCSInputs = certB.bundle.nextPiCCSInputs
  requestedInputs :
    nextA = nextB

/--
Certificate-level no-swap audit for two pointwise private-DEC witnesses.

This is stronger than the accumulator-facing functionality theorem: it exposes
the concrete equalities that rule out hidden child-table substitution, rather
than only returning the final `nextInputs` equality.
-/
theorem privateDecNoSwapAudit_of_certificates
    {n : Nat}
    {Digest : Type}
    {hashEncoded : List Nat → Digest}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment}
    {StatementEncodes :
      ParentOpeningAuthorization.StatementEncodesParentCEB
        SuperNeo.ProofSystem.Commitment}
    {source : DigestParentBinding.Source Digest}
    {nextA nextB : ColumnDigits n}
    (hDigest :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hEncoding :
      ParentOpeningAuthorization.StatementEncodingCommitmentFunctional
        StatementEncodes)
    (hNoCollision :
      AjtaiResidueBinding.NoAjtaiBindingCollision params)
    (adapter :
      AjtaiResidueBinding.CEOpeningAdapter
        n
        params
        ce)
    (certA :
      PointwisePrivateDecCertificate
        (n := n)
        (hashEncoded := hashEncoded)
        (params := params)
        (ce := ce)
        (StatementEncodes := StatementEncodes)
        source
        nextA)
    (certB :
      PointwisePrivateDecCertificate
        (n := n)
        (hashEncoded := hashEncoded)
        (params := params)
        (ce := ce)
        (StatementEncodes := StatementEncodes)
        source
        nextB) :
    PrivateDecNoSwapAudit certA certB := by
  have hResidueBinding :
      ParentOpeningAuthorization.EncodedParentCEBOpeningResiduesFunctionalFor
        (n := n)
        ce
        StatementEncodes :=
    AjtaiResidueBinding.encodedParentCEBOpeningResiduesFunctionalFor_of_noAjtaiBindingCollision
      hEncoding
      hNoCollision
      adapter
  have hParentFunctional :
      GoldilocksChildTableAuthorization.SourceBindsParentFunctionally
        (ParentOpeningAuthorization.BindsOpenedParentCEBResiduesFor
          (n := n)
          hashEncoded
          ce
          StatementEncodes) :=
    ParentOpeningAuthorization.bindsOpenedParentCEBResiduesFor_functionally
      hDigest
      hResidueBinding
  have hParent :
      certA.parentResidues = certB.parentResidues :=
    hParentFunctional
      source
      certA.parentResidues
      certB.parentResidues
      certA.accepted.sourceBound
      certB.accepted.sourceBound
  have hInputs :
      certA.bundle.nextPiCCSInputs = certB.bundle.nextPiCCSInputs :=
    CanonicalPrivatePiDecVerifier.same_opened_parentCEB_digest_source_canonical_private_pidec_inputs_of_ajtaiCEOpening
      hDigest
      hEncoding
      hNoCollision
      adapter
      certA.accepted
      certB.accepted
  have hDigits :
      certA.bundle.digitTable = certB.bundle.digitTable := by
    calc
      certA.bundle.digitTable = certA.bundle.nextPiCCSInputs :=
        certA.wireIdentity.symm
      _ = certB.bundle.nextPiCCSInputs := hInputs
      _ = certB.bundle.digitTable := certB.wireIdentity
  have hWitnessDigits :
      SuperNeoBridge.childWitnessDigitTable
          (k := 14)
          (n := n)
          certA.bundle.witness =
        SuperNeoBridge.childWitnessDigitTable
          (k := 14)
          (n := n)
          certB.bundle.witness := by
    calc
      SuperNeoBridge.childWitnessDigitTable
            (k := 14)
            (n := n)
            certA.bundle.witness =
          certA.bundle.digitTable := certA.witnessTable.symm
      _ = certB.bundle.digitTable := hDigits
      _ =
          SuperNeoBridge.childWitnessDigitTable
            (k := 14)
            (n := n)
            certB.bundle.witness := certB.witnessTable
  have hRequested :
      nextA = nextB := by
    calc
      nextA = certA.bundle.nextPiCCSInputs := certA.nextInputIdentity
      _ = certB.bundle.nextPiCCSInputs := hInputs
      _ = nextB := certB.nextInputIdentity.symm
  exact
    { parentResidues := hParent
      digitTable := hDigits
      witnessDigitTable := hWitnessDigits
      bundleInputs := hInputs
      requestedInputs := hRequested }

/--
Requirement-level no-swap audit for two accepted pointwise private-DEC tables.
-/
theorem privateDecNoSwapAudit_of_requirements
    {n : Nat}
    {Digest : Type}
    {hashEncoded : List Nat → Digest}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment}
    {StatementEncodes :
      ParentOpeningAuthorization.StatementEncodesParentCEB
        SuperNeo.ProofSystem.Commitment}
    {source : DigestParentBinding.Source Digest}
    {nextA nextB : ColumnDigits n}
    (hDigest :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hEncoding :
      ParentOpeningAuthorization.StatementEncodingCommitmentFunctional
        StatementEncodes)
    (hNoCollision :
      AjtaiResidueBinding.NoAjtaiBindingCollision params)
    (adapter :
      AjtaiResidueBinding.CEOpeningAdapter
        n
        params
        ce)
    (hA :
      PointwisePrivateDecRequirements
        (n := n)
        (hashEncoded := hashEncoded)
        (params := params)
        (ce := ce)
        (StatementEncodes := StatementEncodes)
        source
        nextA)
    (hB :
      PointwisePrivateDecRequirements
        (n := n)
        (hashEncoded := hashEncoded)
        (params := params)
        (ce := ce)
        (StatementEncodes := StatementEncodes)
        source
        nextB) :
    ∃
      (certA :
        PointwisePrivateDecCertificate
          (n := n)
          (hashEncoded := hashEncoded)
          (params := params)
          (ce := ce)
          (StatementEncodes := StatementEncodes)
          source
          nextA)
      (certB :
        PointwisePrivateDecCertificate
          (n := n)
          (hashEncoded := hashEncoded)
          (params := params)
          (ce := ce)
          (StatementEncodes := StatementEncodes)
          source
          nextB),
        PrivateDecNoSwapAudit certA certB := by
  rcases hA with ⟨certA⟩
  rcases hB with ⟨certB⟩
  exact
    ⟨certA,
      certB,
      privateDecNoSwapAudit_of_certificates
        hDigest
        hEncoding
        hNoCollision
        adapter
        certA
        certB⟩

/--
Pointwise private-DEC requirements are functional for one parent source under
the same binding assumptions used by canonical private `Pi_DEC`.

This is the direct anti-substitution statement for the exposed pointwise
requirements: two accepted private child tables for the same compact parent
source cannot differ while still satisfying the canonical DEC relation,
child-table witness identity, and next-`Pi_CCS` wire identity.
-/
theorem pointwise_private_dec_requirements_functional_of_ajtaiCEOpening
    {n : Nat}
    {Digest : Type}
    {hashEncoded : List Nat → Digest}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment}
    {StatementEncodes :
      ParentOpeningAuthorization.StatementEncodesParentCEB
        SuperNeo.ProofSystem.Commitment}
    {source : DigestParentBinding.Source Digest}
    {nextA nextB : ColumnDigits n}
    (hDigest :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hEncoding :
      ParentOpeningAuthorization.StatementEncodingCommitmentFunctional
        StatementEncodes)
    (hNoCollision :
      AjtaiResidueBinding.NoAjtaiBindingCollision params)
    (adapter :
      AjtaiResidueBinding.CEOpeningAdapter
        n
        params
        ce)
    (hA :
      PointwisePrivateDecRequirements
        (n := n)
        (hashEncoded := hashEncoded)
        (params := params)
        (ce := ce)
        (StatementEncodes := StatementEncodes)
        source
        nextA)
    (hB :
      PointwisePrivateDecRequirements
        (n := n)
        (hashEncoded := hashEncoded)
        (params := params)
        (ce := ce)
        (StatementEncodes := StatementEncodes)
        source
        nextB) :
    nextA = nextB := by
  rcases
      privateDecNoSwapAudit_of_requirements
        hDigest
        hEncoding
        hNoCollision
        adapter
        hA
        hB with
    ⟨_certA, _certB, hAudit⟩
  exact hAudit.requestedInputs

/--
Implementation-shaped functionality for pointwise private-DEC requirements
with deterministic parent-statement commitment encoding.
-/
theorem pointwise_private_dec_requirements_functional_of_statementCommitment_and_ajtaiCEOpening
    {n : Nat}
    {Digest : Type}
    {hashEncoded : List Nat → Digest}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment}
    {commitmentOfParent :
      ParentEncoding.SomeParentCEB →
        SuperNeo.ProofSystem.Commitment}
    {source : DigestParentBinding.Source Digest}
    {nextA nextB : ColumnDigits n}
    (hDigest :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hNoCollision :
      AjtaiResidueBinding.NoAjtaiBindingCollision params)
    (adapter :
      AjtaiResidueBinding.CEOpeningAdapter
        n
        params
        ce)
    (hA :
      PointwisePrivateDecRequirements
        (n := n)
        (hashEncoded := hashEncoded)
        (params := params)
        (ce := ce)
        (StatementEncodes :=
          ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent)
        source
        nextA)
    (hB :
      PointwisePrivateDecRequirements
        (n := n)
        (hashEncoded := hashEncoded)
        (params := params)
        (ce := ce)
        (StatementEncodes :=
          ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent)
        source
        nextB) :
    nextA = nextB :=
  pointwise_private_dec_requirements_functional_of_ajtaiCEOpening
    hDigest
    ParentOpeningAuthorization.statementEncodesByCommitment_functional
    hNoCollision
    adapter
    hA
    hB

/--
Every accepted parent-only step using canonical private `Pi_DEC` carries a
pointwise-authorized prior child table for the parent-source computation.
-/
theorem pointwise_prior_dec_requirements_of_step
    {n : Nat}
    {Digest : Type}
    {hashEncoded : List Nat → Digest}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment}
    {StatementEncodes :
      ParentOpeningAuthorization.StatementEncodesParentCEB
        SuperNeo.ProofSystem.Commitment}
    {ParentSourceStep :
      Nat →
        AccumulatorHandle (DigestParentBinding.Source Digest) →
        ColumnDigits n →
        DigestParentBinding.Source Digest →
          Prop}
    {i : Nat}
    {prior next : AccumulatorHandle (DigestParentBinding.Source Digest)}
    (hStep :
      Step
        (CanonicalPrivatePiDecVerifier.AuthorizedNextPiCCSInputs
          (n := n)
          (hashEncoded := hashEncoded)
          (params := params)
          (ce := ce)
          (StatementEncodes := StatementEncodes))
        ParentSourceStep
        i
        prior
        next) :
    ∃ priorInputs,
      PointwisePrivateDecRequirements
        (n := n)
        (hashEncoded := hashEncoded)
        (params := params)
        (ce := ce)
        (StatementEncodes := StatementEncodes)
        prior.parentSource
        priorInputs ∧
      ParentSourceStep i prior priorInputs next.parentSource := by
  rcases hStep with ⟨priorInputs, hAuthorized, hParent⟩
  exact
    ⟨priorInputs,
      pointwise_private_dec_requirements_of_authorized hAuthorized,
      hParent⟩

/--
Two accepted canonical parent-only steps from the same prior handle expose one
common pointwise-authorized hidden child table.
-/
theorem pointwise_common_prior_dec_requirements_of_steps
    {n : Nat}
    {Digest : Type}
    {hashEncoded : List Nat → Digest}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment}
    {StatementEncodes :
      ParentOpeningAuthorization.StatementEncodesParentCEB
        SuperNeo.ProofSystem.Commitment}
    {ParentSourceStep :
      Nat →
        AccumulatorHandle (DigestParentBinding.Source Digest) →
        ColumnDigits n →
        DigestParentBinding.Source Digest →
          Prop}
    {i : Nat}
    {prior nextA nextB :
      AccumulatorHandle (DigestParentBinding.Source Digest)}
    (hAuthorized :
      ReducedAccumulatorStep.AuthorizedFunctional
        (CanonicalPrivatePiDecVerifier.AuthorizedNextPiCCSInputs
          (n := n)
          (hashEncoded := hashEncoded)
          (params := params)
          (ce := ce)
          (StatementEncodes := StatementEncodes)))
    (hA :
      Step
        (CanonicalPrivatePiDecVerifier.AuthorizedNextPiCCSInputs
          (n := n)
          (hashEncoded := hashEncoded)
          (params := params)
          (ce := ce)
          (StatementEncodes := StatementEncodes))
        ParentSourceStep
        i
        prior
        nextA)
    (hB :
      Step
        (CanonicalPrivatePiDecVerifier.AuthorizedNextPiCCSInputs
          (n := n)
          (hashEncoded := hashEncoded)
          (params := params)
          (ce := ce)
          (StatementEncodes := StatementEncodes))
        ParentSourceStep
        i
        prior
        nextB) :
    ∃ priorInputs,
      PointwisePrivateDecRequirements
        (n := n)
        (hashEncoded := hashEncoded)
        (params := params)
        (ce := ce)
        (StatementEncodes := StatementEncodes)
        prior.parentSource
        priorInputs ∧
        ParentSourceStep i prior priorInputs nextA.parentSource ∧
        ParentSourceStep i prior priorInputs nextB.parentSource := by
  rcases step_common_authorized_inputs hAuthorized hA hB with
    ⟨priorInputs, hAuth, hParentA, hParentB⟩
  exact
    ⟨priorInputs,
      pointwise_private_dec_requirements_of_authorized hAuth,
      hParentA,
      hParentB⟩

/--
Canonical parent-only latest-step functionality under the same concrete
binding assumptions as the child-carrying reduced-handle theorem.
-/
theorem step_parentSource_functional_of_stages_and_ajtaiCEOpening
    {Digest PiCCSOut : Type}
    {n : Nat}
    {hashEncoded : List Nat → Digest}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment}
    {StatementEncodes :
      ParentOpeningAuthorization.StatementEncodesParentCEB
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
      AccumulatorHandle (DigestParentBinding.Source Digest)}
    (hPiCCS : ParentSourceStep.PiCCSFunctional PiCCS)
    (hPiRLC : ParentSourceStep.PiRLCFunctional PiRLC)
    (hDigest :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hEncoding :
      ParentOpeningAuthorization.StatementEncodingCommitmentFunctional
        StatementEncodes)
    (hNoCollision :
      AjtaiResidueBinding.NoAjtaiBindingCollision params)
    (adapter :
      AjtaiResidueBinding.CEOpeningAdapter
        n
        params
        ce)
    (hA :
      Step
        (CanonicalPrivatePiDecVerifier.AuthorizedNextPiCCSInputs
          (n := n)
          (hashEncoded := hashEncoded)
          (params := params)
          (ce := ce)
          (StatementEncodes := StatementEncodes))
        (ParentSourceFromPiStages
          (n := n)
          PiCCS
          PiRLC)
        i
        prior
        nextA)
    (hB :
      Step
        (CanonicalPrivatePiDecVerifier.AuthorizedNextPiCCSInputs
          (n := n)
          (hashEncoded := hashEncoded)
          (params := params)
          (ce := ce)
          (StatementEncodes := StatementEncodes))
        (ParentSourceFromPiStages
          (n := n)
          PiCCS
          PiRLC)
        i
        prior
        nextB) :
    nextA.parentSource = nextB.parentSource :=
  step_parentSource_functional
    (ReducedAccumulatorStep.canonical_authorized_functional_of_ajtaiCEOpening
      hDigest
      hEncoding
      hNoCollision
      adapter)
    (parentSourceFromPiStages_functional hPiCCS hPiRLC)
    hA
    hB

/--
Implementation-shaped parent-only latest-step functionality for deterministic
statement commitment encoding.
-/
theorem step_parentSource_functional_of_statementCommitment_stages_and_ajtaiCEOpening
    {Digest PiCCSOut : Type}
    {n : Nat}
    {hashEncoded : List Nat → Digest}
    {params : SuperNeo.ProofSystem.AjtaiParams}
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
      AccumulatorHandle (DigestParentBinding.Source Digest)}
    (hPiCCS : ParentSourceStep.PiCCSFunctional PiCCS)
    (hPiRLC : ParentSourceStep.PiRLCFunctional PiRLC)
    (hDigest :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hNoCollision :
      AjtaiResidueBinding.NoAjtaiBindingCollision params)
    (adapter :
      AjtaiResidueBinding.CEOpeningAdapter
        n
        params
        ce)
    (hA :
      Step
        (CanonicalPrivatePiDecVerifier.AuthorizedNextPiCCSInputs
          (n := n)
          (hashEncoded := hashEncoded)
          (params := params)
          (ce := ce)
          (StatementEncodes :=
            ParentOpeningAuthorization.StatementEncodesByCommitment
              commitmentOfParent))
        (ParentSourceFromPiStages
          (n := n)
          PiCCS
          PiRLC)
        i
        prior
        nextA)
    (hB :
      Step
        (CanonicalPrivatePiDecVerifier.AuthorizedNextPiCCSInputs
          (n := n)
          (hashEncoded := hashEncoded)
          (params := params)
          (ce := ce)
          (StatementEncodes :=
            ParentOpeningAuthorization.StatementEncodesByCommitment
              commitmentOfParent))
        (ParentSourceFromPiStages
          (n := n)
          PiCCS
          PiRLC)
        i
        prior
        nextB) :
    nextA.parentSource = nextB.parentSource :=
  step_parentSource_functional
    (ReducedAccumulatorStep.canonical_authorized_functional_of_statementCommitment_and_ajtaiCEOpening
      hDigest
      hNoCollision
      adapter)
    (parentSourceFromPiStages_functional hPiCCS hPiRLC)
    hA
    hB

end ParentOnlyAccumulatorStep

end DirectCcsFPrime
