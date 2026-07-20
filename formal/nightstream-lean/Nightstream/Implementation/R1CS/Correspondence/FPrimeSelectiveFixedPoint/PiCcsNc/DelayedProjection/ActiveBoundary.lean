import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PendingErasure
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RefinementBoundary
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Construction2
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical.Physical

/-!
Active fixed-one boundary for production delayed packed-`y_zcol` authority.

Assurance tier: model-level until generated full-witness, combined-NC,
transcript, accumulator-state, and terminal-opening rows refine `Accepted`.

Owns: the exact Lean contract that a production fixed-one verifier must
execute; canonical output construction from that accepted certificate; the
one-fold adjacent composition which proves the previous active F-prime step;
and terminal closure of the final pending step.

Does not own: a Rust implementation, generated rows, full-`Z` commitment
openings, Poseidon2 or Ajtai internals, probability bounds, `y_ring`, costs, or
row-removal permission. A digest is never treated as authority: both sides of
an accumulator edge recompute the complete parent/children/pending payload,
and binding failure remains an explicit result branch.

Emits constraints: none; executable/refinement contract only.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.active.outer` | execute the retained fixed-one outer equations | checked |
| `f_prime.pi_ccs_nc.delayed.active.raw_z` | execute combined NC over complete packed running witnesses | checked boundary |
| `f_prime.pi_ccs_nc.delayed.active.state.in` | recompute the incoming parent/children/pending accumulator coordinate | checked/security boundary |
| `f_prime.pi_ccs_nc.delayed.active.state.out` | recompute the outgoing parent/children/pending accumulator coordinate | checked/security boundary |
| `f_prime.pi_ccs_nc.delayed.active.recursive` | the successor closes the previous active result | derived/security partition |
| `f_prime.pi_ccs_nc.delayed.active.terminal` | the terminal opening closes the final active result | derived/security partition |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveBoundary

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane
open Nightstream.SuperNeo.SumCheck.Finite
open Nightstream.Protocol.FPrime.ConcretePhi81
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator
open Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.DelayedPending
open Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual
open Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc
open PackedWitness

universe uOuterKey uAppState uWitness uDigest uTranscriptState uEncoding

variable {OuterKey : Type uOuterKey}
variable {AppState : Type uAppState}
variable {Witness : Type uWitness}
variable {Digest : Type uDigest}
variable {TranscriptState : Type uTranscriptState}
variable {Encoding : Type uEncoding}
variable {shape : SemanticShape}
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Exact selected result computed by the production pending-aware context. -/
def resultOf
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (certificate : FixedActive.Certificate
      (ProductionContext.full setup input)) :
    FixedActive.FoldResult shape publicRingColumns publicFits verifierRows :=
  FixedActive.resultOf (ProductionContext.full setup input) certificate

/-- Canonical rich active output. Production checks the input and certificate;
the verifier computes this value rather than accepting an output witness. -/
def outputOf
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (certificate : FixedActive.Certificate
      (ProductionContext.full setup input)) :
    Output Digest AppState shape publicRingColumns publicFits verifierRows 1 :=
  ActiveSemantics.outputOf machine (input.fixedOne.toActive setup)
    ActiveSemantics.FixedOneCanonical.selected
    (resultOf setup input certificate)

/-- Executable production-step refinement contract.

`template` plus `witnesses` decode every running assignment from complete
packed matrices. The two state checks recompute the typed payload on each side
of the recursive edge. A caller cannot satisfy this contract with
`CeClaim.y_zcol` sidecars or an unqualified digest equality. -/
structure Accepted
    [DecidableEq Digest]
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (incomingStateDigest outgoingStateDigest : Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (template : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape)
    (certificate : FixedActive.Certificate
      (ProductionContext.full setup input)) : Prop where
  outer : FixedOneCanonical.outerCheck machine setup input.fixedOne = true
  nifs : PackedWitnessProduction.check
      (ProductionContext.canonical setup input) template witnesses
      certificate = true
  incomingState :
    ProductionChecker.stateBindingCheck scheme incomingStateDigest
        ((ProductionContext.canonical setup input).input.parent.materialize
          (ProductionContext.canonical setup input).input.system)
        (ProductionContext.full setup input).input.running input.pending = true
  outgoingState :
    ProductionChecker.stateBindingCheck scheme outgoingStateDigest
        (derive (ProductionContext.full setup input) certificate).piRlcOutput
        (outputChildren (ProductionContext.full setup input) certificate)
        (some (DelayedProduction.outgoingPending
          (ProductionContext.full setup input) certificate)) = true

/-- Actual claims-level production-step contract. Unlike `Accepted`, the NIFS
check reads no private `Sources.Data`; it computes the combined terminal from
the public Π_CCS output message. Complete matrices are retained only for the
subsequent extraction/refinement partition. -/
structure ClaimsAccepted
    [DecidableEq Digest]
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (incomingStateDigest outgoingStateDigest : Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (template : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape)
    (certificate : FixedActive.Certificate
      (ProductionContext.full setup input)) : Prop where
  outer : FixedOneCanonical.outerCheck machine setup input.fixedOne = true
  nifs : PackedWitnessProduction.messageCheck
      (ProductionContext.canonical setup input) certificate = true
  incomingState :
    ProductionChecker.stateBindingCheck scheme incomingStateDigest
        ((ProductionContext.canonical setup input).input.parent.materialize
          (ProductionContext.canonical setup input).input.system)
        (ProductionContext.full setup input).input.running input.pending = true
  outgoingState :
    ProductionChecker.stateBindingCheck scheme outgoingStateDigest
        (derive (ProductionContext.full setup input) certificate).piRlcOutput
        (outputChildren (ProductionContext.full setup input) certificate)
        (some (DelayedProduction.outgoingPending
          (ProductionContext.full setup input) certificate)) = true

/-- Executable claims-level acceptance contract. This is the Boolean seam that
the native and R1CS verifiers must refine; it consumes only verifier-visible
claims, transcript output, and recomputed accumulator payloads. The private raw
matrices remain outside this check and enter only through extraction/binding. -/
def claimsCheck
    [DecidableEq Digest]
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (incomingStateDigest outgoingStateDigest : Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (certificate : FixedActive.Certificate
      (ProductionContext.full setup input)) : Bool :=
  FixedOneCanonical.outerCheck machine setup input.fixedOne &&
    (PackedWitnessProduction.messageCheck
      (ProductionContext.canonical setup input) certificate &&
    (ProductionChecker.stateBindingCheck scheme incomingStateDigest
        ((ProductionContext.canonical setup input).input.parent.materialize
          (ProductionContext.canonical setup input).input.system)
        (ProductionContext.full setup input).input.running input.pending &&
      ProductionChecker.stateBindingCheck scheme outgoingStateDigest
        (derive (ProductionContext.full setup input) certificate).piRlcOutput
        (outputChildren (ProductionContext.full setup input) certificate)
        (some (DelayedProduction.outgoingPending
          (ProductionContext.full setup input) certificate))))

/-- The executable claims check is exactly the structured production contract.
No projection, raw-witness equality, or semantic acceptance is assumed. -/
theorem claimsCheck_eq_true_iff
    [DecidableEq Digest]
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (incomingStateDigest outgoingStateDigest : Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (template : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape)
    (certificate : FixedActive.Certificate
      (ProductionContext.full setup input)) :
    claimsCheck scheme incomingStateDigest outgoingStateDigest machine setup
        input certificate = true <->
      ClaimsAccepted scheme incomingStateDigest outgoingStateDigest machine
        setup input template witnesses certificate := by
  simp only [claimsCheck, Bool.and_eq_true]
  constructor
  · rintro ⟨outer, nifs, incomingState, outgoingState⟩
    exact ⟨outer, nifs, incomingState, outgoingState⟩
  · rintro ⟨outer, nifs, incomingState, outgoingState⟩
    exact ⟨outer, nifs, incomingState, outgoingState⟩

namespace Accepted

/-- Boolean outer acceptance exposes the exact two fixed-one equations. -/
theorem outerChecks
    [DecidableEq Digest]
    {scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest}
    {incomingStateDigest outgoingStateDigest : Digest}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1}
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1}
    {input :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows}
    {template : Data shape}
    {witnesses : Fin shape.runningCount -> Matrix shape}
    {certificate : FixedActive.Certificate
      (ProductionContext.full setup input)}
    (accepted : Accepted scheme incomingStateDigest outgoingStateDigest
      machine setup input template witnesses certificate) :
    FixedOneCanonical.OuterChecks machine setup input.fixedOne :=
  (FixedOneCanonical.outerCheck_eq_true_iff machine setup input.fixedOne).mp
    accepted.outer

end Accepted

namespace ClaimsAccepted

/-- The public production contract refines to the existing post-extraction
contract or exposes exactly the current Π_CCS packed-output opening failure.
No semantic premise is accepted from the caller. -/
theorem extracted_or_outputBindingFailure
    [DecidableEq Digest]
    {scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest}
    {incomingStateDigest outgoingStateDigest : Digest}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1}
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1}
    {input :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows}
    {template : Data shape}
    {witnesses : Fin shape.runningCount -> Matrix shape}
    {certificate : FixedActive.Certificate
      (ProductionContext.full setup input)}
    (accepted : ClaimsAccepted scheme incomingStateDigest outgoingStateDigest
      machine setup input template witnesses certificate) :
    Accepted scheme incomingStateDigest outgoingStateDigest machine setup
        input template witnesses certificate ∨
      ProductionPiCcs.OutputBindingFailure
        (ProductionContext.full setup input)
        (decodedData template witnesses) certificate := by
  rcases
      PackedWitnessProduction.messageCheck_implies_check_or_outputBindingFailure
        (ProductionContext.canonical setup input) template witnesses
        certificate accepted.nifs with raw | failure
  · exact Or.inl {
      outer := accepted.outer
      nifs := raw
      incomingState := accepted.incomingState
      outgoingState := accepted.outgoingState
    }
  · exact Or.inr failure

end ClaimsAccepted

/-- At the base boundary there is no old packed output to prove. Production
acceptance therefore yields exactly the ordinary raw-source NC phase; the
current output is still emitted for a successor or terminal closure. -/
theorem acceptedBase_implies_ordinaryNc
    [DecidableEq Digest]
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (incomingStateDigest outgoingStateDigest : Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (template : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape)
    (certificate : FixedActive.Certificate
      (ProductionContext.full setup input))
    (accepted : Accepted scheme incomingStateDigest outgoingStateDigest
      machine setup input template witnesses certificate)
    (base : input.pending = none) :
    FixedPhase.Accepted ConcreteCarrier.extensionOps.toOps
      (InitialSum.sumcheckPolynomial
        (ProductionContext.full setup input).covers
        (decodedData template witnesses)
        (ProductionContext.full setup input).ncCoins)
      InitialSum.claimedInitial
      (ProductionPiCcs.ncPoint (ProductionContext.full setup input)
        certificate).coordinates
      certificate.piCcs.nc.toSumCheck := by
  have productionAccepted : ProductionNifs.Accepted
      (ProductionContext.full setup input)
      (decodedData template witnesses) certificate :=
    (PackedWitnessProduction.check_eq_true_iff_accepted
      (ProductionContext.canonical setup input) template witnesses
      certificate).mp accepted.nifs
  apply (ProductionBoundary.baseNcAccepted_iff
    (ProductionContext.full setup input) (decodedData template witnesses)
    certificate (by
      simpa only [ProductionContext.full_pending] using base)).mp
  exact productionAccepted.piCcs.nc

/-- Claims-level base specialization under a packed equation derived from the
successor or terminal anchor. The base has no incoming delayed value; its own
output is nevertheless retained and closed by the same backward trace. -/
theorem claimsAcceptedBase_of_packed_implies_ordinaryNc
    [DecidableEq Digest]
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (incomingStateDigest outgoingStateDigest : Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (template : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape)
    (certificate : FixedActive.Certificate
      (ProductionContext.full setup input))
    (accepted : ClaimsAccepted scheme incomingStateDigest outgoingStateDigest
      machine setup input template witnesses certificate)
    (packed : Terminal.PackedYZcolBoundAtBlock
      (ProductionContext.full setup input).covers
      (decodedData template witnesses)
      (ProductionPiCcs.ncPoint (ProductionContext.full setup input)
        certificate).block certificate.piCcs.output)
    (base : input.pending = none) :
    FixedPhase.Accepted ConcreteCarrier.extensionOps.toOps
      (InitialSum.sumcheckPolynomial
        (ProductionContext.full setup input).covers
        (decodedData template witnesses)
        (ProductionContext.full setup input).ncCoins)
      InitialSum.claimedInitial
      (ProductionPiCcs.ncPoint (ProductionContext.full setup input)
        certificate).coordinates
      certificate.piCcs.nc.toSumCheck := by
  have messageAccepted : ProductionNifs.MessageAccepted
      (ProductionContext.full setup input) certificate :=
    (PackedWitnessProduction.messageCheck_eq_true_iff_accepted
      (ProductionContext.canonical setup input) certificate).mp accepted.nifs
  have rawAccepted : ProductionNifs.Accepted
      (ProductionContext.full setup input)
      (decodedData template witnesses) certificate :=
    ProductionNifs.accepted_of_messageAccepted_and_packed
      (ProductionContext.full setup input) (decodedData template witnesses)
      certificate messageAccepted packed
  apply (ProductionBoundary.baseNcAccepted_iff
    (ProductionContext.full setup input) (decodedData template witnesses)
    certificate (by
      simpa only [ProductionContext.full_pending] using base)).mp
  exact rawAccepted.piCcs.nc

/-- Two adjacent accepted production steps prove the previous independent
active F-prime transition. The same `sharedStateDigest` is checked as the
previous outgoing and successor incoming coordinate, so continuity is
derived by exact recomputation. The conclusion intentionally proves the
previous output: the current output remains pending for its successor or the
terminal boundary. -/
theorem acceptedPair_implies_previousActiveHolds_or_namedFailure
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (previousIncomingDigest sharedStateDigest nextOutgoingDigest : Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (functionIndex : Fin 1)
    (previousInput :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (previousTemplate : Data shape)
    (previousWitnesses : Fin shape.runningCount -> Matrix shape)
    (previousCertificate : FixedActive.Certificate
      (ProductionContext.full setup previousInput))
    (previousAccepted : Accepted scheme previousIncomingDigest
      sharedStateDigest machine setup previousInput previousTemplate
      previousWitnesses previousCertificate)
    (nextInput :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (nextTemplate : Data shape)
    (nextWitnesses : Fin shape.runningCount -> Matrix shape)
    (nextCertificate : FixedActive.Certificate
      (ProductionContext.full setup nextInput))
    (nextAccepted : Accepted scheme sharedStateDigest nextOutgoingDigest
      machine setup nextInput nextTemplate nextWitnesses nextCertificate) :
    ActiveSemantics.Holds setup machine functionIndex
        (previousInput.fixedOne.toActive setup)
        (outputOf machine setup previousInput previousCertificate) \/
      ProductionPiCcs.YRingUnbound
        (ProductionContext.full setup previousInput)
        (decodedData previousTemplate previousWitnesses)
        previousCertificate \/
      ProductionBoundary.RecursiveBadEvent scheme
        (ProductionContext.full setup previousInput)
        (decodedData previousTemplate previousWitnesses)
        previousCertificate (ProductionContext.full setup nextInput)
        (decodedData nextTemplate nextWitnesses) nextCertificate \/
      RefinementBoundary.RecursiveRefinementFailure
        (ProductionContext.canonical setup previousInput) previousTemplate
        previousWitnesses previousCertificate
        (ProductionContext.canonical setup nextInput) nextTemplate
        nextWitnesses := by
  rcases
      RefinementBoundary.checkedPair_implies_previousSemanticFold_or_namedFailure
        noZeroDivisors scheme sharedStateDigest
        (ProductionContext.canonical setup previousInput) previousTemplate
        previousWitnesses previousCertificate previousAccepted.nifs
        previousAccepted.outgoingState
        (ProductionContext.canonical setup nextInput) nextTemplate
        nextWitnesses nextCertificate nextAccepted.nifs
        nextAccepted.incomingState with
    semantic | yRing | bad | refinementFailure
  · have erased :=
      (PendingErasure.semanticFoldHolds_iff_withoutPending setup previousInput
        (decodedData previousTemplate previousWitnesses)
        (derive (ProductionContext.full setup previousInput)
          previousCertificate).piRlcOutput
        (outputChildren (ProductionContext.full setup previousInput)
          previousCertificate)).mp semantic
    have transition : FixedActive.ResultTransition
        (FixedOneCanonical.nifsContext setup
          previousInput.fixedOne).materialize
        (resultOf setup previousInput previousCertificate) := by
      exact ⟨decodedData previousTemplate previousWitnesses, by
        simpa [resultOf] using erased⟩
    have fixedOneHolds :
        ActiveSemantics.FixedOneCanonical.Holds setup machine
          (previousInput.fixedOne.toSemantic setup)
          (outputOf machine setup previousInput previousCertificate) := by
      refine ⟨resultOf setup previousInput previousCertificate, ?_, rfl⟩
      have outer := previousAccepted.outerChecks
      exact {
        iterationPositive := outer.iterationPositive
        priorPublicInput := outer.priorPublicInput
        selectedNifs := by
          simpa [FixedOneCanonical.nifsContext_materialize] using transition
      }
    exact Or.inl
      ((ActiveSemantics.FixedOneCanonical.holds_iff_active setup machine
        functionIndex (previousInput.fixedOne.toSemantic setup)
        (outputOf machine setup previousInput previousCertificate)).mp
          fixedOneHolds)
  · exact Or.inr (Or.inl yRing)
  · exact Or.inr (Or.inr (Or.inl bad))
  · exact Or.inr (Or.inr (Or.inr refinementFailure))

/-- Paper-facing form of the adjacent active theorem. No additional premise
is introduced while projecting the successful branch to HyperNova
Construction 2. -/
theorem acceptedPair_implies_previousConstruction2_or_namedFailure
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (previousIncomingDigest sharedStateDigest nextOutgoingDigest : Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (functionIndex : Fin 1)
    (previousInput :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (previousTemplate : Data shape)
    (previousWitnesses : Fin shape.runningCount -> Matrix shape)
    (previousCertificate : FixedActive.Certificate
      (ProductionContext.full setup previousInput))
    (previousAccepted : Accepted scheme previousIncomingDigest
      sharedStateDigest machine setup previousInput previousTemplate
      previousWitnesses previousCertificate)
    (nextInput :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (nextTemplate : Data shape)
    (nextWitnesses : Fin shape.runningCount -> Matrix shape)
    (nextCertificate : FixedActive.Certificate
      (ProductionContext.full setup nextInput))
    (nextAccepted : Accepted scheme sharedStateDigest nextOutgoingDigest
      machine setup nextInput nextTemplate nextWitnesses nextCertificate) :
    Nightstream.Protocol.FPrime.Paper.Construction2.RecursiveHolds
        (SelectedNifsSemantics.family
          (ActiveSemantics.Construction2.selectedNifsSetup setup))
        machine functionIndex
        (previousInput.fixedOne.toActive setup).toPaper
        (outputOf machine setup previousInput previousCertificate).toPaper \/
      ProductionPiCcs.YRingUnbound
        (ProductionContext.full setup previousInput)
        (decodedData previousTemplate previousWitnesses)
        previousCertificate \/
      ProductionBoundary.RecursiveBadEvent scheme
        (ProductionContext.full setup previousInput)
        (decodedData previousTemplate previousWitnesses)
        previousCertificate (ProductionContext.full setup nextInput)
        (decodedData nextTemplate nextWitnesses) nextCertificate \/
      RefinementBoundary.RecursiveRefinementFailure
        (ProductionContext.canonical setup previousInput) previousTemplate
        previousWitnesses previousCertificate
        (ProductionContext.canonical setup nextInput) nextTemplate
        nextWitnesses := by
  rcases acceptedPair_implies_previousActiveHolds_or_namedFailure
      noZeroDivisors scheme previousIncomingDigest sharedStateDigest
      nextOutgoingDigest machine setup functionIndex previousInput
      previousTemplate previousWitnesses previousCertificate previousAccepted
      nextInput nextTemplate nextWitnesses nextCertificate nextAccepted with
    active | yRing | bad | refinementFailure
  · exact Or.inl (ActiveSemantics.Construction2.sound_selectedNifs active)
  · exact Or.inr (Or.inl yRing)
  · exact Or.inr (Or.inr (Or.inl bad))
  · exact Or.inr (Or.inr (Or.inr refinementFailure))

/-- Claims-level adjacent composition. Both public verifier executions are
first refined through the exact weak-output extraction seam. The only added
outcomes are the specifically indexed previous or successor packed-output
binding failures; no generic `outputUnbound` branch is reintroduced. -/
theorem claimsAcceptedPair_implies_previousConstruction2_or_namedFailure
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (previousIncomingDigest sharedStateDigest nextOutgoingDigest : Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (functionIndex : Fin 1)
    (previousInput :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (previousTemplate : Data shape)
    (previousWitnesses : Fin shape.runningCount -> Matrix shape)
    (previousCertificate : FixedActive.Certificate
      (ProductionContext.full setup previousInput))
    (previousAccepted : ClaimsAccepted scheme previousIncomingDigest
      sharedStateDigest machine setup previousInput previousTemplate
      previousWitnesses previousCertificate)
    (nextInput :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (nextTemplate : Data shape)
    (nextWitnesses : Fin shape.runningCount -> Matrix shape)
    (nextCertificate : FixedActive.Certificate
      (ProductionContext.full setup nextInput))
    (nextAccepted : ClaimsAccepted scheme sharedStateDigest nextOutgoingDigest
      machine setup nextInput nextTemplate nextWitnesses nextCertificate) :
    Nightstream.Protocol.FPrime.Paper.Construction2.RecursiveHolds
        (SelectedNifsSemantics.family
          (ActiveSemantics.Construction2.selectedNifsSetup setup))
        machine functionIndex
        (previousInput.fixedOne.toActive setup).toPaper
        (outputOf machine setup previousInput previousCertificate).toPaper ∨
      ProductionPiCcs.YRingUnbound
        (ProductionContext.full setup previousInput)
        (decodedData previousTemplate previousWitnesses)
        previousCertificate ∨
      ProductionBoundary.RecursiveBadEvent scheme
        (ProductionContext.full setup previousInput)
        (decodedData previousTemplate previousWitnesses)
        previousCertificate (ProductionContext.full setup nextInput)
        (decodedData nextTemplate nextWitnesses) nextCertificate ∨
      RefinementBoundary.RecursiveRefinementFailure
        (ProductionContext.canonical setup previousInput) previousTemplate
        previousWitnesses previousCertificate
        (ProductionContext.canonical setup nextInput) nextTemplate
        nextWitnesses ∨
      ProductionPiCcs.OutputBindingFailure
        (ProductionContext.full setup previousInput)
        (decodedData previousTemplate previousWitnesses)
        previousCertificate ∨
      ProductionPiCcs.OutputBindingFailure
        (ProductionContext.full setup nextInput)
        (decodedData nextTemplate nextWitnesses) nextCertificate := by
  rcases ClaimsAccepted.extracted_or_outputBindingFailure previousAccepted with
    previousExtracted | previousBindingFailure
  · rcases ClaimsAccepted.extracted_or_outputBindingFailure nextAccepted with
      nextExtracted | nextBindingFailure
    · rcases acceptedPair_implies_previousConstruction2_or_namedFailure
          noZeroDivisors scheme previousIncomingDigest sharedStateDigest
          nextOutgoingDigest machine setup functionIndex previousInput
          previousTemplate previousWitnesses previousCertificate
          previousExtracted nextInput nextTemplate nextWitnesses
          nextCertificate nextExtracted with
        paper | yRing | bad | refinementFailure
      · exact Or.inl paper
      · exact Or.inr (Or.inl yRing)
      · exact Or.inr (Or.inr (Or.inl bad))
      · exact Or.inr (Or.inr (Or.inr (Or.inl refinementFailure)))
    · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr
        nextBindingFailure))))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl
      previousBindingFailure))))

/-- Promote an independently established terminal semantic fold through the
unchanged fixed-one outer equations.  Both raw-accepted and claims-accepted
terminal paths use this same success branch. -/
private theorem activeHolds_of_semanticFold_and_outer
    [DecidableEq Digest]
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (functionIndex : Fin 1)
    (input :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (template : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape)
    (certificate : FixedActive.Certificate
      (ProductionContext.full setup input))
    (outer : FixedOneCanonical.outerCheck machine setup input.fixedOne = true)
    (semantic : SemanticFold.Holds (ProductionContext.full setup input)
      (decodedData template witnesses)
      (derive (ProductionContext.full setup input) certificate).piRlcOutput
      (outputChildren (ProductionContext.full setup input) certificate)) :
    ActiveSemantics.Holds setup machine functionIndex
      (input.fixedOne.toActive setup)
      (outputOf machine setup input certificate) := by
  have erased :=
    (PendingErasure.semanticFoldHolds_iff_withoutPending setup input
      (decodedData template witnesses)
      (derive (ProductionContext.full setup input) certificate).piRlcOutput
      (outputChildren (ProductionContext.full setup input) certificate)).mp
      semantic
  have transition : FixedActive.ResultTransition
      (FixedOneCanonical.nifsContext setup input.fixedOne).materialize
      (resultOf setup input certificate) := by
    exact ⟨decodedData template witnesses, by
      simpa [resultOf] using erased⟩
  have fixedOneHolds :
      ActiveSemantics.FixedOneCanonical.Holds setup machine
        (input.fixedOne.toSemantic setup)
        (outputOf machine setup input certificate) := by
    refine ⟨resultOf setup input certificate, ?_, rfl⟩
    have outerChecks :=
      (FixedOneCanonical.outerCheck_eq_true_iff machine setup input.fixedOne).mp
        outer
    exact {
      iterationPositive := outerChecks.iterationPositive
      priorPublicInput := outerChecks.priorPublicInput
      selectedNifs := by
        simpa [FixedOneCanonical.nifsContext_materialize] using transition
    }
  exact
    (ActiveSemantics.FixedOneCanonical.holds_iff_active setup machine
      functionIndex (input.fixedOne.toSemantic setup)
      (outputOf machine setup input certificate)).mp fixedOneHolds

/-- Active claims-level y-zcol edge over the exact full packed witnesses.
The outer fixed-one equations are irrelevant to this projection theorem but
remain part of each `ClaimsAccepted` object.  Successor raw extraction is
authorized by `nextPacked`; the resulting predecessor packed equation is
derived without previous `ChildOpenings`, semantic-input premises, or a
generic output-binding branch. -/
theorem claimsAcceptedPair_of_nextPacked_implies_previousPacked_or_rawParentBadEvent
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (previousIncomingDigest sharedStateDigest nextOutgoingDigest : Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (previousInput :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (previousTemplate : Data shape)
    (previousWitnesses : Fin shape.runningCount -> Matrix shape)
    (previousCertificate : FixedActive.Certificate
      (ProductionContext.full setup previousInput))
    (previousAccepted : ClaimsAccepted scheme previousIncomingDigest
      sharedStateDigest machine setup previousInput previousTemplate
      previousWitnesses previousCertificate)
    (nextInput :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (nextTemplate : Data shape)
    (nextWitnesses : Fin shape.runningCount -> Matrix shape)
    (nextCertificate : FixedActive.Certificate
      (ProductionContext.full setup nextInput))
    (nextAccepted : ClaimsAccepted scheme sharedStateDigest nextOutgoingDigest
      machine setup nextInput nextTemplate nextWitnesses nextCertificate)
    (nextPacked : Terminal.PackedYZcolBoundAtBlock
      (ProductionContext.full setup nextInput).covers
      (decodedData nextTemplate nextWitnesses)
      (ProductionPiCcs.ncPoint (ProductionContext.full setup nextInput)
        nextCertificate).block nextCertificate.piCcs.output) :
    Terminal.PackedYZcolBoundAtBlock
        (ProductionContext.full setup previousInput).covers
        (decodedData previousTemplate previousWitnesses)
        (ProductionPiCcs.ncPoint (ProductionContext.full setup previousInput)
          previousCertificate).block previousCertificate.piCcs.output ∨
      ProductionBoundary.RawParentRecursiveBadEvent scheme
        (ProductionContext.full setup previousInput)
        (decodedData previousTemplate previousWitnesses) previousCertificate
        (ProductionContext.full setup nextInput)
        (decodedData nextTemplate nextWitnesses) nextCertificate := by
  rcases
      PackedWitnessProduction.messageCheckedPair_of_nextPacked_of_stateChecks_implies_previousPacked_or_rawParentBadEvent
        noZeroDivisors scheme sharedStateDigest
        (ProductionContext.canonical setup previousInput) previousTemplate
        previousWitnesses previousCertificate previousAccepted.nifs
        previousAccepted.outgoingState
        (ProductionContext.canonical setup nextInput) nextTemplate
        nextWitnesses nextCertificate nextAccepted.nifs nextPacked
        nextAccepted.incomingState with packed | bad
  · exact Or.inl (by
      simpa [ProductionPiCcs.ncPoint] using packed)
  · exact Or.inr bad

/-- Exact active-level binding failures outside the delayed algebra. The
packed-witness branch owns canonical-parent and raw-commitment extraction;
the key branch keeps commitment-key coordinate continuity explicit. -/
def ParentOpeningActiveBindingFailure
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (previousInput nextInput :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (previousTemplate nextTemplate : Data shape)
    (previousWitnesses nextWitnesses :
      Fin shape.runningCount -> Matrix shape)
    (previousCertificate : FixedActive.Certificate
      (ProductionContext.full setup previousInput)) : Prop :=
  PackedWitnessProduction.ParentOpeningExternalBindingFailure
      (ProductionContext.canonical setup previousInput) previousTemplate
      previousWitnesses previousCertificate
      (ProductionContext.canonical setup nextInput) nextTemplate
      nextWitnesses ∨
    (ProductionContext.full setup nextInput).key ≠
      (ProductionContext.full setup previousInput).key

/-- Active direct-parent y-zcol edge over the exact full packed witnesses.
The predecessor canonical parent commitment/norm and successor decoded raw
commitment alignment are not assumptions: failure of either is returned as
the explicit physical binding branch. On their positive sides, Π_DEC and the accepted successor
combined-NC check derive the predecessor packed equation or only named
algebraic, commitment, and accumulator events. -/
theorem claimsAcceptedPair_of_nextPacked_implies_previousPacked_or_parentOpeningBadEvent
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (previousIncomingDigest sharedStateDigest nextOutgoingDigest : Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (previousInput :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (previousTemplate : Data shape)
    (previousWitnesses : Fin shape.runningCount -> Matrix shape)
    (previousCertificate : FixedActive.Certificate
      (ProductionContext.full setup previousInput))
    (previousAccepted : ClaimsAccepted scheme previousIncomingDigest
      sharedStateDigest machine setup previousInput previousTemplate
      previousWitnesses previousCertificate)
    (nextInput :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (nextTemplate : Data shape)
    (nextWitnesses : Fin shape.runningCount -> Matrix shape)
    (nextCertificate : FixedActive.Certificate
      (ProductionContext.full setup nextInput))
    (nextAccepted : ClaimsAccepted scheme sharedStateDigest nextOutgoingDigest
      machine setup nextInput nextTemplate nextWitnesses nextCertificate)
    (nextPacked : Terminal.PackedYZcolBoundAtBlock
      (ProductionContext.full setup nextInput).covers
      (decodedData nextTemplate nextWitnesses)
      (ProductionPiCcs.ncPoint (ProductionContext.full setup nextInput)
        nextCertificate).block nextCertificate.piCcs.output) :
    Terminal.PackedYZcolBoundAtBlock
        (ProductionContext.full setup previousInput).covers
        (decodedData previousTemplate previousWitnesses)
        (ProductionPiCcs.ncPoint (ProductionContext.full setup previousInput)
          previousCertificate).block previousCertificate.piCcs.output ∨
      ProductionBoundary.ParentOpeningRecursiveBadEvent scheme
        (ProductionContext.full setup previousInput)
        (decodedData previousTemplate previousWitnesses) previousCertificate
        (ProductionContext.full setup nextInput)
        (decodedData nextTemplate nextWitnesses) nextCertificate ∨
      ParentOpeningActiveBindingFailure setup previousInput nextInput
        previousTemplate nextTemplate previousWitnesses nextWitnesses
        previousCertificate := by
  by_cases sameKey : (ProductionContext.full setup nextInput).key =
      (ProductionContext.full setup previousInput).key
  · rcases
        PackedWitnessProduction.messageCheckedPair_of_nextPacked_of_stateChecks_implies_previousPacked_or_parentOpeningBadEvent
          noZeroDivisors scheme sharedStateDigest
          (ProductionContext.canonical setup previousInput) previousTemplate
          previousWitnesses previousCertificate previousAccepted.nifs
          previousAccepted.outgoingState
          (ProductionContext.canonical setup nextInput) nextTemplate
          nextWitnesses nextCertificate nextAccepted.nifs nextPacked
          nextAccepted.incomingState sameKey with packed | bad | binding
    · exact Or.inl (by
        simpa [ProductionPiCcs.ncPoint] using packed)
    · exact Or.inr (Or.inl bad)
    · exact Or.inr (Or.inr (Or.inl binding))
  · exact Or.inr (Or.inr (Or.inr sameKey))

/-- Strong claims-level adjacent composition under the positive successor
packed binding. The successful branch retains the newly derived predecessor
packed equation beside its Construction-2 result, which is exactly the value
needed to continue finite backward induction. -/
theorem claimsAcceptedPair_of_nextPacked_implies_previousPackedAndConstruction2_or_namedFailure
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (previousIncomingDigest sharedStateDigest nextOutgoingDigest : Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (functionIndex : Fin 1)
    (previousInput :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (previousTemplate : Data shape)
    (previousWitnesses : Fin shape.runningCount -> Matrix shape)
    (previousCertificate : FixedActive.Certificate
      (ProductionContext.full setup previousInput))
    (previousAccepted : ClaimsAccepted scheme previousIncomingDigest
      sharedStateDigest machine setup previousInput previousTemplate
      previousWitnesses previousCertificate)
    (nextInput :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (nextTemplate : Data shape)
    (nextWitnesses : Fin shape.runningCount -> Matrix shape)
    (nextCertificate : FixedActive.Certificate
      (ProductionContext.full setup nextInput))
    (nextAccepted : ClaimsAccepted scheme sharedStateDigest nextOutgoingDigest
      machine setup nextInput nextTemplate nextWitnesses nextCertificate)
    (nextPacked : Terminal.PackedYZcolBoundAtBlock
      (ProductionContext.full setup nextInput).covers
      (decodedData nextTemplate nextWitnesses)
      (ProductionPiCcs.ncPoint (ProductionContext.full setup nextInput)
        nextCertificate).block nextCertificate.piCcs.output) :
    (Terminal.PackedYZcolBoundAtBlock
          (ProductionContext.full setup previousInput).covers
          (decodedData previousTemplate previousWitnesses)
          (ProductionPiCcs.ncPoint (ProductionContext.full setup previousInput)
            previousCertificate).block previousCertificate.piCcs.output ∧
        Nightstream.Protocol.FPrime.Paper.Construction2.RecursiveHolds
          (SelectedNifsSemantics.family
            (ActiveSemantics.Construction2.selectedNifsSetup setup))
          machine functionIndex
          (previousInput.fixedOne.toActive setup).toPaper
          (outputOf machine setup previousInput previousCertificate).toPaper) ∨
      ProductionPiCcs.YRingUnbound
        (ProductionContext.full setup previousInput)
        (decodedData previousTemplate previousWitnesses)
        previousCertificate ∨
      ProductionBoundary.RecursiveBadEvent scheme
        (ProductionContext.full setup previousInput)
        (decodedData previousTemplate previousWitnesses)
        previousCertificate (ProductionContext.full setup nextInput)
        (decodedData nextTemplate nextWitnesses) nextCertificate ∨
      RefinementBoundary.RecursiveRefinementFailure
        (ProductionContext.canonical setup previousInput) previousTemplate
        previousWitnesses previousCertificate
        (ProductionContext.canonical setup nextInput) nextTemplate
        nextWitnesses := by
  rcases
      RefinementBoundary.messageCheckedPair_of_nextPacked_implies_previousPackedAndSemanticFold_or_namedFailure
        noZeroDivisors scheme sharedStateDigest
        (ProductionContext.canonical setup previousInput) previousTemplate
        previousWitnesses previousCertificate previousAccepted.nifs
        previousAccepted.outgoingState
        (ProductionContext.canonical setup nextInput) nextTemplate
        nextWitnesses nextCertificate nextAccepted.nifs nextPacked
        nextAccepted.incomingState with
    success | yRing | bad | refinementFailure
  · exact Or.inl ⟨by
        simpa [ProductionPiCcs.ncPoint] using success.1,
      ActiveSemantics.Construction2.sound_selectedNifs
        (activeHolds_of_semanticFold_and_outer machine setup functionIndex
          previousInput previousTemplate previousWitnesses previousCertificate
          previousAccepted.outer success.2)⟩
  · exact Or.inr (Or.inl yRing)
  · exact Or.inr (Or.inr (Or.inl bad))
  · exact Or.inr (Or.inr (Or.inr refinementFailure))

/-- Claims-level adjacent composition under the positive successor packed
binding furnished by backward terminal induction. Unlike the unanchored pair
theorem, this result contains no previous or successor output-binding failure. -/
theorem claimsAcceptedPair_of_nextPacked_implies_previousConstruction2_or_namedFailure
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (previousIncomingDigest sharedStateDigest nextOutgoingDigest : Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (functionIndex : Fin 1)
    (previousInput :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (previousTemplate : Data shape)
    (previousWitnesses : Fin shape.runningCount -> Matrix shape)
    (previousCertificate : FixedActive.Certificate
      (ProductionContext.full setup previousInput))
    (previousAccepted : ClaimsAccepted scheme previousIncomingDigest
      sharedStateDigest machine setup previousInput previousTemplate
      previousWitnesses previousCertificate)
    (nextInput :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (nextTemplate : Data shape)
    (nextWitnesses : Fin shape.runningCount -> Matrix shape)
    (nextCertificate : FixedActive.Certificate
      (ProductionContext.full setup nextInput))
    (nextAccepted : ClaimsAccepted scheme sharedStateDigest nextOutgoingDigest
      machine setup nextInput nextTemplate nextWitnesses nextCertificate)
    (nextPacked : Terminal.PackedYZcolBoundAtBlock
      (ProductionContext.full setup nextInput).covers
      (decodedData nextTemplate nextWitnesses)
      (ProductionPiCcs.ncPoint (ProductionContext.full setup nextInput)
        nextCertificate).block nextCertificate.piCcs.output) :
    Nightstream.Protocol.FPrime.Paper.Construction2.RecursiveHolds
        (SelectedNifsSemantics.family
          (ActiveSemantics.Construction2.selectedNifsSetup setup))
        machine functionIndex
        (previousInput.fixedOne.toActive setup).toPaper
        (outputOf machine setup previousInput previousCertificate).toPaper ∨
      ProductionPiCcs.YRingUnbound
        (ProductionContext.full setup previousInput)
        (decodedData previousTemplate previousWitnesses)
        previousCertificate ∨
      ProductionBoundary.RecursiveBadEvent scheme
        (ProductionContext.full setup previousInput)
        (decodedData previousTemplate previousWitnesses)
        previousCertificate (ProductionContext.full setup nextInput)
        (decodedData nextTemplate nextWitnesses) nextCertificate ∨
      RefinementBoundary.RecursiveRefinementFailure
        (ProductionContext.canonical setup previousInput) previousTemplate
        previousWitnesses previousCertificate
        (ProductionContext.canonical setup nextInput) nextTemplate
        nextWitnesses := by
  rcases
      claimsAcceptedPair_of_nextPacked_implies_previousPackedAndConstruction2_or_namedFailure
        noZeroDivisors scheme previousIncomingDigest sharedStateDigest
        nextOutgoingDigest machine setup functionIndex previousInput
        previousTemplate previousWitnesses previousCertificate previousAccepted
        nextInput nextTemplate nextWitnesses nextCertificate nextAccepted
        nextPacked with success | yRing | bad | refinementFailure
  · exact Or.inl success.2
  · exact Or.inr (Or.inl yRing)
  · exact Or.inr (Or.inr (Or.inl bad))
  · exact Or.inr (Or.inr (Or.inr refinementFailure))

/-- A terminal opening closes the final pending production output. No
successor context is invented. -/
theorem acceptedTerminal_implies_activeHolds_or_namedFailure
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (incomingStateDigest outgoingStateDigest : Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (functionIndex : Fin 1)
    (input :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (template : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape)
    (certificate : FixedActive.Certificate
      (ProductionContext.full setup input))
    (accepted : Accepted scheme incomingStateDigest outgoingStateDigest
      machine setup input template witnesses certificate)
    (terminalWitnesses : Fin productionGlobalParams.k -> Matrix shape)
    (terminal : PackedWitnessProduction.terminalCheck
      (ProductionContext.canonical setup input) certificate terminalWitnesses =
        true) :
    ActiveSemantics.Holds setup machine functionIndex
        (input.fixedOne.toActive setup)
        (outputOf machine setup input certificate) \/
      ProductionPiCcs.YRingUnbound (ProductionContext.full setup input)
        (decodedData template witnesses) certificate \/
      ProductionBoundary.TerminalBadEvent (ProductionContext.full setup input)
        (decodedData template witnesses) certificate \/
      RefinementBoundary.TerminalRefinementFailure
        (ProductionContext.canonical setup input) template witnesses
        certificate := by
  rcases
      RefinementBoundary.checkedTerminal_implies_semanticFold_or_namedFailure
        noZeroDivisors (ProductionContext.canonical setup input) template
        witnesses certificate accepted.nifs terminalWitnesses terminal with
    semantic | yRing | bad | refinementFailure
  · exact Or.inl (activeHolds_of_semanticFold_and_outer machine setup
      functionIndex input template witnesses certificate accepted.outer
      semantic)
  · exact Or.inr (Or.inl yRing)
  · exact Or.inr (Or.inr (Or.inl bad))
  · exact Or.inr (Or.inr (Or.inr refinementFailure))

/-- Paper-facing terminal closure for the last pending production output. -/
theorem acceptedTerminal_implies_construction2_or_namedFailure
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (incomingStateDigest outgoingStateDigest : Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (functionIndex : Fin 1)
    (input :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (template : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape)
    (certificate : FixedActive.Certificate
      (ProductionContext.full setup input))
    (accepted : Accepted scheme incomingStateDigest outgoingStateDigest
      machine setup input template witnesses certificate)
    (terminalWitnesses : Fin productionGlobalParams.k -> Matrix shape)
    (terminal : PackedWitnessProduction.terminalCheck
      (ProductionContext.canonical setup input) certificate terminalWitnesses =
        true) :
    Nightstream.Protocol.FPrime.Paper.Construction2.RecursiveHolds
        (SelectedNifsSemantics.family
          (ActiveSemantics.Construction2.selectedNifsSetup setup))
        machine functionIndex (input.fixedOne.toActive setup).toPaper
        (outputOf machine setup input certificate).toPaper \/
      ProductionPiCcs.YRingUnbound (ProductionContext.full setup input)
        (decodedData template witnesses) certificate \/
      ProductionBoundary.TerminalBadEvent (ProductionContext.full setup input)
        (decodedData template witnesses) certificate \/
      RefinementBoundary.TerminalRefinementFailure
        (ProductionContext.canonical setup input) template witnesses
        certificate := by
  rcases acceptedTerminal_implies_activeHolds_or_namedFailure
      noZeroDivisors scheme incomingStateDigest outgoingStateDigest machine
      setup functionIndex input template witnesses certificate accepted
      terminalWitnesses terminal with active | yRing | bad | refinementFailure
  · exact Or.inl (ActiveSemantics.Construction2.sound_selectedNifs active)
  · exact Or.inr (Or.inl yRing)
  · exact Or.inr (Or.inr (Or.inl bad))
  · exact Or.inr (Or.inr (Or.inr refinementFailure))

/-- Claims-level terminal closure. A verified complete terminal opening
anchors the last pending value before claims-to-raw extraction, so no packed
`y_zcol` output-binding failure remains in the result. -/
theorem claimsAcceptedTerminal_implies_construction2_or_namedFailure
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (incomingStateDigest outgoingStateDigest : Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (functionIndex : Fin 1)
    (input :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (template : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape)
    (certificate : FixedActive.Certificate
      (ProductionContext.full setup input))
    (accepted : ClaimsAccepted scheme incomingStateDigest outgoingStateDigest
      machine setup input template witnesses certificate)
    (terminalWitnesses : Fin productionGlobalParams.k -> Matrix shape)
    (terminal : PackedWitnessProduction.terminalCheck
      (ProductionContext.canonical setup input) certificate terminalWitnesses =
        true) :
    Nightstream.Protocol.FPrime.Paper.Construction2.RecursiveHolds
        (SelectedNifsSemantics.family
          (ActiveSemantics.Construction2.selectedNifsSetup setup))
        machine functionIndex (input.fixedOne.toActive setup).toPaper
        (outputOf machine setup input certificate).toPaper ∨
      ProductionPiCcs.YRingUnbound (ProductionContext.full setup input)
        (decodedData template witnesses) certificate ∨
      ProductionBoundary.TerminalBadEvent (ProductionContext.full setup input)
        (decodedData template witnesses) certificate ∨
      RefinementBoundary.TerminalRefinementFailure
        (ProductionContext.canonical setup input) template witnesses
        certificate := by
  rcases
      RefinementBoundary.messageCheckedTerminal_implies_semanticFold_or_namedFailure
        noZeroDivisors (ProductionContext.canonical setup input) template
        witnesses certificate accepted.nifs terminalWitnesses terminal with
    semantic | yRing | bad | refinementFailure
  · exact Or.inl (ActiveSemantics.Construction2.sound_selectedNifs
      (activeHolds_of_semanticFold_and_outer machine setup functionIndex input
        template witnesses certificate accepted.outer semantic))
  · exact Or.inr (Or.inl yRing)
  · exact Or.inr (Or.inr (Or.inl bad))
  · exact Or.inr (Or.inr (Or.inr refinementFailure))

/-- Packed-y-zcol-only terminal anchor.  The terminal Boolean check consumes
the fourteen complete raw child matrices; the accepted claims tail supplies
strict Π_DEC. The canonical parent commitment/norm is case-partitioned rather
than assumed, so success yields the final packed equation without `ChildOpenings`,
semantic-input premises, or a generic output-binding branch. -/
theorem claimsAcceptedTerminal_implies_packed_or_parentOpeningBadEvent
    [DecidableEq Digest]
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (incomingStateDigest outgoingStateDigest : Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (template : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape)
    (certificate : FixedActive.Certificate
      (ProductionContext.full setup input))
    (accepted : ClaimsAccepted scheme incomingStateDigest outgoingStateDigest
      machine setup input template witnesses certificate)
    (terminalWitnesses : Fin productionGlobalParams.k -> Matrix shape)
    (terminal : PackedWitnessProduction.terminalCheck
      (ProductionContext.canonical setup input) certificate terminalWitnesses =
        true) :
    Terminal.PackedYZcolBoundAtBlock
        (ProductionContext.full setup input).covers
        (decodedData template witnesses)
        (ProductionPiCcs.ncPoint (ProductionContext.full setup input)
          certificate).block certificate.piCcs.output ∨
      ProductionBoundary.ParentOpeningTerminalBadEvent
        (ProductionContext.full setup input)
        (decodedData template witnesses) certificate := by
  classical
  have claims : ProductionNifs.MessageAccepted
      (ProductionContext.full setup input) certificate :=
    (PackedWitnessProduction.messageCheck_eq_true_iff_accepted
      (ProductionContext.canonical setup input) certificate).mp accepted.nifs
  by_cases parentBound : DelayedRawChildren.CanonicalParentBinding
      (ProductionContext.full setup input) (decodedData template witnesses)
      certificate
  · rcases
        PackedWitnessProduction.terminalCheck_of_parentOpening_implies_packed_or_badEvent
          (ProductionContext.canonical setup input) template witnesses
          certificate parentBound claims.tail.piDec terminalWitnesses terminal with
      packed | mixing | binding
    · exact Or.inl (by
        simpa [ProductionPiCcs.ncPoint] using packed)
    · exact Or.inr (.mixing mixing)
    · exact Or.inr (.parentBinding binding)
  · exact Or.inr (.canonicalParentBinding parentBound)

/-- Strong terminal anchor. Complete child openings derive the final packed
equation before claims extraction, and the successful branch retains it beside
the final Construction-2 result for finite backward induction. -/
theorem claimsAcceptedTerminal_implies_packedAndConstruction2_or_namedFailure
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (incomingStateDigest outgoingStateDigest : Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (functionIndex : Fin 1)
    (input :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (template : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape)
    (certificate : FixedActive.Certificate
      (ProductionContext.full setup input))
    (accepted : ClaimsAccepted scheme incomingStateDigest outgoingStateDigest
      machine setup input template witnesses certificate)
    (terminalWitnesses : Fin productionGlobalParams.k -> Matrix shape)
    (terminal : PackedWitnessProduction.terminalCheck
      (ProductionContext.canonical setup input) certificate terminalWitnesses =
        true) :
    (Terminal.PackedYZcolBoundAtBlock
          (ProductionContext.full setup input).covers
          (decodedData template witnesses)
          (ProductionPiCcs.ncPoint (ProductionContext.full setup input)
            certificate).block certificate.piCcs.output ∧
        Nightstream.Protocol.FPrime.Paper.Construction2.RecursiveHolds
          (SelectedNifsSemantics.family
            (ActiveSemantics.Construction2.selectedNifsSetup setup))
          machine functionIndex (input.fixedOne.toActive setup).toPaper
          (outputOf machine setup input certificate).toPaper) ∨
      ProductionPiCcs.YRingUnbound (ProductionContext.full setup input)
        (decodedData template witnesses) certificate ∨
      ProductionBoundary.TerminalBadEvent (ProductionContext.full setup input)
        (decodedData template witnesses) certificate ∨
      RefinementBoundary.TerminalRefinementFailure
        (ProductionContext.canonical setup input) template witnesses
        certificate := by
  classical
  by_cases children : ChildOpenings (ProductionContext.full setup input)
      (decodedData template witnesses) certificate
  · have terminalAccepted : ProductionTerminal.Accepted
        (ProductionContext.full setup input) certificate
        (fun child => unpack (terminalWitnesses child)) :=
      (PackedWitnessProduction.terminalCheck_eq_true_iff_accepted
        (ProductionContext.canonical setup input) certificate
        terminalWitnesses).mp terminal
    rcases
        ProductionTerminal.accepted_implies_packedYZcolBound_or_badEvent
          (ProductionContext.full setup input) (decodedData template witnesses)
          certificate children (fun child => unpack (terminalWitnesses child))
          terminalAccepted with packed | mixing | binding
    · have packedAtNc : Terminal.PackedYZcolBoundAtBlock
          (ProductionContext.full setup input).covers
          (decodedData template witnesses)
          (ProductionPiCcs.ncPoint (ProductionContext.full setup input)
            certificate).block certificate.piCcs.output := by
        simpa [ProductionPiCcs.ncPoint] using packed
      rcases claimsAcceptedTerminal_implies_construction2_or_namedFailure
          noZeroDivisors scheme incomingStateDigest outgoingStateDigest machine
          setup functionIndex input template witnesses certificate accepted
          terminalWitnesses terminal with paper | yRing | bad | refinement
      · exact Or.inl ⟨packedAtNc, paper⟩
      · exact Or.inr (Or.inl yRing)
      · exact Or.inr (Or.inr (Or.inl bad))
      · exact Or.inr (Or.inr (Or.inr refinement))
    · exact Or.inr (Or.inr (Or.inl (.mixing mixing)))
    · exact Or.inr (Or.inr (Or.inl (.childBinding binding)))
  · exact Or.inr (Or.inr (Or.inr (.childOpening children)))

/-- Terminal-anchored two-step composition for the one-fold delay.  Complete
final child witnesses first derive the current packed binding; that positive
fact both closes the final step and serves as the induction anchor that closes
its predecessor.  No generic or packed-y_zcol output-unbound outcome occurs.

`ActiveTrace` owns the finite induction which repeats this edge back to an
explicit no-pending base; this leaf retains the two-step specialization. -/
theorem claimsAcceptedPairAndTerminal_implies_construction2_or_namedFailure
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (previousIncomingDigest sharedStateDigest nextOutgoingDigest : Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (functionIndex : Fin 1)
    (previousInput :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (previousTemplate : Data shape)
    (previousWitnesses : Fin shape.runningCount -> Matrix shape)
    (previousCertificate : FixedActive.Certificate
      (ProductionContext.full setup previousInput))
    (previousAccepted : ClaimsAccepted scheme previousIncomingDigest
      sharedStateDigest machine setup previousInput previousTemplate
      previousWitnesses previousCertificate)
    (nextInput :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (nextTemplate : Data shape)
    (nextWitnesses : Fin shape.runningCount -> Matrix shape)
    (nextCertificate : FixedActive.Certificate
      (ProductionContext.full setup nextInput))
    (nextAccepted : ClaimsAccepted scheme sharedStateDigest nextOutgoingDigest
      machine setup nextInput nextTemplate nextWitnesses nextCertificate)
    (terminalWitnesses : Fin productionGlobalParams.k -> Matrix shape)
    (terminal : PackedWitnessProduction.terminalCheck
      (ProductionContext.canonical setup nextInput) nextCertificate
      terminalWitnesses = true) :
    (Nightstream.Protocol.FPrime.Paper.Construction2.RecursiveHolds
        (SelectedNifsSemantics.family
          (ActiveSemantics.Construction2.selectedNifsSetup setup))
        machine functionIndex
        (previousInput.fixedOne.toActive setup).toPaper
        (outputOf machine setup previousInput previousCertificate).toPaper ∧
      Nightstream.Protocol.FPrime.Paper.Construction2.RecursiveHolds
        (SelectedNifsSemantics.family
          (ActiveSemantics.Construction2.selectedNifsSetup setup))
        machine functionIndex (nextInput.fixedOne.toActive setup).toPaper
        (outputOf machine setup nextInput nextCertificate).toPaper) ∨
      ProductionPiCcs.YRingUnbound
        (ProductionContext.full setup previousInput)
        (decodedData previousTemplate previousWitnesses)
        previousCertificate ∨
      ProductionPiCcs.YRingUnbound
        (ProductionContext.full setup nextInput)
        (decodedData nextTemplate nextWitnesses) nextCertificate ∨
      ProductionBoundary.RecursiveBadEvent scheme
        (ProductionContext.full setup previousInput)
        (decodedData previousTemplate previousWitnesses)
        previousCertificate (ProductionContext.full setup nextInput)
        (decodedData nextTemplate nextWitnesses) nextCertificate ∨
      ProductionBoundary.TerminalBadEvent
        (ProductionContext.full setup nextInput)
        (decodedData nextTemplate nextWitnesses) nextCertificate ∨
      RefinementBoundary.RecursiveRefinementFailure
        (ProductionContext.canonical setup previousInput) previousTemplate
        previousWitnesses previousCertificate
        (ProductionContext.canonical setup nextInput) nextTemplate
        nextWitnesses ∨
      RefinementBoundary.TerminalRefinementFailure
        (ProductionContext.canonical setup nextInput) nextTemplate
        nextWitnesses nextCertificate := by
  rcases
      claimsAcceptedTerminal_implies_packedAndConstruction2_or_namedFailure
        noZeroDivisors scheme sharedStateDigest nextOutgoingDigest machine
        setup functionIndex nextInput nextTemplate nextWitnesses nextCertificate
        nextAccepted terminalWitnesses terminal with
    nextSuccess | nextYRing | terminalBad | terminalRefinement
  · rcases
        claimsAcceptedPair_of_nextPacked_implies_previousConstruction2_or_namedFailure
          noZeroDivisors scheme previousIncomingDigest sharedStateDigest
          nextOutgoingDigest machine setup functionIndex previousInput
          previousTemplate previousWitnesses previousCertificate
          previousAccepted nextInput nextTemplate nextWitnesses nextCertificate
          nextAccepted nextSuccess.1 with
      previousPaper | previousYRing | recursiveBad | recursiveRefinement
    · exact Or.inl ⟨previousPaper, nextSuccess.2⟩
    · exact Or.inr (Or.inl previousYRing)
    · exact Or.inr (Or.inr (Or.inr (Or.inl recursiveBad)))
    · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr
        (Or.inl recursiveRefinement)))))
  · exact Or.inr (Or.inr (Or.inl nextYRing))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl terminalBad))))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr
      (Or.inr terminalRefinement)))))

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveBoundary
