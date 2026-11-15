import NightstreamFPrime.Export.BoundedDepth.IdentityBinding
import NightstreamFPrime.Export.DomainSeparation
import NightstreamFPrime.Lifecycle.BoundedDepth.ApplicationIdentity
import NightstreamFPrime.Lifecycle.BoundedDepth.CircuitKeyDigest
import NightstreamFPrime.Lifecycle.BoundedDepth.State
import NightstreamFPrime.Spec.Poseidon2

/-! # Claims: encodings

Each hashed encoding fixes its input, or gives two distinct sponge inputs with one digest.

- Spec: `SPEC.md` section 4 and `SECURITY_MODEL.md` section Hashed encodings, which lists every
  absorb and its Rust parity test.
- Assumptions: none. Each claim is deterministic. A bound on the collision event needs
  `[P2-PERM]` and `[SPONGE]`, and no claim of this file gives it.
- Open: the inputs that an application hashes can start with any words, so
  `domains_separated` does not cover them. -/

namespace NightstreamFPrime.Claims.Encodings

open Spec Lifecycle Lifecycle.BoundedDepth Export.BoundedDepth

/-- Two inputs with one block list are equal, so `hash` absorbs different blocks for different
inputs. -/
theorem hash_padding_injective
    {left : List F}
    {right : List F}
    (equal : Poseidon2.blocks left = Poseidon2.blocks right) :
    left = right :=
  @Spec.Poseidon2.blocks_injective left right equal

/-- Two different well-formed states with one state hash give a collision on their pair: the
state preimages differ, or a slot differs and its slot preimages differ. -/
theorem state_hash_binds
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
        Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    {left : State logicalWidth publicFits}
    {right : State logicalWidth publicFits}
    (leftWell : left.WellFormed)
    (rightWell : right.WellFormed)
    (different : left ≠ right)
    (same : BoundedDepth.stateHash left = BoundedDepth.stateHash right) :
    StateCollision left right :=
  @Lifecycle.BoundedDepth.stateHash_collision logicalWidth publicFits left right leftWell
    rightWell different same

/-- Two keys with one digest are equal, or their hash inputs are distinct and have one Poseidon2
hash. -/
theorem circuit_key_binds
    {identity : List F}
    {identity' : List F}
    {circuit : Fin 2}
    {circuit' : Fin 2}
    (same : CircuitKeyDigest.digest identity circuit = CircuitKeyDigest.digest identity' circuit') :
    (identity, circuit) = (identity', circuit') ∨
        CircuitKeyDigest.input identity circuit ≠ CircuitKeyDigest.input identity' circuit' ∧
          Poseidon2.hash (CircuitKeyDigest.input identity circuit) =
            Poseidon2.hash (CircuitKeyDigest.input identity' circuit') :=
  @Lifecycle.BoundedDepth.CircuitKeyDigest.digest_binds identity identity' circuit circuit' same

/-- Two pairs of packages with one identity are equal, or their hash inputs are distinct and have
one Poseidon2 hash. `Spec.Sponge.collision_blocks` turns the second case into a sponge collision. -/
theorem package_identity_binds
    {light : StepPackage.Package}
    {refresh : StepPackage.Package}
    {light' : StepPackage.Package}
    {refresh' : StepPackage.Package}
    (fits : IdentityBinding.Fits (StepPackage.identityValue light refresh))
    (fits' : IdentityBinding.Fits (StepPackage.identityValue light' refresh'))
    (same : StepPackage.identity light refresh = StepPackage.identity light' refresh') :
    (light, refresh) = (light', refresh') ∨
        StepPackage.boundedDepthDomain ++
              Export.Package.valuePreimage (StepPackage.identityValue light refresh) ≠
            StepPackage.boundedDepthDomain ++
              Export.Package.valuePreimage (StepPackage.identityValue light' refresh') ∧
          Poseidon2.hash
              (StepPackage.boundedDepthDomain ++
                Export.Package.valuePreimage (StepPackage.identityValue light refresh)) =
            Poseidon2.hash
              (StepPackage.boundedDepthDomain ++
                Export.Package.valuePreimage (StepPackage.identityValue light' refresh')) :=
  @Export.BoundedDepth.IdentityBinding.identity_binds light refresh light' refresh' fits fits'
    same

/-- Two canonical applications with one identity have the same records, or their padded hash
inputs are two distinct block lists with one sponge digest. -/
theorem application_identity_binds
    {left : ApplicationIdentity.Application}
    {right : ApplicationIdentity.Application}
    (leftCanonical : left.Canonical)
    (rightCanonical : right.Canonical)
    (same : left.identity = right.identity) :
    left = right ∨
        Sponge.padding.pad left.input ≠ Sponge.padding.pad right.input ∧
          Sponge.digest (Sponge.padding.pad left.input) =
            Sponge.digest (Sponge.padding.pad right.input) :=
  @Lifecycle.BoundedDepth.ApplicationIdentity.Application.identity_binds left right leftCanonical
    rightCanonical same

/-- Each two distinct rows of the table `Export.DomainSeparation.uses` start with tags that
differ, so two uses of the table never absorb one block list. -/
theorem domains_separated :
    List.Pairwise Export.DomainSeparation.Separated Export.DomainSeparation.uses :=
  @Export.DomainSeparation.separated

end NightstreamFPrime.Claims.Encodings
