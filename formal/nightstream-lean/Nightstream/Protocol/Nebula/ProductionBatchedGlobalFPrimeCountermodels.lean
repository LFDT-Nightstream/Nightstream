import Nightstream.Protocol.Nebula.ProductionBatchedGlobalFPrime

/-!
Countermodel for segment-local transcript authority.

If every segment may select its own lane headers and challenge-derivation
identity, both segments can pass their local binding checks while the lifetime
changes verifier authority at the boundary. The production lifetime type now
fixes both values as parameters of the complete chain.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.Nebula.ProductionBatchedGlobalFPrimeCountermodels

/-- Minimal local authority used only by this countermodel. -/
structure SegmentAuthority where
  header : Bool
  derivationIdentity : Bool
deriving DecidableEq

def derivedChallenge (authority : SegmentAuthority) : Bool :=
  authority.header != authority.derivationIdentity

/-- A weak segment checks only its own selected authority. -/
structure WeakSegment where
  authority : SegmentAuthority
  challenge : Bool
  locallyBound : challenge = derivedChallenge authority

structure WeakTwoSegmentLifetime where
  first : WeakSegment
  second : WeakSegment

/-- Correct lifetime authority requires both segments to use the same header
and the same derivation identity. -/
def LifetimeAuthorityFixed (run : WeakTwoSegmentLifetime) : Prop :=
  run.first.authority = run.second.authority

def changedAuthorityRun : WeakTwoSegmentLifetime :=
  { first :=
      { authority := ⟨false, false⟩
        challenge := false
        locallyBound := rfl }
    second :=
      { authority := ⟨true, false⟩
        challenge := true
        locallyBound := rfl } }

/-- Both local checks pass even though authority changes between segments. -/
theorem weak_local_checks_accept_changed_authority :
    changedAuthorityRun.first.challenge =
        derivedChallenge changedAuthorityRun.first.authority /\
      changedAuthorityRun.second.challenge =
        derivedChallenge changedAuthorityRun.second.authority :=
  ⟨rfl, rfl⟩

/-- The full-lifetime fixed-authority condition rejects the same trace. -/
theorem fixed_lifetime_authority_rejects_countermodel :
    ¬ LifetimeAuthorityFixed changedAuthorityRun := by
  simp [LifetimeAuthorityFixed, changedAuthorityRun]

end Nightstream.Protocol.Nebula.ProductionBatchedGlobalFPrimeCountermodels
