import NightstreamFPrime.Export.Stage1.PerApplicationStreamingIdentity

/-!
Owns the allocation-bounded application component of the verifier context.
It streams the canonical application `Plan` codec twice: once for the framed
word count and once through the proved native Poseidon2 sponge.
-/

namespace NightstreamFPrime.Export.Stage1.PerApplicationVerifierContextStreaming

open NightstreamFPrime.Export
open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec

def applicationPlanNodeCount (plan : ApplicationPackage.Plan) : Nat :=
  PerApplicationStreamingIdentity.processApplicationPlanWith
    StreamingIdentity.countNode 0 plan

theorem applicationPlanNodeCount_eq_nodes_length
    (plan : ApplicationPackage.Plan) :
    applicationPlanNodeCount plan =
      (StreamingIdentity.nodes
        (ApplicationPackage.Plan.format.encode plan)).length := by
  unfold applicationPlanNodeCount
  rw [PerApplicationStreamingIdentity.processApplicationPlanWith_eq_processValueWith]
  simpa using StreamingIdentity.processValueWith_countNode
    (ApplicationPackage.Plan.format.encode plan) 0

def applicationAuthorityWordCount (plan : ApplicationPackage.Plan) : Nat :=
  applicationPlanNodeCount plan * 4

theorem applicationAuthorityWordCount_eq (plan : ApplicationPackage.Plan) :
    applicationAuthorityWordCount plan =
      (ApplicationPackage.authorityWords plan).length := by
  calc
    applicationAuthorityWordCount plan =
        (StreamingIdentity.nodes
          (ApplicationPackage.Plan.format.encode plan)).length * 4 := by
      rw [applicationAuthorityWordCount,
        applicationPlanNodeCount_eq_nodes_length]
    _ = (StreamingIdentity.canonicalWords
          (ApplicationPackage.Plan.format.encode plan)).length :=
      (StreamingIdentity.canonicalWords_length _).symm
    _ = (Package.valuePreimage
          (ApplicationPackage.Plan.format.encode plan)).length := by
      rw [StreamingIdentity.canonicalWords_eq_valuePreimage]
    _ = (ApplicationPackage.authorityWords plan).length := by
      rfl

/-- Two aligned native blocks for `VerifierContext.componentDomain`. The last
seven domain words, the component, and the framed authority length stay
pending. -/
def componentInitialState64 (component wordCount : Nat) :
    NativePoseidon2.HashState64 where
  sponge := NativePoseidon2.absorbWords64
    (NativePoseidon2.absorbWords64 NativePoseidon2.State64.zero
      [78, 105, 103, 104, 116, 115, 116, 114, 101, 97, 109, 47] (by decide))
    [70, 80, 114, 105, 109, 101, 47, 99, 111, 110, 116, 101] (by decide)
  pending := [120, 116, 47, 118, 49, 95, 49, NativePoseidon2.ofNat64 component,
    NativePoseidon2.ofNat64 wordCount]
  pendingCanonical := by
    simp only [List.mem_cons, List.not_mem_nil, or_false]
    rintro word (rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl)
    all_goals first | decide | exact NativePoseidon2.ofNat64_canonical _

def componentInitialState (component wordCount : Nat) :
    StreamingIdentity.HashState :=
  StreamingIdentity.prefixState 2
    (VerifierContext.componentDomain component ++ [Poseidon2.ofNat wordCount])

theorem componentInitialState64_denote (component wordCount : Nat) :
    (componentInitialState64 component wordCount).denote =
      componentInitialState component wordCount := by
  have zeroDenote : NativePoseidon2.State64.zero.denote =
      Poseidon2.zeroState := by
    decide
  have block0 : ([78, 105, 103, 104, 116, 115, 116, 114, 101, 97, 109, 47] :
      List UInt64).map UInt64.denote =
        [(78 : F), 105, 103, 104, 116, 115, 116, 114, 101, 97, 109, 47] := by
    decide
  have block1 : ([70, 80, 114, 105, 109, 101, 47, 99, 111, 110, 116, 101] :
      List UInt64).map UInt64.denote =
        [(70 : F), 80, 114, 105, 109, 101, 47, 99, 111, 110, 116, 101] := by
    decide
  simp only [componentInitialState64, NativePoseidon2.HashState64.denote,
    componentInitialState, StreamingIdentity.prefixState,
    NativePoseidon2.absorbWords64_denote, StreamingIdentity.HashState.mk.injEq]
  constructor
  · rw [zeroDenote, block0, block1]
    rfl
  · simp only [List.map_cons, List.map_nil, NativePoseidon2.ofNat64_denote]
    simp [VerifierContext.componentDomain, Poseidon2.rate, UInt64.denote]

def applicationComponentState64 (plan : ApplicationPackage.Plan) :
    NativePoseidon2.HashState64 :=
  PerApplicationStreamingIdentity.processApplicationPlanWith
    NativePoseidon2.pushNode64
    (componentInitialState64 2 (applicationAuthorityWordCount plan)) plan

def applicationComponentState (plan : ApplicationPackage.Plan) :
    StreamingIdentity.HashState :=
  PerApplicationStreamingIdentity.processApplicationPlanWith
    StreamingIdentity.pushNode
    (componentInitialState 2 (applicationAuthorityWordCount plan)) plan

theorem applicationComponentState64_denote (plan : ApplicationPackage.Plan) :
    (applicationComponentState64 plan).denote =
      applicationComponentState plan := by
  unfold applicationComponentState64 applicationComponentState
  rw [PerApplicationStreamingIdentity.processApplicationPlanWith_eq_processValueWith,
    PerApplicationStreamingIdentity.processApplicationPlanWith_eq_processValueWith]
  have simulation := StreamingIdentity.processValueWith_simulates
    NativePoseidon2.HashState64.denote NativePoseidon2.pushNode64
    StreamingIdentity.pushNode NativePoseidon2.pushNode64_denote
    (ApplicationPackage.Plan.format.encode plan)
    (componentInitialState64 2 (applicationAuthorityWordCount plan))
  rw [componentInitialState64_denote] at simulation
  exact simulation

private def applicationComponentInput (plan : ApplicationPackage.Plan) : List F :=
  VerifierContext.componentDomain 2 ++
    VerifierContext.framed (ApplicationPackage.authorityWords plan)

private theorem applicationComponentInput_eq (plan : ApplicationPackage.Plan) :
    applicationComponentInput plan =
      (VerifierContext.componentDomain 2 ++
          [Poseidon2.ofNat (applicationAuthorityWordCount plan)]) ++
        StreamingIdentity.canonicalWords
          (ApplicationPackage.Plan.format.encode plan) := by
  rw [applicationComponentInput, applicationAuthorityWordCount_eq,
    ApplicationPackage.authorityWords,
    ← StreamingIdentity.canonicalWords_eq_valuePreimage]
  simp [VerifierContext.framed]

private theorem applicationComponentStreamedAbsorbed
    (plan : ApplicationPackage.Plan) :
    Poseidon2.absorbBlocksFast
        (((applicationComponentInput plan).length + Poseidon2.rate - 1) /
          Poseidon2.rate)
        Poseidon2.zeroState (applicationComponentInput plan) =
      Poseidon2.absorbBlock (applicationComponentState plan).sponge
        (applicationComponentState plan).pending := by
  rw [applicationComponentInput_eq,
    StreamingIdentity.prefixStreamed_eq 2 _ _
      (by simp [Poseidon2.rate]) (by simp [Poseidon2.rate])
      (by simp [Poseidon2.rate])]
  unfold applicationComponentState
  rw [PerApplicationStreamingIdentity.processApplicationPlanWith_eq_processValueWith]
  rfl

/-- Allocation-bounded executable application component digest. -/
@[inline] def applicationComponentDigestDirect
    (plan : ApplicationPackage.Plan) : VerifierContext.Digest4 :=
  VerifierContext.Digest4.ofList
    (NativePoseidon2.finalize64 (applicationComponentState64 plan)).denote

/-- The native two-pass stream is the canonical application authority digest. -/
theorem applicationComponentDigestDirect_eq
    (plan : ApplicationPackage.Plan) :
    applicationComponentDigestDirect plan =
      VerifierContext.componentDigest 2
        (ApplicationPackage.authorityWords plan) := by
  unfold applicationComponentDigestDirect VerifierContext.componentDigest
  apply congrArg VerifierContext.Digest4.ofList
  rw [NativePoseidon2.finalize64_denote,
    applicationComponentState64_denote, Poseidon2.hash_eq_hashFast]
  unfold StreamingIdentity.finalize Poseidon2.hashFast
  dsimp only
  have absorbed := applicationComponentStreamedAbsorbed plan
  unfold applicationComponentInput at absorbed
  rw [absorbed]

end NightstreamFPrime.Export.Stage1.PerApplicationVerifierContextStreaming
