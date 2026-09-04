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

@[inline] private def absorbNatBlock64 (state : NativePoseidon2.State64)
    (b0 b1 b2 b3 : Nat) : NativePoseidon2.State64 :=
  NativePoseidon2.absorbBlock64 state
    (NativePoseidon2.ofNat64 b0) (NativePoseidon2.ofNat64 b1)
    (NativePoseidon2.ofNat64 b2) (NativePoseidon2.ofNat64 b3)
    (NativePoseidon2.ofNat64_canonical _)
    (NativePoseidon2.ofNat64_canonical _)
    (NativePoseidon2.ofNat64_canonical _)
    (NativePoseidon2.ofNat64_canonical _)

private theorem absorbNatBlock64_denote (state : NativePoseidon2.State64)
    (b0 b1 b2 b3 : Nat) :
    (absorbNatBlock64 state b0 b1 b2 b3).denote =
      Poseidon2.absorbBlock state.denote
        [Poseidon2.ofNat b0, Poseidon2.ofNat b1,
          Poseidon2.ofNat b2, Poseidon2.ofNat b3] := by
  simp [absorbNatBlock64]

/-- Eight aligned native blocks for `VerifierContext.componentDomain`. The
framed authority length is retained as the next sponge word. -/
def componentInitialState64 (component wordCount : Nat) :
    NativePoseidon2.HashState64 :=
  let state := absorbNatBlock64 NativePoseidon2.State64.zero 78 105 103 104
  let state := absorbNatBlock64 state 116 115 116 114
  let state := absorbNatBlock64 state 101 97 109 47
  let state := absorbNatBlock64 state 70 80 114 105
  let state := absorbNatBlock64 state 109 101 47 99
  let state := absorbNatBlock64 state 111 110 116 101
  let state := absorbNatBlock64 state 120 116 47 118
  let state := absorbNatBlock64 state 49 95 49 component
  {
    sponge := state
    carry := NativePoseidon2.ofNat64 wordCount
    carryCanonical := NativePoseidon2.ofNat64_canonical _
  }

def componentInitialState (component wordCount : Nat) :
    StreamingIdentity.HashState where
  sponge := Poseidon2.absorbBlocksFast 8 Poseidon2.zeroState
    (VerifierContext.componentDomain component)
  carry := Poseidon2.ofNat wordCount

theorem componentInitialState64_denote (component wordCount : Nat) :
    (componentInitialState64 component wordCount).denote =
      componentInitialState component wordCount := by
  have zeroDenote : NativePoseidon2.State64.zero.denote =
      Poseidon2.zeroState := by
    decide
  simp only [componentInitialState64, NativePoseidon2.HashState64.denote,
    componentInitialState, absorbNatBlock64_denote,
    NativePoseidon2.ofNat64_denote, StreamingIdentity.HashState.mk.injEq]
  constructor
  · rw [zeroDenote]
    rfl
  · trivial

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

private theorem applicationComponentBlockCount
    (plan : ApplicationPackage.Plan) :
    ((applicationComponentInput plan).length + Poseidon2.rate - 1) /
        Poseidon2.rate =
      (StreamingIdentity.nodes
        (ApplicationPackage.Plan.format.encode plan)).length + 9 := by
  rw [applicationComponentInput, List.length_append,
    VerifierContext.componentDomain_length]
  simp only [VerifierContext.framed, List.length_cons]
  unfold ApplicationPackage.authorityWords
  rw [← StreamingIdentity.canonicalWords_eq_valuePreimage,
    StreamingIdentity.canonicalWords_length]
  norm_num [Poseidon2.rate]
  omega

private theorem applicationComponentAbsorbFirstEight
    (plan : ApplicationPackage.Plan) :
    Poseidon2.absorbBlocksFast 8 Poseidon2.zeroState
        (applicationComponentInput plan) =
      (componentInitialState 2 (applicationAuthorityWordCount plan)).sponge := by
  simp [applicationComponentInput, componentInitialState,
    VerifierContext.componentDomain, VerifierContext.framed,
    Poseidon2.absorbBlocksFast, Poseidon2.rate]

private theorem applicationComponentDropFirstEight
    (plan : ApplicationPackage.Plan) :
    (applicationComponentInput plan).drop (8 * Poseidon2.rate) =
      (componentInitialState 2 (applicationAuthorityWordCount plan)).carry ::
        StreamingIdentity.canonicalWords
          (ApplicationPackage.Plan.format.encode plan) := by
  rw [applicationComponentInput, applicationAuthorityWordCount_eq,
    ApplicationPackage.authorityWords,
    ← StreamingIdentity.canonicalWords_eq_valuePreimage]
  simp [componentInitialState, VerifierContext.componentDomain,
    VerifierContext.framed, Poseidon2.rate]

private theorem applicationComponentStreamedAbsorbed
    (plan : ApplicationPackage.Plan) :
    Poseidon2.absorbBlocksFast
        (((applicationComponentInput plan).length + Poseidon2.rate - 1) /
          Poseidon2.rate)
        Poseidon2.zeroState (applicationComponentInput plan) =
      Poseidon2.absorbBlock (applicationComponentState plan).sponge
        [(applicationComponentState plan).carry] := by
  rw [applicationComponentBlockCount]
  let value := ApplicationPackage.Plan.format.encode plan
  let initial := componentInitialState 2 (applicationAuthorityWordCount plan)
  calc
    Poseidon2.absorbBlocksFast
        ((StreamingIdentity.nodes value).length + 9)
        Poseidon2.zeroState (applicationComponentInput plan) =
      Poseidon2.absorbBlocksFast
        (8 + ((StreamingIdentity.nodes value).length + 1))
        Poseidon2.zeroState (applicationComponentInput plan) := by
          congr 1
          omega
    _ = Poseidon2.absorbBlocksFast
        ((StreamingIdentity.nodes value).length + 1)
        (Poseidon2.absorbBlocksFast 8 Poseidon2.zeroState
          (applicationComponentInput plan))
        ((applicationComponentInput plan).drop (8 * Poseidon2.rate)) := by
          rw [StreamingIdentity.absorbBlocksFast_add]
    _ = Poseidon2.absorbBlocksFast
        ((StreamingIdentity.nodes value).length + 1) initial.sponge
        (initial.carry :: StreamingIdentity.canonicalWords value) := by
          rw [applicationComponentAbsorbFirstEight,
            applicationComponentDropFirstEight]
    _ = Poseidon2.absorbBlock
        (StreamingIdentity.processNodes initial
          (StreamingIdentity.nodes value)).sponge
        [(StreamingIdentity.processNodes initial
          (StreamingIdentity.nodes value)).carry] := by
          exact StreamingIdentity.processNodes_finalAbsorb initial
            (StreamingIdentity.nodes value)
    _ = Poseidon2.absorbBlock
        (StreamingIdentity.processValue value initial).sponge
        [(StreamingIdentity.processValue value initial).carry] := by
          rw [StreamingIdentity.processValue_eq_processNodes]
    _ = Poseidon2.absorbBlock (applicationComponentState plan).sponge
        [(applicationComponentState plan).carry] := by
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
