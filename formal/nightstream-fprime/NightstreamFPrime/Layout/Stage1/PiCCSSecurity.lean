import NightstreamFPrime.Layout.Stage1.StateEncoding
import NightstreamFPrime.Lifecycle.TranscriptCoverage
import NightstreamFPrime.Lifecycle.VerifierContext
import NightstreamFPrime.Spec.Folding.PiCCS.TranscriptReplay

/-!
Owns the committed-statement reduction for digest-only PiCCS, and the
prior-state link that extends `Lifecycle.TranscriptCoverage` to the running
statement.

The deterministic result stops at explicit collision events. It assigns no
probability to Poseidon2 and does not mix commitment binding, Fiat--Shamir
security, sampling security, or the SuperNeo knowledge reduction into row
soundness. Those assumptions belong to the complete Stage 1 security
composition.
-/

namespace NightstreamFPrime.Layout.Stage1.PiCCSSecurity

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

/-- Two different canonical verifier-context encodings produce one context
digest. This is the exact Poseidon2 context-hash collision event. -/
def ContextDigestCollision
    (left right : VerifierContext.Descriptor) : Prop :=
  left.serialize ≠ right.serialize ∧ left.digest4 = right.digest4

/-- The four verifier-owned authority components use distinct component
domains. -/
inductive AuthorityComponent where
  | relation
  | application
  | nifsKey
  | commitmentKey
deriving DecidableEq

def AuthorityComponent.tag : AuthorityComponent → Nat
  | .relation => 1
  | .application => 2
  | .nifsKey => 3
  | .commitmentKey => 4

def AuthorityComponent.words (authority : VerifierContext.Authority) :
    AuthorityComponent → List F
  | .relation => authority.relationWords
  | .application => authority.applicationWords
  | .nifsKey => authority.nifsKeyWords
  | .commitmentKey => authority.commitmentKeyWords

/-- Two different raw authority components produce one domain-separated
component digest. -/
def AuthorityComponentDigestCollision
    (left right : VerifierContext.Authority) : Prop :=
  ∃ component : AuthorityComponent,
    component.words left ≠ component.words right ∧
      VerifierContext.componentDigest component.tag (component.words left) =
        VerifierContext.componentDigest component.tag (component.words right)

/-- Two different canonical Stage 1 statement encodings produce one state
digest. This is the exact Poseidon2 state-hash collision event. -/
def StateHashCollision
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (left right : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits)) : Prop :=
  serializePreimage (publicFits := publicFits) left ≠
      serializePreimage (publicFits := publicFits) right ∧
    stateHash (publicFits := publicFits) left =
      stateHash (publicFits := publicFits) right

/-- Equal context digests identify one canonical descriptor unless the
context hash collides. -/
theorem contextDigest_identifies_descriptor_or_collision
    (left right : VerifierContext.Descriptor)
    (digestEqual : left.digest4 = right.digest4) :
    left = right ∨ ContextDigestCollision left right := by
  by_cases same : left = right
  · exact Or.inl same
  · apply Or.inr
    refine ⟨?_, digestEqual⟩
    intro encodedEqual
    exact same (VerifierContext.Descriptor.serialize_injective encodedEqual)

/-- Equal component descriptors identify every raw verifier-owned word list
unless one named component digest collides. -/
theorem descriptor_identifies_authority_or_component_collision
    (left right : VerifierContext.Authority)
    (descriptorEqual : VerifierContext.descriptor left =
      VerifierContext.descriptor right) :
    left = right ∨ AuthorityComponentDigestCollision left right := by
  rcases left with ⟨leftRelation, leftApplication, leftNifs, leftCommitment⟩
  rcases right with
    ⟨rightRelation, rightApplication, rightNifs, rightCommitment⟩
  have relationDigestEqual :=
    congrArg VerifierContext.Descriptor.relation descriptorEqual
  have applicationDigestEqual :=
    congrArg VerifierContext.Descriptor.application descriptorEqual
  have nifsDigestEqual :=
    congrArg VerifierContext.Descriptor.nifsKey descriptorEqual
  have commitmentDigestEqual :=
    congrArg VerifierContext.Descriptor.commitmentKey descriptorEqual
  change VerifierContext.componentDigest 1 leftRelation =
    VerifierContext.componentDigest 1 rightRelation at relationDigestEqual
  change VerifierContext.componentDigest 2 leftApplication =
    VerifierContext.componentDigest 2 rightApplication at applicationDigestEqual
  change VerifierContext.componentDigest 3 leftNifs =
    VerifierContext.componentDigest 3 rightNifs at nifsDigestEqual
  change VerifierContext.componentDigest 4 leftCommitment =
    VerifierContext.componentDigest 4 rightCommitment at commitmentDigestEqual
  by_cases relationEqual : leftRelation = rightRelation
  · by_cases applicationEqual : leftApplication = rightApplication
    · by_cases nifsEqual : leftNifs = rightNifs
      · by_cases commitmentEqual : leftCommitment = rightCommitment
        · subst rightRelation
          subst rightApplication
          subst rightNifs
          subst rightCommitment
          exact Or.inl rfl
        · apply Or.inr
          refine ⟨.commitmentKey, commitmentEqual, ?_⟩
          exact commitmentDigestEqual
      · apply Or.inr
        refine ⟨.nifsKey, nifsEqual, ?_⟩
        exact nifsDigestEqual
    · apply Or.inr
      refine ⟨.application, applicationEqual, ?_⟩
      exact applicationDigestEqual
  · apply Or.inr
    refine ⟨.relation, relationEqual, ?_⟩
    exact relationDigestEqual

/-- Equal public context digests identify the complete raw authority unless
a component digest or the fixed outer context hash collides. -/
theorem contextDigest_identifies_authority_or_collision
    (left right : VerifierContext.Authority)
    (digestEqual : (VerifierContext.descriptor left).digest4 =
      (VerifierContext.descriptor right).digest4) :
    left = right ∨ AuthorityComponentDigestCollision left right ∨
      ContextDigestCollision (VerifierContext.descriptor left)
        (VerifierContext.descriptor right) := by
  rcases contextDigest_identifies_descriptor_or_collision
      (VerifierContext.descriptor left) (VerifierContext.descriptor right)
      digestEqual with descriptorSame | contextCollision
  · rcases descriptor_identifies_authority_or_component_collision
        left right descriptorSame with authoritySame | componentCollision
    · exact Or.inl authoritySame
    · exact Or.inr (Or.inl componentCollision)
  · exact Or.inr (Or.inr contextCollision)

/-- Equal state digests identify one complete well-formed statement unless
the state hash collides. The encoding-injectivity theorem covers the context,
iteration, application state, point, commitments, parent public input,
separate `Eval_K`, and every separate `Eval_A` value. Well-formedness fixes
`pc = 1` and makes the child public inputs the canonical split of the
parent. -/
theorem stateHash_identifies_statement_or_collision
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (left right : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (leftWellFormed : StateEncoding.WellFormed left)
    (rightWellFormed : StateEncoding.WellFormed right)
    (digestEqual : stateHash (publicFits := publicFits) left =
      stateHash (publicFits := publicFits) right) :
    left = right ∨ StateHashCollision left right := by
  by_cases same : left = right
  · exact Or.inl same
  · apply Or.inr
    refine ⟨?_, digestEqual⟩
    intro encodedEqual
    exact same (StateEncoding.serializePreimage_injective
      leftWellFormed rightWellFormed encodedEqual)

/-- A valid state encoding cannot be a nonempty trailing extension of another
valid state encoding. This rules out the ambiguity branch before any
cryptographic assumption is used. -/
theorem no_valid_trailing_extension
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    {left right : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits)}
    (leftWellFormed : StateEncoding.WellFormed left)
    (rightWellFormed : StateEncoding.WellFormed right)
    {suffix : List F} (suffixNonempty : suffix ≠ []) :
    serializePreimage (publicFits := publicFits) left ≠
      serializePreimage (publicFits := publicFits) right ++ suffix :=
  StateEncoding.serializePreimage_not_trailing_extension
    leftWellFormed rightWellFormed suffixNonempty

/-- Digest-only committed-statement reduction through the complete verifier
challenge view. Equal context and state digests plus equal derived PiCCS coins
identify the canonical context, full running statement, and exact absorbed
prior-state-and-round-message replay authority, unless one named collision
event occurred. -/
theorem committed_statement_challenges_identify_or_failure
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (oracle : TranscriptReplay.Oracle K
      NightstreamFPrime.Lifecycle.Transcript.State productionShape)
    (leftContext rightContext : VerifierContext.Descriptor)
    (leftState rightState : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (leftReplay rightReplay : TranscriptReplay.ReplayInput K
      NightstreamFPrime.Lifecycle.Transcript.State productionShape)
    (leftWellFormed : StateEncoding.WellFormed leftState)
    (rightWellFormed : StateEncoding.WellFormed rightState)
    (contextDigestEqual : leftContext.digest4 = rightContext.digest4)
    (stateDigestEqual : stateHash (publicFits := publicFits) leftState =
      stateHash (publicFits := publicFits) rightState)
    (alphaEqual : (leftReplay.derive oracle).alpha =
      (rightReplay.derive oracle).alpha)
    (gammaEqual : (leftReplay.derive oracle).gamma =
      (rightReplay.derive oracle).gamma)
    (roundPointEqual : (leftReplay.derive oracle).roundPoint =
      (rightReplay.derive oracle).roundPoint) :
    (leftContext = rightContext ∧ leftState = rightState ∧
      leftReplay.authority = rightReplay.authority) ∨
      ContextDigestCollision leftContext rightContext ∨
      StateHashCollision leftState rightState ∨
      TranscriptReplay.TranscriptReplayCollision oracle leftReplay rightReplay := by
  rcases contextDigest_identifies_descriptor_or_collision
      leftContext rightContext contextDigestEqual with contextSame | contextFailure
  · rcases stateHash_identifies_statement_or_collision leftState rightState
        leftWellFormed rightWellFormed stateDigestEqual with stateSame | stateFailure
    · rcases TranscriptReplay.authority_eq_or_challenge_collision oracle
          leftReplay rightReplay alphaEqual gammaEqual roundPointEqual with
        replaySame | replayFailure
      · exact Or.inl ⟨contextSame, stateSame, replaySame⟩
      · exact Or.inr (Or.inr (Or.inr replayFailure))
    · exact Or.inr (Or.inr (Or.inl stateFailure))
  · exact Or.inr (Or.inl contextFailure)

/-- The same committed-statement reduction for the causal pre-output
transcript state. This names transcript-state collision separately from the
complete challenge-view collision above. -/
theorem committed_statement_finalState_identify_or_failure
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (oracle : TranscriptReplay.Oracle K
      NightstreamFPrime.Lifecycle.Transcript.State productionShape)
    (leftContext rightContext : VerifierContext.Descriptor)
    (leftState rightState : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (leftReplay rightReplay : TranscriptReplay.ReplayInput K
      NightstreamFPrime.Lifecycle.Transcript.State productionShape)
    (leftWellFormed : StateEncoding.WellFormed leftState)
    (rightWellFormed : StateEncoding.WellFormed rightState)
    (contextDigestEqual : leftContext.digest4 = rightContext.digest4)
    (stateDigestEqual : stateHash (publicFits := publicFits) leftState =
      stateHash (publicFits := publicFits) rightState)
    (finalStateEqual : (leftReplay.derive oracle).finalState =
      (rightReplay.derive oracle).finalState) :
    (leftContext = rightContext ∧ leftState = rightState ∧
      leftReplay.authority = rightReplay.authority) ∨
      ContextDigestCollision leftContext rightContext ∨
      StateHashCollision leftState rightState ∨
      TranscriptReplay.TranscriptStateCollision oracle leftReplay rightReplay := by
  rcases contextDigest_identifies_descriptor_or_collision
      leftContext rightContext contextDigestEqual with contextSame | contextFailure
  · rcases stateHash_identifies_statement_or_collision leftState rightState
        leftWellFormed rightWellFormed stateDigestEqual with stateSame | stateFailure
    · rcases TranscriptReplay.authority_eq_or_state_collision oracle
          leftReplay rightReplay finalStateEqual with replaySame | replayFailure
      · exact Or.inl ⟨contextSame, stateSame, replaySame⟩
      · exact Or.inr (Or.inr (Or.inr replayFailure))
    · exact Or.inr (Or.inr (Or.inl stateFailure))
  · exact Or.inr (Or.inl contextFailure)

/-- Raw-authority form of the complete challenge-view reduction. The success
branch identifies all four verifier-owned word lists, not only their
descriptor. -/
theorem committed_authority_statement_challenges_identify_or_failure
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (oracle : TranscriptReplay.Oracle K
      NightstreamFPrime.Lifecycle.Transcript.State productionShape)
    (leftAuthority rightAuthority : VerifierContext.Authority)
    (leftState rightState : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (leftReplay rightReplay : TranscriptReplay.ReplayInput K
      NightstreamFPrime.Lifecycle.Transcript.State productionShape)
    (leftWellFormed : StateEncoding.WellFormed leftState)
    (rightWellFormed : StateEncoding.WellFormed rightState)
    (contextDigestEqual :
      (VerifierContext.descriptor leftAuthority).digest4 =
        (VerifierContext.descriptor rightAuthority).digest4)
    (stateDigestEqual : stateHash (publicFits := publicFits) leftState =
      stateHash (publicFits := publicFits) rightState)
    (alphaEqual : (leftReplay.derive oracle).alpha =
      (rightReplay.derive oracle).alpha)
    (gammaEqual : (leftReplay.derive oracle).gamma =
      (rightReplay.derive oracle).gamma)
    (roundPointEqual : (leftReplay.derive oracle).roundPoint =
      (rightReplay.derive oracle).roundPoint) :
    (leftAuthority = rightAuthority ∧ leftState = rightState ∧
      leftReplay.authority = rightReplay.authority) ∨
      AuthorityComponentDigestCollision leftAuthority rightAuthority ∨
      ContextDigestCollision (VerifierContext.descriptor leftAuthority)
        (VerifierContext.descriptor rightAuthority) ∨
      StateHashCollision leftState rightState ∨
      TranscriptReplay.TranscriptReplayCollision oracle leftReplay rightReplay := by
  rcases contextDigest_identifies_authority_or_collision leftAuthority
      rightAuthority contextDigestEqual with authoritySame |
        componentFailure | contextFailure
  · rcases stateHash_identifies_statement_or_collision leftState rightState
        leftWellFormed rightWellFormed stateDigestEqual with
        stateSame | stateFailure
    · rcases TranscriptReplay.authority_eq_or_challenge_collision oracle
          leftReplay rightReplay alphaEqual gammaEqual roundPointEqual with
          replaySame | replayFailure
      · exact Or.inl ⟨authoritySame, stateSame, replaySame⟩
      · exact Or.inr (Or.inr (Or.inr (Or.inr replayFailure)))
    · exact Or.inr (Or.inr (Or.inr (Or.inl stateFailure)))
  · exact Or.inr (Or.inl componentFailure)
  · exact Or.inr (Or.inr (Or.inl contextFailure))

/-- Raw-authority form of the causal final-state reduction. -/
theorem committed_authority_statement_finalState_identify_or_failure
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (oracle : TranscriptReplay.Oracle K
      NightstreamFPrime.Lifecycle.Transcript.State productionShape)
    (leftAuthority rightAuthority : VerifierContext.Authority)
    (leftState rightState : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (leftReplay rightReplay : TranscriptReplay.ReplayInput K
      NightstreamFPrime.Lifecycle.Transcript.State productionShape)
    (leftWellFormed : StateEncoding.WellFormed leftState)
    (rightWellFormed : StateEncoding.WellFormed rightState)
    (contextDigestEqual :
      (VerifierContext.descriptor leftAuthority).digest4 =
        (VerifierContext.descriptor rightAuthority).digest4)
    (stateDigestEqual : stateHash (publicFits := publicFits) leftState =
      stateHash (publicFits := publicFits) rightState)
    (finalStateEqual : (leftReplay.derive oracle).finalState =
      (rightReplay.derive oracle).finalState) :
    (leftAuthority = rightAuthority ∧ leftState = rightState ∧
      leftReplay.authority = rightReplay.authority) ∨
      AuthorityComponentDigestCollision leftAuthority rightAuthority ∨
      ContextDigestCollision (VerifierContext.descriptor leftAuthority)
        (VerifierContext.descriptor rightAuthority) ∨
      StateHashCollision leftState rightState ∨
      TranscriptReplay.TranscriptStateCollision oracle leftReplay rightReplay := by
  rcases contextDigest_identifies_authority_or_collision leftAuthority
      rightAuthority contextDigestEqual with authoritySame |
        componentFailure | contextFailure
  · rcases stateHash_identifies_statement_or_collision leftState rightState
        leftWellFormed rightWellFormed stateDigestEqual with
        stateSame | stateFailure
    · rcases TranscriptReplay.authority_eq_or_state_collision oracle
          leftReplay rightReplay finalStateEqual with replaySame | replayFailure
      · exact Or.inl ⟨authoritySame, stateSame, replaySame⟩
      · exact Or.inr (Or.inr (Or.inr (Or.inr replayFailure)))
    · exact Or.inr (Or.inr (Or.inr (Or.inl stateFailure)))
  · exact Or.inr (Or.inl componentFailure)
  · exact Or.inr (Or.inr (Or.inl contextFailure))

/-- The HyperNova prior-state link: the digest that the fresh public input
decodes to is the hash of the well-formed prior preimage, the preimage names the
verifier context, and
the NIFS running statement is the preimage's running vector. The context is the
`vk_fs` slot of HyperNova Construction 2's state hash. The key itself is fixed
by the verifier, not by the transcript: unlike Construction 3's `hs = ρ(pp, s)`,
the context does not seed the transcript, and the terminal verifier recomputes
the digest with its own context. The link binds the running statement and the
context digest; it charges no probability to a state-hash collision.
`ActualTerminalSecurity.terminal_implies_nifsOrBaseOrCollision` supplies the
link on the positive, collision-free branch of an accepted recursive terminal. -/
structure PriorLink
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (prior : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (context : KeyDigest) : Prop where
  digest : ProductionKey.priorDigest fresh = stateHash (publicFits := publicFits) prior
  context_eq : prior.verifierKeys functionIndex = context
  running_eq : prior.running functionIndex = running
  wellFormed : StateEncoding.WellFormed prior

/-- The transcript coverage contract with the prior-state link. If two
executions present equal prover-dependent calls before a challenge, they agree
on the prior preimage, the verifier context, the NIFS running statement, the
fresh statement, and every earlier prover message, unless the state hash
collides. `contextDigest_identifies_authority_or_collision` then identifies
the verifier authority behind equal contexts. -/
theorem calls_identify_view_or_collision
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    {degree : Nat} (challenge : TranscriptCoverage.Challenge)
    {prior prior' : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits)}
    {running running' : Running (logicalWidth := logicalWidth) (publicFits := publicFits)}
    {fresh fresh' : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)}
    {context context' : KeyDigest}
    {proof proof' : Proof degree}
    (link : PriorLink prior running fresh context)
    (link' : PriorLink prior' running' fresh' context')
    (same : TranscriptCoverage.proverCalls fresh proof challenge =
      TranscriptCoverage.proverCalls fresh' proof' challenge) :
    (prior = prior' ∧ context = context' ∧ running = running' ∧
        TranscriptCoverage.AgreeOnAbsorbed fresh fresh' proof proof' challenge) ∨
      StateHashCollision prior prior' := by
  have agree := TranscriptCoverage.proverCalls_identify challenge same
  have digestEqual : stateHash (publicFits := publicFits) prior =
      stateHash (publicFits := publicFits) prior' := by
    rw [← link.digest, ← link'.digest, agree.fresh_eq]
  rcases stateHash_identifies_statement_or_collision prior prior'
      link.wellFormed link'.wellFormed digestEqual with priorEqual | collision
  · refine Or.inl ⟨priorEqual, ?_, ?_, agree⟩
    · rw [← link.context_eq, ← link'.context_eq, priorEqual]
    · rw [← link.running_eq, ← link'.running_eq, priorEqual]
  · exact Or.inr collision

end NightstreamFPrime.Layout.Stage1.PiCCSSecurity
