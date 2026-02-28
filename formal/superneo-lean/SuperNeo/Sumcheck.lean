import SuperNeo.Field

/-! Sumcheck identities, checks, and reduction glue. -/

/-!
Third-party attribution policy for this module:
- If code is copied/adapted from ArkLib, keep a local comment near the adapted
  block with upstream repo/path/commit/license and a short modification note.
- Do not import external text verbatim without attribution.
-/

namespace SuperNeo

open F

/-- Boolean hypercube point encoded as field elements in `{0,1}`. -/
def boolPointOfNat (ell idx : Nat) : Array F :=
  Array.ofFn (n := ell) (fun j : Fin ell =>
    F.ofNat (((idx / (Nat.pow 2 j.1)) % 2)))

theorem boolPointOfNat_size (ell idx : Nat) :
  (boolPointOfNat ell idx).size = ell := by
  unfold boolPointOfNat
  simp

/-- Sum of a multivariate field function over `{0,1}^ell`. -/
def sumOverBoolHypercube (ell : Nat) (q : Array F -> F) : F :=
  Id.run do
    let mut acc : F := 0
    for idx in [0:Nat.pow 2 ell] do
      acc := acc + q (boolPointOfNat ell idx)
    return acc

/-- Public instance for a SumCheck claim `T = sum_{x in {0,1}^ell} Q(x)`. -/
structure SumcheckInstance where
  ell : Nat
  q : Array F -> F
  claimedSum : F

/-- Output claim produced by SumCheck after sampling `r`. -/
structure SumcheckOutput where
  r : Array F
  value : F

def SumcheckOutputShape (inst : SumcheckInstance) (out : SumcheckOutput) : Prop :=
  out.r.size = inst.ell

def SumcheckOutputValid (inst : SumcheckInstance) (out : SumcheckOutput) : Prop :=
  SumcheckOutputShape inst out ∧ out.value = inst.q out.r

def SumcheckClaimTrue (inst : SumcheckInstance) : Prop :=
  sumOverBoolHypercube inst.ell inst.q = inst.claimedSum

def SumcheckResultValid (inst : SumcheckInstance) (out : SumcheckOutput) : Prop :=
  SumcheckClaimTrue inst ∧ SumcheckOutputValid inst out

theorem sumcheckOutputValid_shape
  {inst : SumcheckInstance} {out : SumcheckOutput}
  (h : SumcheckOutputValid inst out) :
  SumcheckOutputShape inst out := by
  exact h.1

theorem sumcheckOutputValid_value
  {inst : SumcheckInstance} {out : SumcheckOutput}
  (h : SumcheckOutputValid inst out) :
  out.value = inst.q out.r := by
  exact h.2

theorem sumcheckResultValid_claim
  {inst : SumcheckInstance} {out : SumcheckOutput}
  (h : SumcheckResultValid inst out) :
  SumcheckClaimTrue inst := by
  exact h.1

theorem sumcheckResultValid_output
  {inst : SumcheckInstance} {out : SumcheckOutput}
  (h : SumcheckResultValid inst out) :
  SumcheckOutputValid inst out := by
  exact h.2

/--
Transcript shell for protocol-facing interfaces.
`roundPolys` stores per-round prover messages in a lightweight representation.
-/
structure SumcheckTranscript where
  roundPolys : Array (Array F)
  r : Array F
  value : F

def SumcheckTranscriptShape (inst : SumcheckInstance) (tr : SumcheckTranscript) : Prop :=
  tr.roundPolys.size = inst.ell ∧ tr.r.size = inst.ell

def SumcheckTranscript.toOutput (tr : SumcheckTranscript) : SumcheckOutput :=
  { r := tr.r, value := tr.value }

/--
Accepted transcript interface (minimal shell):
shape checks plus final point-evaluation check.
Detailed polynomial-round consistency is intentionally deferred to later modules.
-/
def SumcheckAcceptedProp (inst : SumcheckInstance) (tr : SumcheckTranscript) : Prop :=
  SumcheckTranscriptShape inst tr ∧ tr.value = inst.q tr.r

/--
Evaluate a round polynomial represented by its values at `0` and `1`.
If shape is invalid, returns `0` as a conservative default.
-/
def sumcheckLineEval01 (vals01 : Array F) (r : F) : F :=
  if vals01.size != 2 then
    0
  else
    vals01[0]! + r * (vals01[1]! - vals01[0]!)

/-- Sum of round polynomial endpoint values (`g(0)+g(1)`) under 2-point encoding. -/
def sumcheckLineSum01 (vals01 : Array F) : F :=
  if vals01.size != 2 then
    0
  else
    vals01[0]! + vals01[1]!

/-- Every round polynomial is represented by exactly two endpoint values. -/
def SumcheckRoundShape (tr : SumcheckTranscript) : Prop :=
  ∀ i (hi : i < tr.roundPolys.size), (tr.roundPolys[i]'hi).size = 2

/--
Per-round chain constraints for the 2-point (`0/1`) round-polynomial encoding.
This keeps transcript consistency explicit while staying lightweight.
-/
def SumcheckRoundConsistency (inst : SumcheckInstance) (tr : SumcheckTranscript) : Prop :=
  SumcheckTranscriptShape inst tr ∧
    SumcheckRoundShape tr ∧
    (inst.ell = 0 ∨ inst.claimedSum = sumcheckLineSum01 tr.roundPolys[0]!) ∧
    (∀ i : Nat, i + 1 < inst.ell ->
      sumcheckLineEval01 tr.roundPolys[i]! tr.r[i]! = sumcheckLineSum01 tr.roundPolys[i + 1]!) ∧
    (inst.ell = 0 ∨
      sumcheckLineEval01 tr.roundPolys[inst.ell - 1]! tr.r[inst.ell - 1]! = tr.value)

/--
Stronger acceptance predicate: shape/evaluation acceptance plus per-round chain consistency.
-/
def SumcheckAcceptedStrongProp (inst : SumcheckInstance) (tr : SumcheckTranscript) : Prop :=
  SumcheckAcceptedProp inst tr ∧ SumcheckRoundConsistency inst tr

theorem sumcheckRoundConsistency_transcriptShape
  {inst : SumcheckInstance} {tr : SumcheckTranscript}
  (hCons : SumcheckRoundConsistency inst tr) :
  SumcheckTranscriptShape inst tr := by
  exact hCons.1

theorem sumcheckRoundConsistency_roundShape
  {inst : SumcheckInstance} {tr : SumcheckTranscript}
  (hCons : SumcheckRoundConsistency inst tr) :
  SumcheckRoundShape tr := by
  exact hCons.2.1

theorem sumcheckRoundConsistency_rootLine
  {inst : SumcheckInstance} {tr : SumcheckTranscript}
  (hCons : SumcheckRoundConsistency inst tr) :
  inst.ell = 0 ∨ inst.claimedSum = sumcheckLineSum01 tr.roundPolys[0]! := by
  exact hCons.2.2.1

theorem sumcheckRoundConsistency_chain
  {inst : SumcheckInstance} {tr : SumcheckTranscript}
  (hCons : SumcheckRoundConsistency inst tr) :
  ∀ i : Nat, i + 1 < inst.ell ->
    sumcheckLineEval01 tr.roundPolys[i]! tr.r[i]! = sumcheckLineSum01 tr.roundPolys[i + 1]! := by
  exact hCons.2.2.2.1

theorem sumcheckRoundConsistency_finalLine
  {inst : SumcheckInstance} {tr : SumcheckTranscript}
  (hCons : SumcheckRoundConsistency inst tr) :
  inst.ell = 0 ∨
    sumcheckLineEval01 tr.roundPolys[inst.ell - 1]! tr.r[inst.ell - 1]! = tr.value := by
  exact hCons.2.2.2.2

theorem sumcheckAcceptedStrong_implies_accepted
  {inst : SumcheckInstance} {tr : SumcheckTranscript}
  (hStrong : SumcheckAcceptedStrongProp inst tr) :
  SumcheckAcceptedProp inst tr := by
  exact hStrong.1

theorem sumcheckAcceptedStrong_roundConsistency
  {inst : SumcheckInstance} {tr : SumcheckTranscript}
  (hStrong : SumcheckAcceptedStrongProp inst tr) :
  SumcheckRoundConsistency inst tr := by
  exact hStrong.2

theorem sumcheckAcceptedStrong_shape
  {inst : SumcheckInstance} {tr : SumcheckTranscript}
  (hStrong : SumcheckAcceptedStrongProp inst tr) :
  SumcheckTranscriptShape inst tr := by
  exact sumcheckRoundConsistency_transcriptShape (sumcheckAcceptedStrong_roundConsistency hStrong)

theorem sumcheckAcceptedStrong_roundShape
  {inst : SumcheckInstance} {tr : SumcheckTranscript}
  (hStrong : SumcheckAcceptedStrongProp inst tr) :
  SumcheckRoundShape tr := by
  exact sumcheckRoundConsistency_roundShape (sumcheckAcceptedStrong_roundConsistency hStrong)

/-- Soundness boundary interface for SumCheck. -/
def SumcheckSoundnessAssumption : Prop :=
  ∀ inst tr, SumcheckAcceptedProp inst tr -> SumcheckResultValid inst tr.toOutput

/-- Strong soundness boundary (consumes the stronger accepted predicate). -/
def SumcheckStrongSoundnessAssumption : Prop :=
  ∀ inst tr, SumcheckAcceptedStrongProp inst tr -> SumcheckResultValid inst tr.toOutput

/-- Completeness boundary interface for SumCheck. -/
def SumcheckCompletenessAssumption : Prop :=
  ∀ inst, SumcheckClaimTrue inst -> ∃ tr, SumcheckAcceptedProp inst tr

def SumcheckProtocolAssumption : Prop :=
  SumcheckSoundnessAssumption ∧ SumcheckCompletenessAssumption

theorem sumcheckAccepted_implies_result_of_assumption
  (hSound : SumcheckSoundnessAssumption)
  {inst : SumcheckInstance} {tr : SumcheckTranscript}
  (hAccepted : SumcheckAcceptedProp inst tr) :
  SumcheckResultValid inst tr.toOutput := by
  exact hSound inst tr hAccepted

theorem sumcheckAcceptedStrong_implies_result_of_assumption
  (hSound : SumcheckStrongSoundnessAssumption)
  {inst : SumcheckInstance} {tr : SumcheckTranscript}
  (hAccepted : SumcheckAcceptedStrongProp inst tr) :
  SumcheckResultValid inst tr.toOutput := by
  exact hSound inst tr hAccepted

theorem sumcheckStrongSoundnessAssumption_of_soundnessAssumption
  (hSound : SumcheckSoundnessAssumption) :
  SumcheckStrongSoundnessAssumption := by
  intro inst tr hAcceptedStrong
  exact hSound inst tr (sumcheckAcceptedStrong_implies_accepted hAcceptedStrong)

theorem sumcheckAccepted_implies_claim_of_assumption
  (hSound : SumcheckSoundnessAssumption)
  {inst : SumcheckInstance} {tr : SumcheckTranscript}
  (hAccepted : SumcheckAcceptedProp inst tr) :
  SumcheckClaimTrue inst := by
  exact (sumcheckAccepted_implies_result_of_assumption hSound hAccepted).1

theorem sumcheckAccepted_implies_outputValid_of_assumption
  (hSound : SumcheckSoundnessAssumption)
  {inst : SumcheckInstance} {tr : SumcheckTranscript}
  (hAccepted : SumcheckAcceptedProp inst tr) :
  SumcheckOutputValid inst tr.toOutput := by
  exact (sumcheckAccepted_implies_result_of_assumption hSound hAccepted).2

theorem sumcheckCompleteness_of_assumption
  (hComp : SumcheckCompletenessAssumption)
  {inst : SumcheckInstance}
  (hClaim : SumcheckClaimTrue inst) :
  ∃ tr, SumcheckAcceptedProp inst tr := by
  exact hComp inst hClaim

/--
Executable table-based checker for the root SumCheck claim.
`values[idx]` is interpreted as `Q(boolPointOfNat ell idx)`.
-/
def sumcheckTableClaimCheck (ell : Nat) (values : Array F) (claimedSum : F) : Bool :=
  if values.size != Nat.pow 2 ell then
    false
  else
    decide (values.foldl (fun acc x => acc + x) (0 : F) = claimedSum)

def sumcheckTableClaimProp (ell : Nat) (values : Array F) (claimedSum : F) : Prop :=
  values.size = Nat.pow 2 ell ∧
    values.foldl (fun acc x => acc + x) (0 : F) = claimedSum

theorem sumcheckTableClaimCheck_sound
  {ell : Nat} {values : Array F} {claimedSum : F}
  (hOk : sumcheckTableClaimCheck ell values claimedSum = true) :
  sumcheckTableClaimProp ell values claimedSum := by
  simpa [sumcheckTableClaimCheck, sumcheckTableClaimProp] using hOk

theorem sumcheckTableClaimCheck_complete
  {ell : Nat} {values : Array F} {claimedSum : F}
  (hProp : sumcheckTableClaimProp ell values claimedSum) :
  sumcheckTableClaimCheck ell values claimedSum = true := by
  simpa [sumcheckTableClaimCheck, sumcheckTableClaimProp] using hProp

theorem sumcheckTableClaimCheck_iff_prop
  {ell : Nat} {values : Array F} {claimedSum : F} :
  sumcheckTableClaimCheck ell values claimedSum = true ↔
    sumcheckTableClaimProp ell values claimedSum := by
  constructor
  · exact sumcheckTableClaimCheck_sound
  · exact sumcheckTableClaimCheck_complete

end SuperNeo
