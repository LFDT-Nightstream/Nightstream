import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.Projection.FixedProfile

/-!
Model-level necessity witnesses for the fixed-profile `Pi_CCS` output
projection.

Assurance tier: model-level counterexamples. The typed active output message
plus named derived context uniquely determine `canonicalExpand`; these
theorems instead show that the observable projections alone still permit
noncanonical alternatives when a listed canonicalization obligation is
removed. They do not prove that any particular production R1CS row is the
unique or cheapest way to enforce the missing equality.

Owns: three explicit alternative legacy envelopes that preserve the same
source/transcript-derived context and active `OutputMessage` while changing,
respectively, cached `ct`, padded `yRing`, or padded `yZcol`.

Does not own: authority for the claimed active payload; authority for the
source or transcript views; `PiCCS.Accepted`; Rust/R1CS conformance; the
`m = 257` assignment carrier or its 257-to-270 completion; row counts; or row
removal.

Emits constraints: no.

Authority boundary: every witness is conditional on the concrete
`ContextDerived source transcript context` premise. The witness only proves
that active-message serialization and context equality cannot replace the
separate legacy canonicalization obligations.

| Protocol | Phase | Omitted family | Preserved observations | Counterexample |
|---|---|---|---|---|
| recursive verifier | legacy expansion | `ct` binding | same source view, transcript view, context, active `yRing`, active `yZcol` | alter one independently carried `ct[j]` |
| recursive verifier | legacy expansion | `yRing` padding | same source view, transcript view, context, all active message lanes | set 54-to-64 coefficient tail to `K.one` |
| recursive verifier | legacy expansion | `yZcol` padding | same source view, transcript view, context, all active message lanes | set 54-to-64 coefficient tail to `K.one` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsOutputDigest.Projection.Necessity

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.PiCcsOutputDigest.Projection.FixedProfile

universe uCommitment uPublicInput uEvaluationPoint uColumnPoint uDigest uAdvice

/-- Observational surface available before the legacy canonicalization
obligations are imposed. -/
def CarriesSameDerivedContextAndActivePayload
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {EvaluationPoint : Type uEvaluationPoint}
    {ColumnPoint : Type uColumnPoint}
    {Digest : Type uDigest}
    {Advice : Type uAdvice}
    (source : SourceView Commitment PublicInput Advice)
    (transcript : TranscriptView EvaluationPoint ColumnPoint Digest)
    (context : VerifierContext Commitment PublicInput EvaluationPoint
      ColumnPoint Digest Advice)
    (payload : ActivePayload)
    (legacy : LegacyEnvelope Commitment PublicInput EvaluationPoint ColumnPoint
      Digest Advice) : Prop :=
  ContextDerived source transcript context ∧
    extractContext legacy = context ∧
    extractActivePayload legacy = payload

/-- Pick a concrete `K` value different from the supplied cached value. -/
def differentK (value : K) : K :=
  if value = K.zero then K.one else K.zero

theorem differentK_ne (value : K) : differentK value ≠ value := by
  by_cases zero : value = K.zero
  · subst value
    decide
  · simp [differentK, zero, Ne.symm zero]

/-- Change only one cached `ct` value in the canonical envelope. -/
def alterCt
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {EvaluationPoint : Type uEvaluationPoint}
    {ColumnPoint : Type uColumnPoint}
    {Digest : Type uDigest}
    {Advice : Type uAdvice}
    (context : VerifierContext Commitment PublicInput EvaluationPoint
      ColumnPoint Digest Advice)
    (payload : ActivePayload) :
    LegacyEnvelope Commitment PublicInput EvaluationPoint ColumnPoint Digest
      Advice :=
  let canonical := canonicalExpand context payload
  { canonical with
    ct := fun row =>
      if row = firstRow
      then differentK (canonical.ct firstRow)
      else canonical.ct row }

/-- Change only padded `yRing` coefficient lanes. Active lanes are untouched.
-/
def alterYRingPadding
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {EvaluationPoint : Type uEvaluationPoint}
    {ColumnPoint : Type uColumnPoint}
    {Digest : Type uDigest}
    {Advice : Type uAdvice}
    (context : VerifierContext Commitment PublicInput EvaluationPoint
      ColumnPoint Digest Advice)
    (payload : ActivePayload) :
    LegacyEnvelope Commitment PublicInput EvaluationPoint ColumnPoint Digest
      Advice :=
  let canonical := canonicalExpand context payload
  { canonical with
    yRing := fun row lane =>
      if Nightstream.Implementation.R1CS.PiCcsOutputDigest.Semantics.activeWidth <=
          lane.val
      then K.one
      else canonical.yRing row lane }

/-- Change only padded `yZcol` coefficient lanes. Active lanes are untouched.
-/
def alterYZcolPadding
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {EvaluationPoint : Type uEvaluationPoint}
    {ColumnPoint : Type uColumnPoint}
    {Digest : Type uDigest}
    {Advice : Type uAdvice}
    (context : VerifierContext Commitment PublicInput EvaluationPoint
      ColumnPoint Digest Advice)
    (payload : ActivePayload) :
    LegacyEnvelope Commitment PublicInput EvaluationPoint ColumnPoint Digest
      Advice :=
  let canonical := canonicalExpand context payload
  { canonical with
    yZcol := fun lane =>
      if Nightstream.Implementation.R1CS.PiCcsOutputDigest.Semantics.activeWidth <=
          lane.val
      then K.one
      else canonical.yZcol lane }

@[simp] theorem extractContext_alterCt
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {EvaluationPoint : Type uEvaluationPoint}
    {ColumnPoint : Type uColumnPoint}
    {Digest : Type uDigest}
    {Advice : Type uAdvice}
    (context : VerifierContext Commitment PublicInput EvaluationPoint
      ColumnPoint Digest Advice)
    (payload : ActivePayload) :
    extractContext (alterCt context payload) = context := by
  change extractContext (canonicalExpand context payload) = context
  exact extractContext_canonicalExpand context payload

@[simp] theorem extractActivePayload_alterCt
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {EvaluationPoint : Type uEvaluationPoint}
    {ColumnPoint : Type uColumnPoint}
    {Digest : Type uDigest}
    {Advice : Type uAdvice}
    (context : VerifierContext Commitment PublicInput EvaluationPoint
      ColumnPoint Digest Advice)
    (payload : ActivePayload) :
    extractActivePayload (alterCt context payload) = payload := by
  change extractActivePayload (canonicalExpand context payload) = payload
  exact extractActivePayload_canonicalExpand context payload

@[simp] theorem extractContext_alterYRingPadding
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {EvaluationPoint : Type uEvaluationPoint}
    {ColumnPoint : Type uColumnPoint}
    {Digest : Type uDigest}
    {Advice : Type uAdvice}
    (context : VerifierContext Commitment PublicInput EvaluationPoint
      ColumnPoint Digest Advice)
    (payload : ActivePayload) :
    extractContext (alterYRingPadding context payload) = context := by
  change extractContext (canonicalExpand context payload) = context
  exact extractContext_canonicalExpand context payload

theorem activePayload_ext
    {left right : ActivePayload}
    (yRing : left.yRing = right.yRing)
    (yZcol : left.yZcol = right.yZcol) :
    left = right := by
  cases left
  cases right
  simp_all

@[simp] theorem extractActivePayload_alterYRingPadding
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {EvaluationPoint : Type uEvaluationPoint}
    {ColumnPoint : Type uColumnPoint}
    {Digest : Type uDigest}
    {Advice : Type uAdvice}
    (context : VerifierContext Commitment PublicInput EvaluationPoint
      ColumnPoint Digest Advice)
    (payload : ActivePayload) :
    extractActivePayload (alterYRingPadding context payload) = payload := by
  apply activePayload_ext
  · funext row lane
    change
      (if Nightstream.Implementation.R1CS.PiCcsOutputDigest.Semantics.activeWidth <=
          (activeLane lane).val
       then K.one
       else zeroExtend (payload.yRing row) (activeLane lane)) =
        payload.yRing row lane
    have notPadding :
        ¬ Nightstream.Implementation.R1CS.PiCcsOutputDigest.Semantics.activeWidth <=
          (activeLane lane).val := by
      simp [activeLane, Nat.not_le.mpr lane.isLt]
    rw [if_neg notPadding]
    exact zeroExtend_active (payload.yRing row) lane
  · funext lane
    simp [extractActivePayload, alterYRingPadding, canonicalExpand]

@[simp] theorem extractContext_alterYZcolPadding
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {EvaluationPoint : Type uEvaluationPoint}
    {ColumnPoint : Type uColumnPoint}
    {Digest : Type uDigest}
    {Advice : Type uAdvice}
    (context : VerifierContext Commitment PublicInput EvaluationPoint
      ColumnPoint Digest Advice)
    (payload : ActivePayload) :
    extractContext (alterYZcolPadding context payload) = context := by
  change extractContext (canonicalExpand context payload) = context
  exact extractContext_canonicalExpand context payload

@[simp] theorem extractActivePayload_alterYZcolPadding
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {EvaluationPoint : Type uEvaluationPoint}
    {ColumnPoint : Type uColumnPoint}
    {Digest : Type uDigest}
    {Advice : Type uAdvice}
    (context : VerifierContext Commitment PublicInput EvaluationPoint
      ColumnPoint Digest Advice)
    (payload : ActivePayload) :
    extractActivePayload (alterYZcolPadding context payload) = payload := by
  apply activePayload_ext
  · funext row lane
    simp [extractActivePayload, alterYZcolPadding, canonicalExpand]
  · funext lane
    change
      (if Nightstream.Implementation.R1CS.PiCcsOutputDigest.Semantics.activeWidth <=
          (activeLane lane).val
       then K.one
       else zeroExtend payload.yZcol (activeLane lane)) = payload.yZcol lane
    have notPadding :
        ¬ Nightstream.Implementation.R1CS.PiCcsOutputDigest.Semantics.activeWidth <=
          (activeLane lane).val := by
      simp [activeLane, Nat.not_le.mpr lane.isLt]
    rw [if_neg notPadding]
    exact zeroExtend_active payload.yZcol lane

theorem alterCt_differs
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {EvaluationPoint : Type uEvaluationPoint}
    {ColumnPoint : Type uColumnPoint}
    {Digest : Type uDigest}
    {Advice : Type uAdvice}
    (context : VerifierContext Commitment PublicInput EvaluationPoint
      ColumnPoint Digest Advice)
    (payload : ActivePayload) :
    (alterCt context payload).ct ≠ (canonicalExpand context payload).ct := by
  intro equal
  have atFirstRow := congrFun equal firstRow
  exact differentK_ne ((canonicalExpand context payload).ct firstRow)
    (by simpa [alterCt] using atFirstRow)

theorem alterYRingPadding_differs
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {EvaluationPoint : Type uEvaluationPoint}
    {ColumnPoint : Type uColumnPoint}
    {Digest : Type uDigest}
    {Advice : Type uAdvice}
    (context : VerifierContext Commitment PublicInput EvaluationPoint
      ColumnPoint Digest Advice)
    (payload : ActivePayload) :
    (alterYRingPadding context payload).yRing ≠
      (canonicalExpand context payload).yRing := by
  intro equal
  have atFirstPadding := congrFun (congrFun equal firstRow) firstPaddingLane
  have one_ne_zero : K.one ≠ K.zero := by decide
  exact one_ne_zero (by
    simpa [alterYRingPadding, firstPaddingLane] using atFirstPadding)

theorem alterYZcolPadding_differs
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {EvaluationPoint : Type uEvaluationPoint}
    {ColumnPoint : Type uColumnPoint}
    {Digest : Type uDigest}
    {Advice : Type uAdvice}
    (context : VerifierContext Commitment PublicInput EvaluationPoint
      ColumnPoint Digest Advice)
    (payload : ActivePayload) :
    (alterYZcolPadding context payload).yZcol ≠
      (canonicalExpand context payload).yZcol := by
  intro equal
  have atFirstPadding := congrFun equal firstPaddingLane
  have one_ne_zero : K.one ≠ K.zero := by decide
  exact one_ne_zero (by
    simpa [alterYZcolPadding, firstPaddingLane] using atFirstPadding)

theorem alterCt_sameProjection_notCanonical
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {EvaluationPoint : Type uEvaluationPoint}
    {ColumnPoint : Type uColumnPoint}
    {Digest : Type uDigest}
    {Advice : Type uAdvice}
    {source : SourceView Commitment PublicInput Advice}
    {transcript : TranscriptView EvaluationPoint ColumnPoint Digest}
    {context : VerifierContext Commitment PublicInput EvaluationPoint
      ColumnPoint Digest Advice}
    (payload : ActivePayload)
    (contextDerived : ContextDerived source transcript context) :
    CarriesSameDerivedContextAndActivePayload source transcript context payload
        (alterCt context payload) ∧
      ¬ CanonicalExpansion context payload (alterCt context payload) := by
  constructor
  · exact ⟨contextDerived, by simp, by simp⟩
  · intro canonical
    have equal := eq_canonicalExpand_of_canonicalExpansion canonical
    exact alterCt_differs context payload (congrArg LegacyEnvelope.ct equal)

theorem alterYRingPadding_sameProjection_notCanonical
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {EvaluationPoint : Type uEvaluationPoint}
    {ColumnPoint : Type uColumnPoint}
    {Digest : Type uDigest}
    {Advice : Type uAdvice}
    {source : SourceView Commitment PublicInput Advice}
    {transcript : TranscriptView EvaluationPoint ColumnPoint Digest}
    {context : VerifierContext Commitment PublicInput EvaluationPoint
      ColumnPoint Digest Advice}
    (payload : ActivePayload)
    (contextDerived : ContextDerived source transcript context) :
    CarriesSameDerivedContextAndActivePayload source transcript context payload
        (alterYRingPadding context payload) ∧
      ¬ CanonicalExpansion context payload
        (alterYRingPadding context payload) := by
  constructor
  · exact ⟨contextDerived, by simp, by simp⟩
  · intro canonical
    have equal := eq_canonicalExpand_of_canonicalExpansion canonical
    exact alterYRingPadding_differs context payload
      (congrArg LegacyEnvelope.yRing equal)

theorem alterYZcolPadding_sameProjection_notCanonical
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {EvaluationPoint : Type uEvaluationPoint}
    {ColumnPoint : Type uColumnPoint}
    {Digest : Type uDigest}
    {Advice : Type uAdvice}
    {source : SourceView Commitment PublicInput Advice}
    {transcript : TranscriptView EvaluationPoint ColumnPoint Digest}
    {context : VerifierContext Commitment PublicInput EvaluationPoint
      ColumnPoint Digest Advice}
    (payload : ActivePayload)
    (contextDerived : ContextDerived source transcript context) :
    CarriesSameDerivedContextAndActivePayload source transcript context payload
        (alterYZcolPadding context payload) ∧
      ¬ CanonicalExpansion context payload
        (alterYZcolPadding context payload) := by
  constructor
  · exact ⟨contextDerived, by simp, by simp⟩
  · intro canonical
    have equal := eq_canonicalExpand_of_canonicalExpansion canonical
    exact alterYZcolPadding_differs context payload
      (congrArg LegacyEnvelope.yZcol equal)

end Nightstream.Implementation.R1CS.PiCcsOutputDigest.Projection.Necessity
