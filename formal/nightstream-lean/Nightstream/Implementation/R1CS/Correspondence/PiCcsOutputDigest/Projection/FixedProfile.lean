import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.Semantics

/-!
Fixed-profile projection from typed claimed `Pi_CCS` output data to the complete
legacy CE envelope carried by the current recursive verifier.

Assurance tier: model-level. This module defines a typed reconstruction
contract. It does not prove that `PiCCS.Accepted`, Rust, or R1CS rows provide
the correct claimed payload or derive the verifier context.

Owns: the typed distinction between three claimed CE evaluation rows and one
claimed active SplitNC sidecar; coefficient-lane zero extension from 54 to 64;
the complete legacy-envelope field list; `ct[j] = yRing[j][0]`; canonical
empty auxiliary sidecars; zero Pattern-A offsets; and reconstruction
uniqueness.

Does not own: the `m = 257` assignment carrier or any tail in that carrier;
paper-to-payload authority; derivation of commitment, `X`, `r`, `s_col`,
`m_in`, header digest, or `adv`; Rust/R1CS conformance; constraint counts;
necessity of production rows; or row removal.

Emits constraints: no.

Authority boundary: `ActivePayload` reuses the independently serialized
`OutputMessage`. Its `yRing` is typed claimed CE evaluation data and its
`yZcol` is a typed claimed active SplitNC message field; neither is established
as semantically true here. `canonicalExpand` is deterministic legacy
representation only; it creates no authority. `ContextDerived` spells out
the source-view and transcript-view equalities, but proving that those views
are authoritative remains an open premise for later paper/Rust/R1CS work.

| Protocol | Phase | Family | Field/shape | Model-level obligation |
|---|---|---|---|---|
| `Pi_CCS` | output projection | CE evaluations | `yRing : Fin 3 -> Fin 54 -> K` | typed claimed active payload |
| SplitNC | output projection | delayed NC sidecar | `yZcol : Fin 54 -> K` | typed claimed payload, distinct from CE evaluations |
| recursive verifier | legacy expansion | coefficient layout | `Fin 54 -> Fin 64` | preserve active lanes and append ten `K.zero` lanes |
| recursive verifier | legacy expansion | cached scalar | `ct[j]` | derive from padded `yRing[j][0]` |
| recursive verifier | legacy expansion | unsupported sidecars | `auxOpenings`, `cStepCoords` | both are exactly empty |
| recursive verifier | legacy expansion | Pattern-A offsets | `uOffset`, `uLen` | both are exactly zero |
| recursive verifier | claim context | source view | commitment, `X`, `m_in`, `adv` | named source-to-context equalities |
| recursive verifier | claim context | transcript view | `r`, `s_col`, fold digest | named transcript-to-context equalities |
-/

namespace Nightstream.Implementation.R1CS.PiCcsOutputDigest.Projection.FixedProfile

open Nightstream.SuperNeo.Concrete

universe uCommitment uPublicInput uEvaluationPoint uColumnPoint uDigest uAdvice

/-- Typed claimed output information committed by the output-message
serialization. This type does not assert that either field is the correct
evaluation of an authoritative witness. -/
abbrev ActivePayload :=
  Nightstream.Implementation.R1CS.PiCcsOutputDigest.Semantics.OutputMessage

/-- The legacy recursive representation pads coefficient vectors to 64. This
is unrelated to assignment-carrier dimensions such as the current `m = 257`.
-/
def paddedWidth : Nat := 64

theorem activeWidth_eq_54 :
    Nightstream.Implementation.R1CS.PiCcsOutputDigest.Semantics.activeWidth =
      54 := by
  rfl

theorem yRingRows_eq_3 :
    Nightstream.Implementation.R1CS.PiCcsOutputDigest.Semantics.yRingRows =
      3 := by
  rfl

theorem activeWidth_lt_paddedWidth :
    Nightstream.Implementation.R1CS.PiCcsOutputDigest.Semantics.activeWidth <
      paddedWidth := by
  decide

/-- Embed an active coefficient index into the padded legacy index. -/
def activeLane
    (lane : Fin
      Nightstream.Implementation.R1CS.PiCcsOutputDigest.Semantics.activeWidth) :
    Fin paddedWidth :=
  ⟨lane.val, Nat.lt_trans lane.isLt activeWidth_lt_paddedWidth⟩

/-- First coefficient lane omitted from the active output digest. -/
def firstPaddingLane : Fin paddedWidth :=
  ⟨Nightstream.Implementation.R1CS.PiCcsOutputDigest.Semantics.activeWidth,
    activeWidth_lt_paddedWidth⟩

/-- First CE output row, used only to state concrete necessity witnesses. -/
def firstRow : Fin
    Nightstream.Implementation.R1CS.PiCcsOutputDigest.Semantics.yRingRows :=
  ⟨0, by decide⟩

/-- First active coefficient, hence the cached `ct` coefficient. -/
def constantLane : Fin
    Nightstream.Implementation.R1CS.PiCcsOutputDigest.Semantics.activeWidth :=
  ⟨0, by decide⟩

/-- Typed zero extension from the claimed active 54 coefficients to the legacy 64
coefficient slots. -/
def zeroExtend
    (values : Fin
      Nightstream.Implementation.R1CS.PiCcsOutputDigest.Semantics.activeWidth ->
      K) :
    Fin paddedWidth -> K :=
  fun lane =>
    if active : lane.val <
        Nightstream.Implementation.R1CS.PiCcsOutputDigest.Semantics.activeWidth
    then values ⟨lane.val, active⟩
    else K.zero

@[simp] theorem zeroExtend_active
    (values : Fin
      Nightstream.Implementation.R1CS.PiCcsOutputDigest.Semantics.activeWidth ->
      K)
    (lane : Fin
      Nightstream.Implementation.R1CS.PiCcsOutputDigest.Semantics.activeWidth) :
    zeroExtend values (activeLane lane) = values lane := by
  simp [zeroExtend, activeLane, lane.isLt]

theorem zeroExtend_padding
    (values : Fin
      Nightstream.Implementation.R1CS.PiCcsOutputDigest.Semantics.activeWidth ->
      K)
    (lane : Fin paddedWidth)
    (padding :
      Nightstream.Implementation.R1CS.PiCcsOutputDigest.Semantics.activeWidth <=
        lane.val) :
    zeroExtend values lane = K.zero := by
  simp [zeroExtend, Nat.not_lt.mpr padding]

@[simp] theorem zeroExtend_firstPadding
    (values : Fin
    Nightstream.Implementation.R1CS.PiCcsOutputDigest.Semantics.activeWidth ->
      K) :
    zeroExtend values firstPaddingLane = K.zero := by
  apply zeroExtend_padding
  simp [firstPaddingLane]

/-- Source-owned view from which inherited output-claim fields must be
derived. A later theorem must bind this view to accepted authoritative inputs.
-/
structure SourceView
    (Commitment : Type uCommitment)
    (PublicInput : Type uPublicInput)
    (Advice : Type uAdvice) where
  commitment : Commitment
  publicInputX : PublicInput
  inputLengthMIn : Nat
  adviceCommitment : Advice

/-- Verifier-transcript view from which challenge and header fields must be
derived. A later theorem must bind this view to the verifier-owned transcript.
-/
structure TranscriptView
    (EvaluationPoint : Type uEvaluationPoint)
    (ColumnPoint : Type uColumnPoint)
    (Digest : Type uDigest) where
  evaluationPointR : EvaluationPoint
  columnPointSCol : ColumnPoint
  foldDigest : Digest

/-- Named verifier context carried by a complete CE claim. -/
structure VerifierContext
    (Commitment : Type uCommitment)
    (PublicInput : Type uPublicInput)
    (EvaluationPoint : Type uEvaluationPoint)
    (ColumnPoint : Type uColumnPoint)
    (Digest : Type uDigest)
    (Advice : Type uAdvice) where
  commitment : Commitment
  publicInputX : PublicInput
  evaluationPointR : EvaluationPoint
  columnPointSCol : ColumnPoint
  inputLengthMIn : Nat
  foldDigest : Digest
  adviceCommitment : Advice

/-- The unique context assembled from its two named authority views. This is
pure reconstruction; it does not establish authority for either view. -/
def deriveContext
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {EvaluationPoint : Type uEvaluationPoint}
    {ColumnPoint : Type uColumnPoint}
    {Digest : Type uDigest}
    {Advice : Type uAdvice}
    (source : SourceView Commitment PublicInput Advice)
    (transcript : TranscriptView EvaluationPoint ColumnPoint Digest) :
    VerifierContext Commitment PublicInput EvaluationPoint ColumnPoint Digest
      Advice :=
  { commitment := source.commitment
    publicInputX := source.publicInputX
    evaluationPointR := transcript.evaluationPointR
    columnPointSCol := transcript.columnPointSCol
    inputLengthMIn := source.inputLengthMIn
    foldDigest := transcript.foldDigest
    adviceCommitment := source.adviceCommitment }

/-- Real, non-opaque derivation relation for the carried claim context. The
relation is an open premise at this assurance tier because this module does
not prove that `source` and `transcript` come from accepted verifier state. -/
def ContextDerived
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {EvaluationPoint : Type uEvaluationPoint}
    {ColumnPoint : Type uColumnPoint}
    {Digest : Type uDigest}
    {Advice : Type uAdvice}
    (source : SourceView Commitment PublicInput Advice)
    (transcript : TranscriptView EvaluationPoint ColumnPoint Digest)
    (context : VerifierContext Commitment PublicInput EvaluationPoint
      ColumnPoint Digest Advice) : Prop :=
  context = deriveContext source transcript

/-- Complete fixed-profile legacy envelope, including fields omitted from the
active output digest. Unsupported sidecars remain visible so their canonical
emptiness cannot disappear behind a smaller type. -/
structure LegacyEnvelope
    (Commitment : Type uCommitment)
    (PublicInput : Type uPublicInput)
    (EvaluationPoint : Type uEvaluationPoint)
    (ColumnPoint : Type uColumnPoint)
    (Digest : Type uDigest)
    (Advice : Type uAdvice) where
  commitment : Commitment
  publicInputX : PublicInput
  evaluationPointR : EvaluationPoint
  columnPointSCol : ColumnPoint
  yRing : Fin
    Nightstream.Implementation.R1CS.PiCcsOutputDigest.Semantics.yRingRows ->
    Fin paddedWidth -> K
  ct : Fin
    Nightstream.Implementation.R1CS.PiCcsOutputDigest.Semantics.yRingRows -> K
  auxOpenings : List K
  yZcol : Fin paddedWidth -> K
  inputLengthMIn : Nat
  foldDigest : Digest
  cStepCoords : List F
  uOffset : Nat
  uLen : Nat
  adviceCommitment : Advice

/-- Read back only the fields whose authority is assigned to the verifier
context. -/
def extractContext
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {EvaluationPoint : Type uEvaluationPoint}
    {ColumnPoint : Type uColumnPoint}
    {Digest : Type uDigest}
    {Advice : Type uAdvice}
    (legacy : LegacyEnvelope Commitment PublicInput EvaluationPoint ColumnPoint
      Digest Advice) :
    VerifierContext Commitment PublicInput EvaluationPoint ColumnPoint Digest
      Advice :=
  { commitment := legacy.commitment
    publicInputX := legacy.publicInputX
    evaluationPointR := legacy.evaluationPointR
    columnPointSCol := legacy.columnPointSCol
    inputLengthMIn := legacy.inputLengthMIn
    foldDigest := legacy.foldDigest
    adviceCommitment := legacy.adviceCommitment }

/-- Project a legacy claim back to the typed claimed active message. `ct`, padded
lanes, and unsupported sidecars cannot affect this projection. -/
def extractActivePayload
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {EvaluationPoint : Type uEvaluationPoint}
    {ColumnPoint : Type uColumnPoint}
    {Digest : Type uDigest}
    {Advice : Type uAdvice}
    (legacy : LegacyEnvelope Commitment PublicInput EvaluationPoint ColumnPoint
      Digest Advice) : ActivePayload :=
  { yRing := fun row lane => legacy.yRing row (activeLane lane)
    yZcol := fun lane => legacy.yZcol (activeLane lane) }

/-- Deterministic expansion of typed claimed active data and named verifier
context into the exact supported legacy profile. -/
def canonicalExpand
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
  { commitment := context.commitment
    publicInputX := context.publicInputX
    evaluationPointR := context.evaluationPointR
    columnPointSCol := context.columnPointSCol
    yRing := fun row => zeroExtend (payload.yRing row)
    ct := fun row => payload.yRing row constantLane
    auxOpenings := []
    yZcol := zeroExtend payload.yZcol
    inputLengthMIn := context.inputLengthMIn
    foldDigest := context.foldDigest
    cStepCoords := []
    uOffset := 0
    uLen := 0
    adviceCommitment := context.adviceCommitment }

@[simp] theorem extractContext_canonicalExpand
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {EvaluationPoint : Type uEvaluationPoint}
    {ColumnPoint : Type uColumnPoint}
    {Digest : Type uDigest}
    {Advice : Type uAdvice}
    (context : VerifierContext Commitment PublicInput EvaluationPoint
      ColumnPoint Digest Advice)
    (payload : ActivePayload) :
    extractContext (canonicalExpand context payload) = context := by
  cases context
  rfl

@[simp] theorem extractActivePayload_canonicalExpand
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {EvaluationPoint : Type uEvaluationPoint}
    {ColumnPoint : Type uColumnPoint}
    {Digest : Type uDigest}
    {Advice : Type uAdvice}
    (context : VerifierContext Commitment PublicInput EvaluationPoint
      ColumnPoint Digest Advice)
    (payload : ActivePayload) :
    extractActivePayload (canonicalExpand context payload) = payload := by
  cases payload
  simp [extractActivePayload, canonicalExpand]

@[simp] theorem canonicalExpand_ct
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {EvaluationPoint : Type uEvaluationPoint}
    {ColumnPoint : Type uColumnPoint}
    {Digest : Type uDigest}
    {Advice : Type uAdvice}
    (context : VerifierContext Commitment PublicInput EvaluationPoint
      ColumnPoint Digest Advice)
    (payload : ActivePayload)
    (row : Fin
      Nightstream.Implementation.R1CS.PiCcsOutputDigest.Semantics.yRingRows) :
    (canonicalExpand context payload).ct row =
      (canonicalExpand context payload).yRing row (activeLane constantLane) := by
  simp [canonicalExpand]

theorem canonicalExpand_yRing_padding
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {EvaluationPoint : Type uEvaluationPoint}
    {ColumnPoint : Type uColumnPoint}
    {Digest : Type uDigest}
    {Advice : Type uAdvice}
    (context : VerifierContext Commitment PublicInput EvaluationPoint
      ColumnPoint Digest Advice)
    (payload : ActivePayload)
    (row : Fin
      Nightstream.Implementation.R1CS.PiCcsOutputDigest.Semantics.yRingRows)
    (lane : Fin paddedWidth)
    (padding :
      Nightstream.Implementation.R1CS.PiCcsOutputDigest.Semantics.activeWidth <=
        lane.val) :
    (canonicalExpand context payload).yRing row lane = K.zero := by
  exact zeroExtend_padding (payload.yRing row) lane padding

theorem canonicalExpand_yZcol_padding
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {EvaluationPoint : Type uEvaluationPoint}
    {ColumnPoint : Type uColumnPoint}
    {Digest : Type uDigest}
    {Advice : Type uAdvice}
    (context : VerifierContext Commitment PublicInput EvaluationPoint
      ColumnPoint Digest Advice)
    (payload : ActivePayload)
    (lane : Fin paddedWidth)
    (padding :
      Nightstream.Implementation.R1CS.PiCcsOutputDigest.Semantics.activeWidth <=
        lane.val) :
    (canonicalExpand context payload).yZcol lane = K.zero := by
  exact zeroExtend_padding payload.yZcol lane padding

/-- Field-by-field contract for canonical expansion. Keeping every omitted
legacy field explicit makes later Rust/R1CS refinement obligations auditable.
-/
structure CanonicalExpansion
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {EvaluationPoint : Type uEvaluationPoint}
    {ColumnPoint : Type uColumnPoint}
    {Digest : Type uDigest}
    {Advice : Type uAdvice}
    (context : VerifierContext Commitment PublicInput EvaluationPoint
      ColumnPoint Digest Advice)
    (payload : ActivePayload)
    (legacy : LegacyEnvelope Commitment PublicInput EvaluationPoint ColumnPoint
      Digest Advice) : Prop where
  commitment : legacy.commitment = context.commitment
  publicInputX : legacy.publicInputX = context.publicInputX
  evaluationPointR : legacy.evaluationPointR = context.evaluationPointR
  columnPointSCol : legacy.columnPointSCol = context.columnPointSCol
  yRing : legacy.yRing = fun row => zeroExtend (payload.yRing row)
  ct : legacy.ct = fun row => payload.yRing row constantLane
  auxOpenings : legacy.auxOpenings = []
  yZcol : legacy.yZcol = zeroExtend payload.yZcol
  inputLengthMIn : legacy.inputLengthMIn = context.inputLengthMIn
  foldDigest : legacy.foldDigest = context.foldDigest
  cStepCoords : legacy.cStepCoords = []
  uOffset : legacy.uOffset = 0
  uLen : legacy.uLen = 0
  adviceCommitment : legacy.adviceCommitment = context.adviceCommitment

theorem canonicalExpand_satisfies
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {EvaluationPoint : Type uEvaluationPoint}
    {ColumnPoint : Type uColumnPoint}
    {Digest : Type uDigest}
    {Advice : Type uAdvice}
    (context : VerifierContext Commitment PublicInput EvaluationPoint
      ColumnPoint Digest Advice)
    (payload : ActivePayload) :
    CanonicalExpansion context payload (canonicalExpand context payload) := by
  constructor <;> rfl

/-- The enumerated reconstruction obligations determine the full legacy
envelope; no independently carried `ct`, padding, sidecar, or offset remains.
-/
theorem eq_canonicalExpand_of_canonicalExpansion
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {EvaluationPoint : Type uEvaluationPoint}
    {ColumnPoint : Type uColumnPoint}
    {Digest : Type uDigest}
    {Advice : Type uAdvice}
    {context : VerifierContext Commitment PublicInput EvaluationPoint
      ColumnPoint Digest Advice}
    {payload : ActivePayload}
    {legacy : LegacyEnvelope Commitment PublicInput EvaluationPoint ColumnPoint
      Digest Advice}
    (canonical : CanonicalExpansion context payload legacy) :
    legacy = canonicalExpand context payload := by
  cases legacy
  cases canonical
  simp_all [canonicalExpand]

theorem canonicalExpansion_iff_eq
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {EvaluationPoint : Type uEvaluationPoint}
    {ColumnPoint : Type uColumnPoint}
    {Digest : Type uDigest}
    {Advice : Type uAdvice}
    {context : VerifierContext Commitment PublicInput EvaluationPoint
      ColumnPoint Digest Advice}
    {payload : ActivePayload}
    {legacy : LegacyEnvelope Commitment PublicInput EvaluationPoint ColumnPoint
      Digest Advice} :
    CanonicalExpansion context payload legacy ↔
      legacy = canonicalExpand context payload := by
  constructor
  · exact eq_canonicalExpand_of_canonicalExpansion
  · intro equal
    subst legacy
    exact canonicalExpand_satisfies context payload

/-- Open composition contract. The derivation relation is concrete, while a
proof that the source/transcript views are the accepted verifier-owned views
must be supplied by a later semantic/refinement theorem. -/
def SupportedProjection
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
    CanonicalExpansion context payload legacy

/-- Conditional uniqueness preserves the open context-authority premise
instead of silently treating carried context fields as derived. -/
theorem supportedProjection_unique
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
    {payload : ActivePayload}
    {left right : LegacyEnvelope Commitment PublicInput EvaluationPoint
      ColumnPoint Digest Advice}
    (leftSupported : SupportedProjection source transcript context payload left)
    (rightSupported : SupportedProjection source transcript context payload right) :
    left = right := by
  rw [eq_canonicalExpand_of_canonicalExpansion leftSupported.2,
    eq_canonicalExpand_of_canonicalExpansion rightSupported.2]

end Nightstream.Implementation.R1CS.PiCcsOutputDigest.Projection.FixedProfile
