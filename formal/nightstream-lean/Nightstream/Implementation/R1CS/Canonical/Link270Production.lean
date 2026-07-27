import Nightstream.Implementation.R1CS.Canonical.Link270

/-!
Contract: the Phase-1b comparison surface between the frozen canonical
270-coordinate link and the production emitter.

Owns: the typed shape a production capture must present; the decision
procedure comparing a captured row against the canonical row for the same
coordinate; and the classification of a mismatch.

Does not own: the canonical encoding.  Nothing in this module may be imported
by `Link270`, and no captured value may be used to define a canonical one.
The dependency is one-directional by construction.

Status: the capture itself is not yet supplied.  This module fixes the
comparison *before* any production coefficient is read, so the canonical side
cannot be tuned to match.  `CaptureAgrees` is stated over an arbitrary capture.

The specific hypothesis Phase 1b exists to decide is `TailCopies` versus
`TailPinsZero` on coordinates 257..269.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.Link270Production

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.Link270

/-- What a production capture must present: one row per coordinate of the
selected range, together with the column identities it used. -/
structure Capture where
  row : Fin carrierWidth → Row
  sourceColumn : Fin carrierWidth → Nat
  destinationColumn : Fin carrierWidth → Nat

/-- The canonical row rewritten over a capture's own column identities.  This
is what the capture *should* have emitted for coordinate `i`, expressed in the
capture's allocation so that a difference is semantic rather than cosmetic. -/
def expectedRow (capture : Capture) (i : Fin carrierWidth) : Row where
  a := [(capture.destinationColumn i, 1),
        (capture.sourceColumn i, goldilocksP - 1)]
  b := [(0, 1)]
  c := []

/-- Coordinate-level agreement, modulo column allocation. -/
def AgreesAt (capture : Capture) (i : Fin carrierWidth) : Prop :=
  capture.row i = expectedRow capture i

instance (capture : Capture) (i : Fin carrierWidth) :
    Decidable (AgreesAt capture i) := by
  unfold AgreesAt; infer_instance

/-- Whole-range agreement. -/
def CaptureAgrees (capture : Capture) : Prop :=
  ∀ i, AgreesAt capture i

/-! ## The decisive hypothesis

A capture that pins the tail to zero emits, for `i` in `257..269`, a row
asserting `destination i = 0` rather than `destination i = source i`.  The two
are distinguishable by the source coefficient alone. -/

/-- The row a zero-pinning emitter would produce for coordinate `i`. -/
def zeroPinRow (capture : Capture) (i : Fin carrierWidth) : Row where
  a := [(capture.destinationColumn i, 1)]
  b := [(0, 1)]
  c := []

/-- The capture copies coordinate `i` (the correct behaviour). -/
def CopiesAt (capture : Capture) (i : Fin carrierWidth) : Prop :=
  capture.row i = expectedRow capture i

/-- The capture pins coordinate `i` to zero (the predicted defect). -/
def PinsZeroAt (capture : Capture) (i : Fin carrierWidth) : Prop :=
  capture.row i = zeroPinRow capture i

/-- Copying and zero-pinning are mutually exclusive, so the Phase-1b
measurement is decisive rather than ambiguous. -/
theorem copies_not_pinsZero (capture : Capture) (i : Fin carrierWidth) :
    CopiesAt capture i → ¬ PinsZeroAt capture i := by
  intro copies pins
  have equal : expectedRow capture i = zeroPinRow capture i := by
    rw [← copies, pins]
  have lengths := congrArg (fun row => row.a.length) equal
  simp [expectedRow, zeroPinRow] at lengths

/-- Tail coordinates, `257 .. 269`. -/
def IsTail (i : Fin carrierWidth) : Prop := legacyPublicWidth ≤ i.val

instance (i : Fin carrierWidth) : Decidable (IsTail i) := by
  unfold IsTail; infer_instance

/-- The two outcomes Phase 1b must decide between. -/
def TailCopies (capture : Capture) : Prop :=
  ∀ i, IsTail i → CopiesAt capture i

def TailPinsZero (capture : Capture) : Prop :=
  ∀ i, IsTail i → PinsZeroAt capture i

/-- There are exactly thirteen tail coordinates, so the measurement is
thirteen rows of coefficient inspection. -/
theorem tail_count :
    ((List.finRange carrierWidth).filter (fun i => decide (IsTail i))).length
      = 13 := by
  decide

/-- If the capture agrees everywhere then the tail copies; so a zero-pinning
tail is a genuine disagreement with the canonical encoding, not a presentation
difference. -/
theorem captureAgrees_tailCopies
    (capture : Capture) (agrees : CaptureAgrees capture) :
    TailCopies capture := fun i _ => agrees i

/-- A zero-pinning tail contradicts whole-range agreement, provided the tail is
inhabited. -/
theorem tailPinsZero_not_agrees
    (capture : Capture) (pins : TailPinsZero capture) :
    ¬ CaptureAgrees capture := by
  intro agrees
  have tail : IsTail firstTail := by
    unfold IsTail
    rw [firstTail_val]
    decide
  exact copies_not_pinsZero capture firstTail (agrees firstTail) (pins firstTail tail)

/-! ## Column alignment

`AgreesAt` compares row shape *given the capture's own claimed column
identities*.  That is not sufficient on its own: a capture could declare an
arbitrary column to be "source coordinate `i`" and still agree.  Alignment is
therefore a separate obligation, and it is what ties the comparison to the
authoritative coordinates. -/

/-- The capture's claimed columns are the authoritative ones. -/
structure ColumnsAligned (capture : Capture) : Prop where
  source : ∀ i, capture.sourceColumn i = Link270.sourceColumn i
  destination : ∀ i, capture.destinationColumn i = Link270.destinationColumn i

/-- Under alignment, agreement is equality with the canonical row itself, not
merely with a row of the same shape over unknown columns. -/
theorem agreesAt_of_aligned
    (capture : Capture) (aligned : ColumnsAligned capture) (i : Fin carrierWidth)
    (agrees : AgreesAt capture i) :
    capture.row i = coordinateRow i := by
  rw [agrees]
  simp [expectedRow, coordinateRow, aligned.source i, aligned.destination i]

/-- **Full equality.**  An aligned, agreeing capture is exactly the canonical
encoding — all 270 rows, not only the tail. -/
theorem capture_eq_canonical
    (capture : Capture) (aligned : ColumnsAligned capture)
    (agrees : CaptureAgrees capture) :
    (List.finRange carrierWidth).map capture.row = canonicalRows := by
  refine List.map_congr_left ?_
  intro i _
  exact agreesAt_of_aligned capture aligned i (agrees i)

/-! ## Exhaustive classification

Mutual exclusivity is not exhaustiveness: a capture may be neither a copy nor a
zero-pin.  The measurement must therefore be a three-way classification. -/

inductive Classification where
  | exactCopies
  | tailPinsZero
  | other
deriving DecidableEq, Repr

/-- The classification a capture actually falls into. -/
noncomputable def classify (capture : Capture) : Classification :=
  open Classical in
  if CaptureAgrees capture then .exactCopies
  else if TailPinsZero capture then .tailPinsZero
  else .other

/-- **Exhaustiveness.**  Every capture receives exactly one classification, and
the classification determines which property holds. -/
theorem classify_exhaustive (capture : Capture) :
    (classify capture = .exactCopies ∧ CaptureAgrees capture) ∨
      (classify capture = .tailPinsZero ∧ TailPinsZero capture) ∨
      (classify capture = .other ∧
        ¬ CaptureAgrees capture ∧ ¬ TailPinsZero capture) := by
  classical
  unfold classify
  by_cases agrees : CaptureAgrees capture
  · exact Or.inl ⟨by simp [agrees], agrees⟩
  · by_cases pins : TailPinsZero capture
    · exact Or.inr (Or.inl ⟨by simp [agrees, pins], pins⟩)
    · exact Or.inr (Or.inr ⟨by simp [agrees, pins], agrees, pins⟩)

/-- The two informative classifications are distinct: a capture cannot be both,
so `tailPinsZero` is a genuine defect verdict rather than a weaker reading of
agreement. -/
theorem exactCopies_ne_tailPinsZero (capture : Capture)
    (agrees : CaptureAgrees capture) : ¬ TailPinsZero capture :=
  fun pins => tailPinsZero_not_agrees capture pins agrees

end Nightstream.Implementation.R1CS.Canonical.Link270Production
