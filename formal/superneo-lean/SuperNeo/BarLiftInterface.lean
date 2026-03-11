import SuperNeo.BarLift

/-!
Contract interface for `SuperNeo.BarLift`.

Spec: `specs/BarLift.spec.md`

Paper anchors:
- Definition 8 (Lifting the Transform), Section 5, lines 376-382.
-/

namespace SuperNeo

namespace BarLiftInterface

/-! ## Core Surfaces -/

/-- [Role: Theorem-Target] Curated re-export of `barLiftVector`. -/
abbrev barLiftVector := SuperNeo.barLiftVector

/-- [Role: Theorem-Target] Curated re-export of `barLiftMatrix`. -/
abbrev barLiftMatrix := SuperNeo.barLiftMatrix

/-! ## Shape Contracts -/

/-- [Role: Theorem-Target] Vector chunkability predicate for Definition-8 lifting. -/
abbrev barLiftChunkableVec := SuperNeo.barLiftChunkableVec

/-- [Role: Theorem-Target] Matrix chunkability predicate (row-wise). -/
abbrev barLiftChunkableMatrix := SuperNeo.barLiftChunkableMatrix

/-! ## Key Theorems -/

/-- [Role: Theorem-Target] Curated theorem surface `barLiftVector_eq`. -/
abbrev barLiftVector_eq := SuperNeo.barLiftVector_eq

/-- [Role: Theorem-Target] Curated theorem surface `barLiftMatrix_eq`. -/
abbrev barLiftMatrix_eq := SuperNeo.barLiftMatrix_eq

/-- [Role: Boundary] Bar-block identity boundary needed for identity-style closures. -/
abbrev barBlockIdentityAssumption := SuperNeo.barBlockIdentityAssumption

/-- [Role: Boundary] Shape-driven bar-block identity closure (`bar.size ≠ d`). -/
abbrev barBlockIdentityAssumption_of_bar_size_ne_d :=
  SuperNeo.barBlockIdentityAssumption_of_bar_size_ne_d

/-- [Role: Theorem-Target] Constructive linearity closure from bar-block identity. -/
abbrev barLiftLinearityAssumption_of_barBlockIdentity :=
  SuperNeo.barLiftLinearityAssumption_of_barBlockIdentity

/-- [Role: Theorem-Target] Constructive linearity closure for fallback shape branch. -/
abbrev barLiftLinearityAssumption_of_bar_size_ne_d :=
  SuperNeo.barLiftLinearityAssumption_of_bar_size_ne_d

/-- [Role: Theorem-Target] Curated theorem surface `barLiftVector_add`. -/
abbrev barLiftVector_add := SuperNeo.barLiftVector_add

/-- [Role: Theorem-Target] Curated theorem surface `barLiftVector_add_of_size_eq`. -/
abbrev barLiftVector_add_of_size_eq := SuperNeo.barLiftVector_add_of_size_eq

/-- [Role: Theorem-Target] Curated theorem surface `barLiftVector_scale`. -/
abbrev barLiftVector_scale := SuperNeo.barLiftVector_scale

/-- [Role: Theorem-Target] Curated theorem surface `barLiftVector_size`. -/
abbrev barLiftVector_size := SuperNeo.barLiftVector_size

/-- [Role: Theorem-Target] Curated theorem surface `barLiftMatrix_size`. -/
abbrev barLiftMatrix_size := SuperNeo.barLiftMatrix_size

/-- [Role: Theorem-Target] Chunkable vectors follow the embedding round-trip lifting path. -/
abbrev barLiftVector_eq_embedRoundTrip_of_chunkable :=
  SuperNeo.barLiftVector_eq_embedRoundTrip_of_chunkable

/-- [Role: Theorem-Target] Chunkable vectors follow explicit mapped bar-block round-trip path. -/
abbrev barLiftVector_eq_barBlockRoundTrip_of_chunkable :=
  SuperNeo.barLiftVector_eq_barBlockRoundTrip_of_chunkable

/-- [Role: Theorem-Target] Non-chunkable vectors take the identity fallback path. -/
abbrev barLiftVector_eq_self_of_not_chunkable :=
  SuperNeo.barLiftVector_eq_self_of_not_chunkable

/-! ## Boundary Surfaces -/

/-- [Role: Theorem-Target] Theorem-facing linearity boundary surface. -/
abbrev barLiftLinearityAssumption := SuperNeo.barLiftLinearityAssumption

/-- [Role: Theorem-Target] Check-facing linearity boundary surface. -/
abbrev barLiftLinearityCheckAssumption := SuperNeo.barLiftLinearityCheckAssumption

/-- [Role: Theorem-Target] Native closure of the theorem-facing linearity boundary. -/
theorem barLiftLinearityAssumption_native
  (bar : Array (Array F))
  (hLift : barLiftLinearityAssumption bar) :
  barLiftLinearityAssumption bar :=
  SuperNeo.barLiftLinearityAssumption_native bar hLift

/-- [Role: Theorem-Target] P9-threaded closure of the theorem-facing linearity boundary. -/
theorem barLiftLinearityAssumption_of_p9Embedding
  (bar : Array (Array F))
  (hP9 : p9EmbeddingAssumption)
  (hLift : barLiftLinearityAssumption bar) :
  barLiftLinearityAssumption bar :=
  SuperNeo.barLiftLinearityAssumption_of_p9Embedding bar hP9 hLift

/-- [Role: Theorem-Target] Closed P9 theorem-native linearity boundary. -/
theorem barLiftLinearityAssumption_of_p9Embedding_closed
  (bar : Array (Array F))
  (hLift : barLiftLinearityAssumption bar) :
  barLiftLinearityAssumption bar :=
  SuperNeo.barLiftLinearityAssumption_of_p9Embedding_closed bar hLift

/-- [Role: Theorem-Target] Conversion from theorem-facing to check-facing boundary. -/
theorem barLiftLinearityCheckAssumption_of_assumption
  {bar : Array (Array F)}
  (hAssm : barLiftLinearityAssumption bar) :
  barLiftLinearityCheckAssumption bar :=
  SuperNeo.barLiftLinearityCheckAssumption_of_assumption hAssm

/-- [Role: Theorem-Target] Conversion from check-facing to theorem-facing boundary. -/
theorem barLiftLinearityAssumption_of_checkAssumption
  {bar : Array (Array F)}
  (hCheck : barLiftLinearityCheckAssumption bar) :
  barLiftLinearityAssumption bar :=
  SuperNeo.barLiftLinearityAssumption_of_checkAssumption hCheck

/-- [Role: Theorem-Target] Equivalence between theorem and check boundaries. -/
theorem barLiftLinearityAssumption_iff_checkAssumption
  {bar : Array (Array F)} :
  barLiftLinearityAssumption bar ↔ barLiftLinearityCheckAssumption bar :=
  SuperNeo.barLiftLinearityAssumption_iff_checkAssumption

end BarLiftInterface

end SuperNeo
