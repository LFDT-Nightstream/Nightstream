import SuperNeo.EqPoly

/-!
Contract interface for `SuperNeo.EqPoly`.

Paper anchors:
- `./formal/superneo-lean/SuperNeo.pdf.md`, Section 4 (Preliminaries), line 274:
  eq(x,y) = Π_i (x_i·y_i + (1-x_i)·(1-y_i))
- `./formal/superneo-lean/SuperNeo.pdf.md`, Lemma 6, line 737:
  eq(X,Z)·Q(X) zero-sum characterization
-/

namespace SuperNeo

namespace EqPolyInterface

/-! ## Core Definitions -/

/-- [Role: Theorem-Target] Boolean predicate: `x = 0 ∨ x = 1`. -/
abbrev IsBit := SuperNeo.IsBit

/-- [Role: Theorem-Target] All entries satisfy `IsBit`. -/
abbrev IsBitVec := SuperNeo.IsBitVec

/-- [Role: Theorem-Target] Single-coordinate factor: `x*y + (1-x)*(1-y)`. -/
abbrev eqTerm := SuperNeo.eqTerm

/-- [Role: Theorem-Target] Product equality polynomial: `Π_i eqTerm x[i] y[i]`; returns 0 on size mismatch. -/
abbrev eqPoly := SuperNeo.eqPoly

/-- [Role: Theorem-Target] Bit-vector embedding: natural number mask → `{0,1}^ℓ` as field elements. -/
abbrev bitsToFArray := SuperNeo.bitsToFArray

/-! ## Key Theorems -/

/-- [Role: Theorem-Target] Size mismatch ⟹ `eqPoly x y = 0`. -/
theorem eqPoly_eq_zero_of_size_ne
  {x y : Array F}
  (hNe : x.size ≠ y.size) :
  eqPoly x y = 0 :=
  SuperNeo.eqPoly_eq_zero_of_size_ne hNe

/-- [Role: Theorem-Target] Bit-level Kronecker: `eqTerm x y = if x = y then 1 else 0` for bits. -/
theorem eqTerm_eq_delta_of_isBit
  {x y : F}
  (hx : IsBit x)
  (hy : IsBit y) :
  eqTerm x y = (if x = y then 1 else 0) :=
  SuperNeo.eqTerm_eq_delta_of_isBit hx hy

/-- [Role: Theorem-Target] Boolean-cube selector: `eqPoly x y = if x = y then 1 else 0` for equal-length bit-vectors. -/
theorem eqPoly_eq_delta_of_isBitVec
  {x y : Array F}
  (hSize : x.size = y.size)
  (hx : IsBitVec x)
  (hy : IsBitVec y) :
  eqPoly x y = (if x = y then 1 else 0) :=
  SuperNeo.eqPoly_eq_delta_of_isBitVec hSize hx hy

/-! ## Package Wrapper (closed) -/

/-- [Role: Theorem-Target] Bundled Prop carrier for selector behavior; compatibility wrapper for assumption-threaded consumers. -/
abbrev eqPolyAssumption := SuperNeo.eqPolyAssumption

/-- [Role: Theorem-Target] Discharges `eqPolyAssumption` from the selector theorem. -/
abbrev eqPolyAssumption_holds := SuperNeo.eqPolyAssumption_holds

end EqPolyInterface

end SuperNeo
