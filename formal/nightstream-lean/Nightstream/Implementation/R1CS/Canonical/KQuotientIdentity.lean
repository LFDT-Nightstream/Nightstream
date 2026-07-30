import Nightstream.Implementation.R1CS.Canonical.KQuotient
import Nightstream.Implementation.R1CS.Canonical.KHornerSupport
import Nightstream.Implementation.R1CS.Canonical.KHornerHonest
import Nightstream.Implementation.R1CS.Canonical.KEquality
import Nightstream.Implementation.Lowering.Typed.Cost

/-!
Contract: the emitted row program for one projected quotient identity.

Owns: the row program, its derived row count, its column allocation with
collision-freedom, conservation, a derived `Typed.Cost`, and soundness — both
of the projected-product atom and of the whole check. Honest completeness is
proved in the responsibility-split companion `KQuotientIdentityHonest`.

Does **not** own: the decoder from a frozen `ProjectionTrace` to the
`Carried` lists consumed here, NIFS call framing, transcript binding, or the
semantic root-event refinement. `KTraceDecoder` and `KTraceProgram` provide
the projection-level adapters without changing this row program.

## What is being encoded

The frozen check (`ProjectionProgram.ProjectionTrace.identity`) is a
coefficient identity between fixed-width vectors,

```text
Σᵢ ρᵢ · xᵢ  =  q · Φ₈₁ + out
```

tested at one challenge. An R1CS encoding never materialises those degree-106
vectors. It uses the two algebraic facts the `K` tower proves:

- `KPolyEval.polyEval_polyMul` — evaluation is multiplicative, so
  `eval(ρᵢ · xᵢ) = eval(ρᵢ) · eval(xᵢ)`;
- `KQuotient.polyEval_ringKMul_quotientForm` — the quotient form evaluates to
  `eval(out) + eval(q) · eval(Φ₈₁)`.

Neither needs any condition on the challenge. So the emitted program checks

```text
Σᵢ eval(ρᵢ) · eval(xᵢ)  =  eval(out) + eval(q) · eval(Φ₈₁)
```

with every `eval` a Horner projection.

## Why `Φ₈₁` costs rows at all

Its coefficients are constants, but `Φ₈₁(β)` is not — it is a degree-54
polynomial in the challenge, and Horner spends one `K` multiplication per step.
162 rows is what the *straightforward* encoding costs. An addition chain would
compute `β²⁷` in about seven multiplications and `Φ₈₁(β) = (β²⁷)² + β²⁷ + 1` in
one more, so roughly 24 rows. That is a real optimisation and it is deliberately
not taken here: the number below is derived from the program actually emitted,
and a cheaper program has to be emitted before a cheaper number may be recorded.
-/

set_option autoImplicit false
-- The conservation induction rewrites through 321-column atom blocks;
-- the default depth is not enough for the nested layout arithmetic.
set_option maxRecDepth 8000

namespace Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KHorner

/-! ## The projected-product atom

Two Horner evaluations over disjoint frame blocks, then one `K` multiplication
of their carried results. -/

/-- The carried output of a `K` multiplication: linear in the frame's three
columns, so it emits no row of its own. -/
def productCarried (frame : Frame) : Carried :=
  ⟨outLow frame, outHigh frame⟩

/-- **A `K` multiplication computes the product of its operands' values.**  The
two coordinate lemmas, packaged as one statement about `Pair`s. -/
theorem mulRows_sound
    (z : Nat → Nat) (left right : Carried) (frame : Frame)
    (satisfied : Satisfies (KMul.rows left right frame) z) :
    carriedValue z (productCarried frame)
      = mulPair (carriedValue z left) (carriedValue z right) := by
  unfold carriedValue productCarried mulPair
  simp only [Pair.mk.injEq]
  exact ⟨outLow_sound z left right frame satisfied,
    outHigh_sound z left right frame satisfied⟩

/-- **One projected product.** -/
def productRows (beta : Carried) (leftBase rightBase : Nat) (frame : Frame)
    (left right : List Carried) : List Row :=
  hornerRows beta (KFrames.frameAt leftBase) left 0
    ++ hornerRows beta (KFrames.frameAt rightBase) right 0
    ++ KMul.rows (hornerCarried beta (KFrames.frameAt leftBase) left 0)
        (hornerCarried beta (KFrames.frameAt rightBase) right 0) frame

/-- **The atom's derived row count.**  Three per multiplication on each side,
plus three for the product itself. -/
theorem productRows_length
    (beta : Carried) (leftBase rightBase : Nat) (frame : Frame)
    (left right : List Carried) :
    (productRows beta leftBase rightBase frame left right).length
      = 3 * (left.length - 1) + 3 * (right.length - 1) + 3 := by
  unfold productRows
  rw [List.length_append, List.length_append, hornerRows_length,
    hornerRows_length, KMul.rows_length]

/-- At the production degree of 53 an atom costs 321 rows. -/
theorem productRows_length_production
    (beta : Carried) (leftBase rightBase : Nat) (frame : Frame)
    (left right : List Carried)
    (leftSized : left.length = 54) (rightSized : right.length = 54) :
    (productRows beta leftBase rightBase frame left right).length = 321 := by
  rw [productRows_length, leftSized, rightSized]

/-- **The atom is sound.**  Satisfaction forces the frame's carried output to be
the product of the two projections. -/
theorem productRows_sound
    (z : Nat → Nat) (beta : Carried) (leftBase rightBase : Nat) (frame : Frame)
    (left right : List Carried)
    (satisfied : Satisfies (productRows beta leftBase rightBase frame left right) z) :
    carriedValue z (productCarried frame)
      = mulPair (hornerValue (carriedValue z beta) (left.map (carriedValue z)))
          (hornerValue (carriedValue z beta) (right.map (carriedValue z))) := by
  have leftSat : Satisfies (hornerRows beta (KFrames.frameAt leftBase) left 0) z :=
    fun row member => satisfied row
      (List.mem_append.2 (Or.inl (List.mem_append.2 (Or.inl member))))
  have rightSat :
      Satisfies (hornerRows beta (KFrames.frameAt rightBase) right 0) z :=
    fun row member => satisfied row
      (List.mem_append.2 (Or.inl (List.mem_append.2 (Or.inr member))))
  have mulSat : Satisfies
      (KMul.rows (hornerCarried beta (KFrames.frameAt leftBase) left 0)
        (hornerCarried beta (KFrames.frameAt rightBase) right 0) frame) z :=
    fun row member => satisfied row (List.mem_append.2 (Or.inr member))
  rw [mulRows_sound z _ _ frame mulSat,
    hornerRows_sound z beta (KFrames.frameAt leftBase) left 0 leftSat,
    hornerRows_sound z beta (KFrames.frameAt rightBase) right 0 rightSat]

/-! ## Block widths

Each Horner block allocates three columns per multiplication, so a vector of
`n` coefficients takes `3(n − 1)` columns. The three widths below are the only
place the production vector lengths enter, and each is derived from
`KFrames.frameColumns_length` rather than declared. -/

/-- Columns one degree-53 projection allocates: 53 multiplications. -/
def projectionWidth : Nat := 159

/-- Columns the 53-coefficient quotient projection allocates. -/
def quotientWidth : Nat := 156

/-- Columns the 55-coefficient modulus evaluation allocates. -/
def modulusWidth : Nat := 162

theorem projectionWidth_derived (base : Nat) :
    (KFrames.frameColumns base (54 - 1)).length = projectionWidth :=
  KFrames.frameColumns_length base (54 - 1)

theorem quotientWidth_derived (base : Nat) :
    (KFrames.frameColumns base (53 - 1)).length = quotientWidth :=
  KFrames.frameColumns_length base (53 - 1)

theorem modulusWidth_derived (base : Nat) :
    (KFrames.frameColumns base (55 - 1)).length = modulusWidth :=
  KFrames.frameColumns_length base (55 - 1)

/-! ## The left-hand side

One atom per input pair, laid out sequentially. The sum of the atoms' outputs is
a concatenation of combinations, so it emits no row — which is why the
left-hand side costs exactly `321` per pair and nothing for the summation. -/

/-- Columns one atom takes: two projections and one product frame. -/
def atomWidth : Nat := 2 * projectionWidth + 3

/-- **The left-hand side's rows**, laid out sequentially from `base`. -/
def pairsRows (beta : Carried) :
    List (List Carried × List Carried) → Nat → List Row
  | [], _ => []
  | (left, right) :: rest, base =>
      productRows beta base (base + projectionWidth)
          (KFrames.frameAt (base + 2 * projectionWidth) 0) left right
        ++ pairsRows beta rest (base + atomWidth)

/-- Two carried values added: concatenated combinations, so no row. -/
def concatCarried (left right : Carried) : Carried :=
  ⟨left.low ++ right.low, left.high ++ right.high⟩

/-- **Concatenating combinations adds their values.**  This is exactly why the
summation over input pairs is free. -/
theorem carriedValue_concat (z : Nat → Nat) (left right : Carried) :
    carriedValue z (concatCarried left right)
      = addPair (carriedValue z left) (carriedValue z right) := by
  unfold carriedValue concatCarried addPair
  simp only [Pair.mk.injEq]
  exact ⟨lcEval_append z _ _, lcEval_append z _ _⟩

/-- **The left-hand side's carried value.**  Concatenation: free. -/
def pairsCarried : List (List Carried × List Carried) → Nat → Carried
  | [], _ => ⟨[], []⟩
  | _ :: rest, base =>
      concatCarried (productCarried (KFrames.frameAt (base + 2 * projectionWidth) 0))
        (pairsCarried rest (base + atomWidth))

/-- **The left-hand side's derived row count.**  321 per pair, and nothing for
the summation. -/
theorem pairsRows_length (beta : Carried) :
    ∀ (pairs : List (List Carried × List Carried)) (base : Nat),
      (∀ pair ∈ pairs, pair.1.length = 54 ∧ pair.2.length = 54) →
      (pairsRows beta pairs base).length = 321 * pairs.length
  | [], _, _ => rfl
  | (left, right) :: rest, base, sized => by
      have head : left.length = 54 ∧ right.length = 54 := sized (left, right) (by simp)
      have tail := pairsRows_length beta rest (base + atomWidth)
        (fun pair member => sized pair (List.mem_cons_of_mem _ member))
      show (productRows beta base (base + projectionWidth)
          (KFrames.frameAt (base + 2 * projectionWidth) 0) left right
        ++ pairsRows beta rest (base + atomWidth)).length = _
      rw [List.length_append, tail,
        productRows_length_production _ _ _ _ _ _ head.1 head.2,
        List.length_cons]
      omega

/-! ## The whole check

Left-hand side, then the output, quotient and modulus projections, one
multiplication for `q(β) · Φ₈₁(β)`, and one `K`-equality — which is two rows,
not one. -/

/-- Where the output projection starts: after every atom. -/
def outBase (base pairCount : Nat) : Nat := base + atomWidth * pairCount

/-- Where the quotient projection starts. -/
def quotientBase (base pairCount : Nat) : Nat :=
  outBase base pairCount + projectionWidth

/-- Where the modulus evaluation starts. -/
def modulusBase (base pairCount : Nat) : Nat :=
  quotientBase base pairCount + quotientWidth

/-- The single frame for `q(beta) * Phi81(beta)`. -/
def productFrameAt (base pairCount : Nat) : Frame :=
  KFrames.frameAt (modulusBase base pairCount + modulusWidth) 0

/-- **The emitted quotient-identity program.** -/
def identityRows (beta : Carried) (base : Nat)
    (pairs : List (List Carried × List Carried))
    (output quotient modulus : List Carried) : List Row :=
  pairsRows beta pairs base
    ++ hornerRows beta (KFrames.frameAt (outBase base pairs.length)) output 0
    ++ hornerRows beta (KFrames.frameAt (quotientBase base pairs.length)) quotient 0
    ++ hornerRows beta (KFrames.frameAt (modulusBase base pairs.length)) modulus 0
    ++ KMul.rows
        (hornerCarried beta (KFrames.frameAt (quotientBase base pairs.length))
          quotient 0)
        (hornerCarried beta (KFrames.frameAt (modulusBase base pairs.length))
          modulus 0)
        (productFrameAt base pairs.length)
    ++ KEquality.rows (pairsCarried pairs base)
        (concatCarried
          (hornerCarried beta (KFrames.frameAt (outBase base pairs.length))
            output 0)
          (productCarried (productFrameAt base pairs.length)))

/-- **The derived row count of one quotient identity.**

Every term comes from the emitted program: `321` per input pair, `159` for the
output projection, `156` for the quotient, `162` for the modulus, `3` for their
product, and `2` — not 1 — for the `K`-equality. -/
theorem identityRows_length (beta : Carried) (base : Nat)
    (pairs : List (List Carried × List Carried))
    (output quotient modulus : List Carried)
    (sized : ∀ pair ∈ pairs, pair.1.length = 54 ∧ pair.2.length = 54)
    (outputSized : output.length = 54) (quotientSized : quotient.length = 53)
    (modulusSized : modulus.length = 55) :
    (identityRows beta base pairs output quotient modulus).length
      = 321 * pairs.length + 482 := by
  unfold identityRows
  rw [List.length_append, List.length_append, List.length_append,
    List.length_append, List.length_append,
    pairsRows_length beta pairs base sized,
    hornerRows_length, hornerRows_length, hornerRows_length,
    KMul.rows_length, KEquality.rows_length,
    outputSized, quotientSized, modulusSized]

/-- At the production arity of two inputs, one quotient identity costs 1124
rows.

This is the number that replaces the withdrawn `combineEquationCost 2 = 803`.
The difference is exactly `321` — the quotient projection (156), the modulus
evaluation (162), and their product (3): the three terms the impossible
root assumption was hiding. -/
theorem identityRows_length_production (beta : Carried) (base : Nat)
    (pairs : List (List Carried × List Carried))
    (output quotient modulus : List Carried)
    (sized : ∀ pair ∈ pairs, pair.1.length = 54 ∧ pair.2.length = 54)
    (arity : pairs.length = 2)
    (outputSized : output.length = 54) (quotientSized : quotient.length = 53)
    (modulusSized : modulus.length = 55) :
    (identityRows beta base pairs output quotient modulus).length = 1124 := by
  rw [identityRows_length beta base pairs output quotient modulus sized
    outputSized quotientSized modulusSized, arity]

/-! ## Equation-level soundness

The atom proves one product. Composing the atoms needs the summation, and that
is where `carriedValue_concat` earns its place: the left-hand side is a
concatenation of combinations, so its value is the `Pair` sum of the atom
outputs and no row is involved. -/

/-- The reference sum of the projected products. -/
def pairSum : List Pair → Pair
  | [] => ⟨0, 0⟩
  | value :: rest => addPair value (pairSum rest)

/-- The value a coefficient carrier projects to at the challenge. -/
def projected (z : Nat → Nat) (beta : Carried) (coefficients : List Carried) :
    Pair :=
  hornerValue (carriedValue z beta) (coefficients.map (carriedValue z))

/-- **The left-hand side computes the sum of the projected products.** -/
theorem pairsRows_sound (z : Nat → Nat) (beta : Carried) :
    ∀ (pairs : List (List Carried × List Carried)) (base : Nat),
      Satisfies (pairsRows beta pairs base) z →
      carriedValue z (pairsCarried pairs base)
        = pairSum (pairs.map (fun pair =>
            mulPair (projected z beta pair.1) (projected z beta pair.2)))
  | [], _, _ => rfl
  | (left, right) :: rest, base, satisfied => by
      have headSat : Satisfies (productRows beta base (base + projectionWidth)
          (KFrames.frameAt (base + 2 * projectionWidth) 0) left right) z :=
        fun row member => satisfied row (List.mem_append.2 (Or.inl member))
      have tailSat : Satisfies (pairsRows beta rest (base + atomWidth)) z :=
        fun row member => satisfied row (List.mem_append.2 (Or.inr member))
      show carriedValue z (concatCarried
          (productCarried (KFrames.frameAt (base + 2 * projectionWidth) 0))
          (pairsCarried rest (base + atomWidth))) = _
      rw [carriedValue_concat,
        productRows_sound z beta base (base + projectionWidth) _ left right
          headSat,
        pairsRows_sound z beta rest (base + atomWidth) tailSat]
      rfl

/-- **The whole check is sound.**

Satisfaction forces the projected quotient identity at the challenge:

```text
Σᵢ eval(ρᵢ) · eval(xᵢ) = eval(out) + eval(q) · eval(Φ₈₁)
```

This is the equation `KQuotient.polyEval_ringKMul_quotientForm` says the frozen
ring multiplication satisfies. Reaching `ProjectionCheck.Accepted` from here is
a decoding step, not a further algebraic one — and it is not written. -/
theorem identityRows_sound
    (z : Nat → Nat) (beta : Carried) (base : Nat)
    (pairs : List (List Carried × List Carried))
    (output quotient modulus : List Carried) (constantWire : z 0 = 1)
    (satisfied :
      Satisfies (identityRows beta base pairs output quotient modulus) z) :
    pairSum (pairs.map (fun pair =>
        mulPair (projected z beta pair.1) (projected z beta pair.2)))
      = addPair (projected z beta output)
          (mulPair (projected z beta quotient) (projected z beta modulus)) := by
  have pairsSat : Satisfies (pairsRows beta pairs base) z :=
    fun row member => satisfied row (List.mem_append.2 (Or.inl
      (List.mem_append.2 (Or.inl (List.mem_append.2 (Or.inl
        (List.mem_append.2 (Or.inl (List.mem_append.2 (Or.inl member))))))))))
  have outSat : Satisfies
      (hornerRows beta (KFrames.frameAt (outBase base pairs.length)) output 0) z :=
    fun row member => satisfied row (List.mem_append.2 (Or.inl
      (List.mem_append.2 (Or.inl (List.mem_append.2 (Or.inl
        (List.mem_append.2 (Or.inl (List.mem_append.2 (Or.inr member))))))))))
  have quotientSat : Satisfies
      (hornerRows beta (KFrames.frameAt (quotientBase base pairs.length))
        quotient 0) z :=
    fun row member => satisfied row (List.mem_append.2 (Or.inl
      (List.mem_append.2 (Or.inl (List.mem_append.2 (Or.inl
        (List.mem_append.2 (Or.inr member))))))))
  have modulusSat : Satisfies
      (hornerRows beta (KFrames.frameAt (modulusBase base pairs.length))
        modulus 0) z :=
    fun row member => satisfied row (List.mem_append.2 (Or.inl
      (List.mem_append.2 (Or.inl (List.mem_append.2 (Or.inr member))))))
  have mulSat : Satisfies
      (KMul.rows
        (hornerCarried beta (KFrames.frameAt (quotientBase base pairs.length))
          quotient 0)
        (hornerCarried beta (KFrames.frameAt (modulusBase base pairs.length))
          modulus 0)
        (productFrameAt base pairs.length)) z :=
    fun row member => satisfied row (List.mem_append.2 (Or.inl
      (List.mem_append.2 (Or.inr member))))
  have equalitySat : Satisfies
      (KEquality.rows (pairsCarried pairs base)
        (concatCarried
          (hornerCarried beta (KFrames.frameAt (outBase base pairs.length))
            output 0)
          (productCarried (productFrameAt base pairs.length)))) z :=
    fun row member => satisfied row (List.mem_append.2 (Or.inr member))
  rcases KEquality.rows_sound z _ _ constantWire equalitySat with ⟨lowEq, highEq⟩
  have equal : carriedValue z (pairsCarried pairs base)
      = carriedValue z (concatCarried
          (hornerCarried beta (KFrames.frameAt (outBase base pairs.length))
            output 0)
          (productCarried (productFrameAt base pairs.length))) := by
    unfold carriedValue
    simp only [Pair.mk.injEq]
    exact ⟨lowEq, highEq⟩
  rw [← pairsRows_sound z beta pairs base pairsSat, equal, carriedValue_concat,
    hornerRows_sound z beta (KFrames.frameAt (outBase base pairs.length))
      output 0 outSat,
    mulRows_sound z _ _ (productFrameAt base pairs.length) mulSat,
    hornerRows_sound z beta (KFrames.frameAt (quotientBase base pairs.length))
      quotient 0 quotientSat,
    hornerRows_sound z beta (KFrames.frameAt (modulusBase base pairs.length))
      modulus 0 modulusSat]
  rfl

/-! ## Column ownership

Every block this program allocates is a `KFrames.frameColumns` run, and those
are contiguous by construction. The whole allocation is therefore itself one
contiguous run, which is what makes ownership arithmetic rather than
combinatorial: `frameColumns_mem_iff` turns membership into an interval and
`omega` closes the rest.

This is the first place `KHorner`'s deferred frame disjointness is actually
discharged rather than passed on. -/

/-- Multiplications one degree-53 projection performs. -/
def projectionFrames : Nat := 53

/-- Multiplications the 53-coefficient quotient projection performs. -/
def quotientFrames : Nat := 52

/-- Multiplications the 55-coefficient modulus evaluation performs. -/
def modulusFrames : Nat := 54

/-- Multiplications one atom performs: two projections and one product. -/
def atomFrames : Nat := 2 * projectionFrames + 1

/-- Multiplications the whole check performs. -/
def identityFrames (pairCount : Nat) : Nat :=
  atomFrames * pairCount
    + (projectionFrames + quotientFrames + modulusFrames + 1)

theorem atomFrames_eq : atomFrames = 107 := rfl

theorem atomWidth_eq : atomWidth = 3 * atomFrames := rfl

theorem identityFrames_eq (pairCount : Nat) :
    identityFrames pairCount = 107 * pairCount + 160 := by
  unfold identityFrames atomFrames projectionFrames quotientFrames modulusFrames
  omega

/-- **Everything the program allocates.**  One contiguous run. -/
def identityColumns (base pairCount : Nat) : List Nat :=
  KFrames.frameColumns base (identityFrames pairCount)

/-- **The derived column count.**  Three per multiplication, and the row count
exceeds it by exactly the two equality rows, which allocate nothing. -/
theorem identityColumns_length (base pairCount : Nat) :
    (identityColumns base pairCount).length = 321 * pairCount + 480 := by
  unfold identityColumns
  rw [KFrames.frameColumns_length, identityFrames_eq]
  omega

/-- **No column is allocated twice.** -/
theorem identityColumns_nodup (base pairCount : Nat) :
    (identityColumns base pairCount).Nodup :=
  KFrames.frameColumns_nodup _ _

/-! ### The two interval facts every block obligation reduces to -/

/-- Separated blocks share no column. -/
theorem frameColumns_disjoint
    (leftBase leftCount rightBase rightCount : Nat)
    (separated : leftBase + 3 * leftCount ≤ rightBase) (column : Nat)
    (inLeft : column ∈ KFrames.frameColumns leftBase leftCount) :
    column ∉ KFrames.frameColumns rightBase rightCount := by
  rw [KFrames.frameColumns_mem_iff] at inLeft
  intro inRight
  rw [KFrames.frameColumns_mem_iff] at inRight
  omega

/-- A block inside another block's interval is inside that block. -/
theorem frameColumns_subset
    (outerBase outerCount innerBase innerCount : Nat)
    (lower : outerBase ≤ innerBase)
    (upper : innerBase + 3 * innerCount ≤ outerBase + 3 * outerCount)
    (column : Nat) (inner : column ∈ KFrames.frameColumns innerBase innerCount) :
    column ∈ KFrames.frameColumns outerBase outerCount := by
  rw [KFrames.frameColumns_mem_iff] at inner ⊢
  omega

/-! ### Atoms

Atom `index` occupies `[base + 321·index, base + 321·(index+1))`, and inside it
the left projection, right projection and product frame occupy `[0,159)`,
`[159,318)` and `[318,321)`. -/

/-- **Distinct atoms never share a column.** -/
theorem atoms_disjoint (base index other : Nat) (distinct : index ≠ other)
    (column : Nat)
    (inIndex : column ∈ KFrames.frameColumns (base + 321 * index) atomFrames) :
    column ∉ KFrames.frameColumns (base + 321 * other) atomFrames := by
  rw [KFrames.frameColumns_mem_iff, atomFrames_eq] at inIndex
  intro inOther
  rw [KFrames.frameColumns_mem_iff, atomFrames_eq] at inOther
  rcases Nat.lt_or_ge index other with less | more
  · omega
  · have strict : other < index := by omega
    omega

/-- **An atom's three blocks never share a column**, and each lies inside the
atom. Stated as the three pairwise separations, which is what the interval
argument needs. -/
theorem atom_blocks_separated (atomBase : Nat) :
    atomBase + 3 * projectionFrames ≤ atomBase + projectionWidth
      ∧ atomBase + projectionWidth + 3 * projectionFrames
          ≤ atomBase + 2 * projectionWidth
      ∧ atomBase + 2 * projectionWidth + 3 * 1
          ≤ atomBase + 3 * atomFrames := by
  simp only [atomFrames, projectionFrames, projectionWidth]
  omega

/-- **Every atom lies inside the program's allocation.** -/
theorem atom_inside (base pairCount index : Nat) (below : index < pairCount)
    (column : Nat)
    (member : column ∈ KFrames.frameColumns (base + 321 * index) atomFrames) :
    column ∈ identityColumns base pairCount := by
  refine frameColumns_subset base (identityFrames pairCount) _ _
    (by omega) ?_ column member
  rw [identityFrames_eq, atomFrames_eq]
  omega

/-! ### The tail blocks

After the atoms come the output projection, the quotient projection, the modulus
evaluation and one product frame — 53, 52, 54 and 1 multiplications, which is
the 160 in `identityFrames`. -/

/-- **The four tail blocks are separated from each other and from the atoms.**
Every gap is exactly the preceding block's width, so the allocation is gapless
as well as collision-free. -/
theorem tail_blocks_separated (base pairCount : Nat) :
    base + 3 * (atomFrames * pairCount)
        = base + atomWidth * pairCount
      ∧ base + atomWidth * pairCount + 3 * projectionFrames
          = base + atomWidth * pairCount + projectionWidth
      ∧ base + atomWidth * pairCount + projectionWidth + 3 * quotientFrames
          = base + atomWidth * pairCount + projectionWidth + quotientWidth
      ∧ base + atomWidth * pairCount + projectionWidth + quotientWidth
              + 3 * modulusFrames
          = base + atomWidth * pairCount + projectionWidth + quotientWidth
              + modulusWidth := by
  unfold atomFrames atomWidth projectionFrames projectionWidth quotientFrames
    quotientWidth modulusFrames modulusWidth
  omega

/-- **Every tail block lies inside the program's allocation**, and the last one
ends exactly at its end — so the allocation is exhausted, not merely
contained. -/
theorem tail_inside (base pairCount : Nat) :
    base + atomWidth * pairCount + projectionWidth + quotientWidth
        + modulusWidth + 3 * 1
      = base + 3 * identityFrames pairCount := by
  unfold atomWidth projectionWidth quotientWidth modulusWidth
  rw [identityFrames_eq]
  omega

/-! ## Conservation

No emitted row reaches outside the allocation plus the declared shared reads.
The shared reads are exactly the challenge and the coefficient carriers — data
this program consumes but does not allocate.

This is where `KHornerSupport.FrameOfRun`'s **upper** bound is load-bearing. The
witness used to say only "some frame at or after this step", which places a
column with no ceiling and so cannot be shown to lie inside a finite block. The
bound was added for exactly this obligation; the freshness argument in
`KHornerHonest` never needed it. -/

/-- Columns the program allocates, as an interval.  The same set
`identityColumns` lists, in the form the arithmetic wants. -/
def Allocated (base pairCount column : Nat) : Prop :=
  base ≤ column ∧ column < base + (321 * pairCount + 480)

theorem allocated_iff (base pairCount column : Nat) :
    Allocated base pairCount column
      ↔ column ∈ identityColumns base pairCount := by
  unfold Allocated identityColumns
  rw [KFrames.frameColumns_mem_iff, identityFrames_eq]
  omega

/-- **A run's frames lie in its own block.**  The bridge from
`KHornerSupport.FrameOfRun` to an interval. -/
theorem frameOfRun_interval
    (blockBase : Nat) (coefficients : List Carried) (column : Nat)
    (inRun : KHornerSupport.FrameOfRun (KFrames.frameAt blockBase)
      coefficients 0 column) :
    blockBase ≤ column
      ∧ column < blockBase + 3 * (coefficients.length - 1) := by
  rcases inRun with ⟨later, _, bounded, slot⟩
  rcases slot with rfl | rfl | rfl <;>
    · simp only [KFrames.frameAt, KFrames.frameColumn, KFrames.columnsPerFrame]
      omega

/-- What the program reads but does not allocate. -/
def SharedRead (beta : Carried) (coefficients : List Carried)
    (column : Nat) : Prop :=
  (Mentions beta.low column ∨ Mentions beta.high column)
    ∨ KHornerSupport.CoefficientColumn coefficients column

/-- **One Horner block is conservative.**  Every column it mentions is the
challenge, a coefficient, or inside its own block. -/
theorem hornerBlock_conservation
    (beta : Carried) (blockBase : Nat) (coefficients : List Carried)
    (row : Row)
    (member : row ∈ hornerRows beta (KFrames.frameAt blockBase) coefficients 0)
    (column : Nat)
    (mentioned : Mentions row.a column ∨ Mentions row.b column
      ∨ Mentions row.c column) :
    (blockBase ≤ column ∧ column < blockBase + 3 * (coefficients.length - 1))
      ∨ SharedRead beta coefficients column := by
  rcases KHornerSupport.hornerRows_mentions beta (KFrames.frameAt blockBase)
      coefficients 0 row member column mentioned with
    inBeta | inCoefficient | inRun
  · exact Or.inr (Or.inl inBeta)
  · exact Or.inr (Or.inr inCoefficient)
  · exact Or.inl (frameOfRun_interval blockBase coefficients column inRun)

/-- **A carried Horner result is conservative.**  Needed because the product
rows read the two evaluations' carried values, not their rows. -/
theorem hornerCarried_conservation
    (beta : Carried) (blockBase : Nat) (coefficients : List Carried)
    (column : Nat)
    (mentioned :
      Mentions (hornerCarried beta (KFrames.frameAt blockBase) coefficients 0).low
          column
        ∨ Mentions (hornerCarried beta (KFrames.frameAt blockBase) coefficients 0).high
            column) :
    (blockBase ≤ column ∧ column < blockBase + 3 * (coefficients.length - 1))
      ∨ KHornerSupport.CoefficientColumn coefficients column := by
  rcases KHornerSupport.hornerCarried_mentions beta (KFrames.frameAt blockBase)
      coefficients 0 column mentioned with inCoefficient | inRun
  · exact Or.inr inCoefficient
  · exact Or.inl (frameOfRun_interval blockBase coefficients column inRun)

/-- **The atom is conservative.** -/
theorem productRows_conservation
    (beta : Carried) (atomBase : Nat) (left right : List Carried)
    (leftSized : left.length = 54) (rightSized : right.length = 54)
    (row : Row)
    (member : row ∈ productRows beta atomBase (atomBase + projectionWidth)
      (KFrames.frameAt (atomBase + 2 * projectionWidth) 0) left right)
    (column : Nat)
    (mentioned : Mentions row.a column ∨ Mentions row.b column
      ∨ Mentions row.c column) :
    (atomBase ≤ column ∧ column < atomBase + atomWidth)
      ∨ SharedRead beta left column ∨ SharedRead beta right column := by
  have widen : ∀ bound : Nat, column < atomBase + 3 * (54 - 1) + bound →
      bound ≤ 3 → column < atomBase + atomWidth := by
    simp only [atomWidth, projectionWidth]
    omega
  unfold productRows at member
  simp only [List.mem_append] at member
  rcases member with (inLeft | inRight) | inMul
  · rcases hornerBlock_conservation beta atomBase left row inLeft column
      mentioned with interval | shared
    · exact Or.inl ⟨interval.1, widen 0 (by rw [leftSized] at interval; omega)
        (by omega)⟩
    · exact Or.inr (Or.inl shared)
  · rcases hornerBlock_conservation beta (atomBase + projectionWidth) right row
      inRight column mentioned with interval | shared
    · refine Or.inl ⟨by omega, ?_⟩
      have upper := interval.2
      rw [rightSized] at upper
      simp only [atomWidth, projectionWidth] at upper ⊢
      omega
    · exact Or.inr (Or.inr shared)
  · rcases KMulOwnership.rows_conservation _ _ _ row inMul column mentioned with
      operand | frameCol
    · have leftCase : ∀ (side : Bool),
          (if side then
            Mentions (hornerCarried beta (KFrames.frameAt atomBase) left 0).low column
           else
            Mentions (hornerCarried beta (KFrames.frameAt atomBase) left 0).high column) →
          (atomBase ≤ column ∧ column < atomBase + atomWidth)
            ∨ SharedRead beta left column ∨ SharedRead beta right column := by
        intro side hit
        rcases hornerCarried_conservation beta atomBase left column
          (by cases side with
              | true => exact Or.inl (by simpa using hit)
              | false => exact Or.inr (by simpa using hit)) with
          interval | inCoefficient
        · exact Or.inl ⟨interval.1, widen 0
            (by rw [leftSized] at interval; omega) (by omega)⟩
        · exact Or.inr (Or.inl (Or.inr inCoefficient))
      have rightCase : ∀ (side : Bool),
          (if side then
            Mentions (hornerCarried beta
              (KFrames.frameAt (atomBase + projectionWidth)) right 0).low column
           else
            Mentions (hornerCarried beta
              (KFrames.frameAt (atomBase + projectionWidth)) right 0).high column) →
          (atomBase ≤ column ∧ column < atomBase + atomWidth)
            ∨ SharedRead beta left column ∨ SharedRead beta right column := by
        intro side hit
        rcases hornerCarried_conservation beta (atomBase + projectionWidth) right
          column
          (by cases side with
              | true => exact Or.inl (by simpa using hit)
              | false => exact Or.inr (by simpa using hit)) with
          interval | inCoefficient
        · refine Or.inl ⟨by omega, ?_⟩
          have upper := interval.2
          rw [rightSized] at upper
          simp only [atomWidth, projectionWidth] at upper ⊢
          omega
        · exact Or.inr (Or.inr (Or.inr inCoefficient))
      rcases operand with l0 | l1 | r0 | r1
      · exact leftCase true (by simpa using l0)
      · exact leftCase false (by simpa using l1)
      · exact rightCase true (by simpa using r0)
      · exact rightCase false (by simpa using r1)
    · refine Or.inl ?_
      rcases frameCol with rfl | rfl | rfl <;>
        · simp only [KFrames.frameAt, KFrames.frameColumn,
            KFrames.columnsPerFrame, atomWidth, projectionWidth]
          omega

/-- **The left-hand side is conservative.** -/
theorem pairsRows_conservation (beta : Carried) :
    ∀ (pairs : List (List Carried × List Carried)) (base : Nat),
      (∀ pair ∈ pairs, pair.1.length = 54 ∧ pair.2.length = 54) →
      ∀ (row : Row), row ∈ pairsRows beta pairs base → ∀ column,
        (Mentions row.a column ∨ Mentions row.b column
          ∨ Mentions row.c column) →
        (base ≤ column ∧ column < base + atomWidth * pairs.length)
          ∨ ∃ pair ∈ pairs,
              SharedRead beta pair.1 column ∨ SharedRead beta pair.2 column
  | [], _, _, _, member, _, _ => by simp [pairsRows] at member
  | (left, right) :: rest, base, sized, row, member, column, mentioned => by
      have head : left.length = 54 ∧ right.length = 54 :=
        sized (left, right) (by simp)
      have widthEq : atomWidth = 321 := rfl
      rw [show pairsRows beta ((left, right) :: rest) base
          = productRows beta base (base + projectionWidth)
              (KFrames.frameAt (base + 2 * projectionWidth) 0) left right
            ++ pairsRows beta rest (base + atomWidth) from rfl,
        List.mem_append] at member
      rcases member with inHead | inTail
      · rcases productRows_conservation beta base left right head.1 head.2 row
          inHead column mentioned with interval | shared
        · refine Or.inl ⟨interval.1, ?_⟩
          have upper := interval.2
          rw [widthEq] at upper ⊢
          rw [List.length_cons]
          omega
        · exact Or.inr ⟨(left, right), by simp, shared⟩
      · rcases pairsRows_conservation beta rest (base + atomWidth)
          (fun pair pairMember => sized pair (List.mem_cons_of_mem _ pairMember))
          row inTail column mentioned with interval | ⟨pair, pairMember, shared⟩
        · refine Or.inl ⟨by omega, ?_⟩
          have upper := interval.2
          rw [widthEq] at upper ⊢
          rw [List.length_cons]
          omega
        · exact Or.inr ⟨pair, List.mem_cons_of_mem _ pairMember, shared⟩

/-- **A product frame's carried output mentions only that frame.** -/
theorem productCarried_mentions (frame : Frame) (column : Nat)
    (mentioned : Mentions (productCarried frame).low column
      ∨ Mentions (productCarried frame).high column) :
    column = frame.lowLow ∨ column = frame.highHigh ∨ column = frame.cross := by
  simp only [productCarried, outLow, outHigh, Mentions, List.map_cons,
    List.map_nil, List.mem_cons, List.not_mem_nil, or_false] at mentioned
  rcases mentioned with (h | h) | (h | h | h)
  · exact Or.inl h
  · exact Or.inr (Or.inl h)
  · exact Or.inr (Or.inr h)
  · exact Or.inl h
  · exact Or.inr (Or.inl h)

/-- **The left-hand side's carried value stays inside the atoms' block.**  It is
a concatenation of product-frame outputs and nothing else. -/
theorem pairsCarried_mentions :
    ∀ (pairs : List (List Carried × List Carried)) (base column : Nat),
      (Mentions (pairsCarried pairs base).low column
        ∨ Mentions (pairsCarried pairs base).high column) →
      base ≤ column ∧ column < base + atomWidth * pairs.length
  | [], _, _, mentioned => by
      simp only [pairsCarried, Mentions, List.map_nil, List.not_mem_nil,
        or_self] at mentioned
  | (left, right) :: rest, base, column, mentioned => by
      have widthEq : atomWidth = 321 := rfl
      rw [show pairsCarried ((left, right) :: rest) base
          = concatCarried
              (productCarried (KFrames.frameAt (base + 2 * projectionWidth) 0))
              (pairsCarried rest (base + atomWidth)) from rfl] at mentioned
      simp only [concatCarried, Mentions, List.map_append,
        List.mem_append] at mentioned
      have head : column = (KFrames.frameAt (base + 2 * projectionWidth) 0).lowLow
          ∨ column = (KFrames.frameAt (base + 2 * projectionWidth) 0).highHigh
          ∨ column = (KFrames.frameAt (base + 2 * projectionWidth) 0).cross →
          base ≤ column ∧ column < base + atomWidth * ((left, right) :: rest).length := by
        intro slot
        rw [widthEq, List.length_cons]
        rcases slot with rfl | rfl | rfl <;>
          · simp only [KFrames.frameAt, KFrames.frameColumn,
              KFrames.columnsPerFrame, projectionWidth]
            omega
      have tail : base + atomWidth ≤ column
          ∧ column < base + atomWidth + atomWidth * rest.length →
          base ≤ column ∧ column < base + atomWidth * ((left, right) :: rest).length := by
        intro interval
        rw [widthEq, List.length_cons]
        rw [widthEq] at interval
        omega
      rcases mentioned with (inHead | inTail) | (inHead | inTail)
      · exact head (productCarried_mentions _ column (Or.inl inHead))
      · exact tail (pairsCarried_mentions rest (base + atomWidth) column
          (Or.inl inTail))
      · exact head (productCarried_mentions _ column (Or.inr inHead))
      · exact tail (pairsCarried_mentions rest (base + atomWidth) column
          (Or.inr inTail))

/-- The three block bases, in the closed form the interval arithmetic wants. -/
theorem layout_bases (base pairCount : Nat) :
    outBase base pairCount = base + 321 * pairCount
      ∧ quotientBase base pairCount = base + 321 * pairCount + 159
      ∧ modulusBase base pairCount = base + 321 * pairCount + 159 + 156 :=
  ⟨rfl, rfl, rfl⟩

/-- **The whole program is conservative.**  Every column any emitted row
mentions is the constant wire, a column the program allocates, or one of the
declared shared reads — the challenge and the coefficient carriers.

The constant wire appears because `KEquality`'s rows write a literal one; it
allocates nothing and is shared by every program in the system. -/
theorem identityRows_conservation
    (beta : Carried) (base : Nat)
    (pairs : List (List Carried × List Carried))
    (output quotient modulus : List Carried)
    (sized : ∀ pair ∈ pairs, pair.1.length = 54 ∧ pair.2.length = 54)
    (outputSized : output.length = 54) (quotientSized : quotient.length = 53)
    (modulusSized : modulus.length = 55)
    (row : Row)
    (member : row ∈ identityRows beta base pairs output quotient modulus)
    (column : Nat)
    (mentioned : Mentions row.a column ∨ Mentions row.b column
      ∨ Mentions row.c column) :
    column = 0
      ∨ Allocated base pairs.length column
      ∨ (∃ pair ∈ pairs,
          SharedRead beta pair.1 column ∨ SharedRead beta pair.2 column)
      ∨ SharedRead beta output column
      ∨ SharedRead beta quotient column
      ∨ SharedRead beta modulus column := by
  have widthEq : atomWidth = 321 := rfl
  obtain ⟨outEq, quotientEq, modulusEq⟩ := layout_bases base pairs.length
  simp only [identityRows, List.mem_append] at member
  rcases member with ((((inA | inB) | inC) | inD) | inE) | inF
  · rcases pairsRows_conservation beta pairs base sized row inA column mentioned
      with interval | shared
    · refine Or.inr (Or.inl ⟨interval.1, ?_⟩)
      have upper := interval.2
      rw [widthEq] at upper
      omega
    · exact Or.inr (Or.inr (Or.inl shared))
  · rcases hornerBlock_conservation beta (outBase base pairs.length) output row
      inB column mentioned with interval | shared
    · refine Or.inr (Or.inl ⟨by omega, ?_⟩)
      have upper := interval.2
      rw [outputSized, outEq] at upper
      omega
    · exact Or.inr (Or.inr (Or.inr (Or.inl shared)))
  · rcases hornerBlock_conservation beta (quotientBase base pairs.length)
      quotient row inC column mentioned with interval | shared
    · refine Or.inr (Or.inl ⟨by omega, ?_⟩)
      have upper := interval.2
      rw [quotientSized, quotientEq] at upper
      omega
    · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl shared))))
  · rcases hornerBlock_conservation beta (modulusBase base pairs.length)
      modulus row inD column mentioned with interval | shared
    · refine Or.inr (Or.inl ⟨by omega, ?_⟩)
      have upper := interval.2
      rw [modulusSized, modulusEq] at upper
      omega
    · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr shared))))
  · rcases KMulOwnership.rows_conservation _ _ _ row inE column mentioned with
      operand | frameCol
    · rcases operand with q0 | q1 | m0 | m1
      · rcases hornerCarried_conservation beta (quotientBase base pairs.length)
          quotient column (Or.inl q0) with interval | inCoefficient
        · refine Or.inr (Or.inl ⟨by omega, ?_⟩)
          have upper := interval.2
          rw [quotientSized, quotientEq] at upper
          omega
        · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl (Or.inr inCoefficient)))))
      · rcases hornerCarried_conservation beta (quotientBase base pairs.length)
          quotient column (Or.inr q1) with interval | inCoefficient
        · refine Or.inr (Or.inl ⟨by omega, ?_⟩)
          have upper := interval.2
          rw [quotientSized, quotientEq] at upper
          omega
        · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl (Or.inr inCoefficient)))))
      · rcases hornerCarried_conservation beta (modulusBase base pairs.length)
          modulus column (Or.inl m0) with interval | inCoefficient
        · refine Or.inr (Or.inl ⟨by omega, ?_⟩)
          have upper := interval.2
          rw [modulusSized, modulusEq] at upper
          omega
        · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr inCoefficient)))))
      · rcases hornerCarried_conservation beta (modulusBase base pairs.length)
          modulus column (Or.inr m1) with interval | inCoefficient
        · refine Or.inr (Or.inl ⟨by omega, ?_⟩)
          have upper := interval.2
          rw [modulusSized, modulusEq] at upper
          omega
        · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr inCoefficient)))))
    · refine Or.inr (Or.inl ?_)
      unfold Allocated
      rcases frameCol with rfl | rfl | rfl <;>
        · simp only [productFrameAt, KFrames.frameAt, KFrames.frameColumn,
            KFrames.columnsPerFrame, modulusWidth, modulusBase, quotientBase,
            outBase, atomWidth, projectionWidth, quotientWidth]
          omega
  · rcases KEquality.rows_conservation _ _ row inF column mentioned with
      wire | l0 | l1 | r0 | r1
    · exact Or.inl wire
    · refine Or.inr (Or.inl ?_)
      have interval := pairsCarried_mentions pairs base column (Or.inl l0)
      rw [widthEq] at interval
      unfold Allocated
      omega
    · refine Or.inr (Or.inl ?_)
      have interval := pairsCarried_mentions pairs base column (Or.inr l1)
      rw [widthEq] at interval
      unfold Allocated
      omega
    · simp only [concatCarried, Mentions, List.map_append,
        List.mem_append] at r0
      rcases r0 with inOut | inProduct
      · rcases hornerCarried_conservation beta (outBase base pairs.length) output
          column (Or.inl inOut) with interval | inCoefficient
        · refine Or.inr (Or.inl ⟨by omega, ?_⟩)
          have upper := interval.2
          rw [outputSized, outEq] at upper
          omega
        · exact Or.inr (Or.inr (Or.inr (Or.inl (Or.inr inCoefficient))))
      · refine Or.inr (Or.inl ?_)
        unfold Allocated
        rcases productCarried_mentions _ column (Or.inl inProduct) with rfl | rfl | rfl <;>
          · simp only [productFrameAt, KFrames.frameAt, KFrames.frameColumn,
            KFrames.columnsPerFrame, modulusWidth, modulusBase, quotientBase,
            outBase, atomWidth, projectionWidth, quotientWidth]
            omega
    · simp only [concatCarried, Mentions, List.map_append,
        List.mem_append] at r1
      rcases r1 with inOut | inProduct
      · rcases hornerCarried_conservation beta (outBase base pairs.length) output
          column (Or.inr inOut) with interval | inCoefficient
        · refine Or.inr (Or.inl ⟨by omega, ?_⟩)
          have upper := interval.2
          rw [outputSized, outEq] at upper
          omega
        · exact Or.inr (Or.inr (Or.inr (Or.inl (Or.inr inCoefficient))))
      · refine Or.inr (Or.inl ?_)
        unfold Allocated
        rcases productCarried_mentions _ column (Or.inr inProduct) with rfl | rfl | rfl <;>
          · simp only [productFrameAt, KFrames.frameAt, KFrames.frameColumn,
            KFrames.columnsPerFrame, modulusWidth, modulusBase, quotientBase,
            outBase, atomWidth, projectionWidth, quotientWidth]
            omega

/-! ## The cost tuple

Both nonzero components are receipts: the rows from `identityRows_length`, the
auxiliary columns from `identityColumns_length`. Neither is declared. -/

/-- **The recipe's cost**, in the project's `Typed.Cost`.

`committedColumns` and `publicColumns` are zero, and that is a statement about
ownership rather than a gap. This recipe allocates only the intermediate
products of its Horner ladders and multiplications. The vectors it *reads* — the
challenge, the two operand vectors per pair, the output, the quotient and the
modulus — are preallocated inputs, referenced through the `Carried` lists the
caller supplies. Counting them here would double-count them against whatever
recipe allocates them, which is prompt section 4.4's trap exactly. -/
def identityCost (pairCount : Nat) : Lowering.Typed.Cost where
  recurringRows := 321 * pairCount + 482
  committedColumns := 0
  publicColumns := 0
  auxiliaryColumns := 321 * pairCount + 480

/-- **The row component is a receipt.** -/
theorem identityCost_rows (beta : Carried) (base : Nat)
    (pairs : List (List Carried × List Carried))
    (output quotient modulus : List Carried)
    (sized : ∀ pair ∈ pairs, pair.1.length = 54 ∧ pair.2.length = 54)
    (outputSized : output.length = 54) (quotientSized : quotient.length = 53)
    (modulusSized : modulus.length = 55) :
    (identityRows beta base pairs output quotient modulus).length
      = (identityCost pairs.length).recurringRows :=
  identityRows_length beta base pairs output quotient modulus sized outputSized
    quotientSized modulusSized

/-- **The auxiliary-column component is a receipt.** -/
theorem identityCost_columns (base pairCount : Nat) :
    (identityColumns base pairCount).length
      = (identityCost pairCount).auxiliaryColumns :=
  identityColumns_length base pairCount

/-- The two receipts differ by exactly two: the `K`-equality emits two rows and
allocates nothing. -/
theorem identityCost_gap (pairCount : Nat) :
    (identityCost pairCount).recurringRows
      = (identityCost pairCount).auxiliaryColumns + 2 := by
  simp only [identityCost]

end Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity
