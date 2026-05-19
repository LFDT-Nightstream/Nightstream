# DEC Digit Uniqueness

This spec records the digit-level fact needed by direct CCS `F'` accumulator
authorization.

## Negative Result

Signed low-norm base-2 decomposition is not unique. In integers:

```text
1 = 1 + 2*0
1 = -1 + 2*1
```

Both digit vectors satisfy `|digit| < 2`. Therefore the SuperNeo-style low-norm
window alone does not prove that private post-DEC children are unique.

Binary digits without a fixed length are also not unique, because leading zeroes
can be added without changing the recomposed value.

Fixed-length binary digits are not enough if recomposition is checked only
modulo a field modulus and the implementation does not prove the recomposed
integer is below that modulus. Modular wraparound can identify different
integer decompositions.

## Positive Result

If digits are restricted to the canonical binary set `{0,1}` and the digit
length is fixed, then same-length base-2 decompositions are unique. The theorem
applies both to a single digit list and column-wise to a coefficient table.

Binary recomposition is also range-bounded by its length: a binary digit list of
length `k` recomposes to a value below `2^k`. Therefore modular field equality
can be used only when the implementation also proves `2^k < q` or an equivalent
no-wrap condition.

## Implementation Consequence

The Rust circuit cannot rely on `Pi_DEC` recomposition plus a signed low-norm
bound alone to authorize hidden children from a bound `CE(B)` parent. It must
either enforce canonical digit construction/bitness, or add a stronger theorem
showing that the exact production constraints imply canonical children.

The Rust circuit must also enforce the exact child count/digit length `k`.
If Rust checks recomposition through field equality, it must prove the
recomposed integer is below the field modulus so that modular equality implies
integer equality.
