/-!
Wire schema for the fixed-point selective compiler's public-coordinate decoder.

Owns: untrusted proof-free records naming one final encoded column and its
compiler source class.

Does not own: record validity, assignment semantics, private coordinates,
relation satisfaction, commitment alignment, or row removal.

Emits constraints: no.

| Wire field | Rust source | Semantic status |
|---|---|---|
| `column` | prepared selective layout | untrusted until exact checking |
| `source` | validated encoder owner | untrusted until exact checking |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Wire

inductive RawSource where
  | constantOne
  | sourceField (field : Nat)
  | fixedZero
deriving DecidableEq, Repr

structure RawCoordinate where
  schemaVersion : Nat
  column : Nat
  source : RawSource
deriving DecidableEq, Repr

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Wire
