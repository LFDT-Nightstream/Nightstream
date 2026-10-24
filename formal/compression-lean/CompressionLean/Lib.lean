import CompressionLean.Lib.Array
import CompressionLean.Lib.Combination
import CompressionLean.Lib.Cube
import CompressionLean.Lib.CubicField
import CompressionLean.Lib.Fraction
import CompressionLean.Lib.IdentityTest
import CompressionLean.Lib.List
import CompressionLean.Lib.LogUp
import CompressionLean.Lib.Nat
import CompressionLean.Lib.Roots
import CompressionLean.Lib.SumCheck
import CompressionLean.Lib.Univariate

/-!
`Lib/` owns the generic mathematics of the library: lemmas on lists, arrays and
numbers, tables on the Boolean cube and their multilinear extensions (`Cube`),
the bad sets of identity tests (`Roots`, `IdentityTest`), the round
polynomials of a sum-check (`SumCheck`, `Combination`), univariate forms
(`Univariate`), the logUp pole argument (`LogUp`), fraction trees
(`Fraction`) and the irreducibility of `x³ - x - 1` over Goldilocks
(`CubicField`).

It imports no other folder. It does not own a game (`Game/`), a field
instance (`Fields.lean`) or a layer's protocol.
-/
