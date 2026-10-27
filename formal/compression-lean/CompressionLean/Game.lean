import CompressionLean.Game.Compose
import CompressionLean.Game.Game
import CompressionLean.Game.Path
import CompressionLean.Game.Restoration
import CompressionLean.Game.SumCheck

/-!
`Game/` owns the probability model of the soundness proofs: a public-coin
verifier as a game against an adaptive prover (`Game`), the paths of a game
(`Path`), the state-restoration experiment of Fiat–Shamir (`Restoration`),
two games in one experiment (`Compose`), and the two generic games that the
layers share, a point of draws and the sum-check rounds (`SumCheck`).

It does not own a layer's verifier, a field or an error term.
-/
