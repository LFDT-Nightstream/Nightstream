import CompressionLean.Layer1.ClosedForms
import CompressionLean.Layer1.Cost
import CompressionLean.Layer1.Early
import CompressionLean.Layer1.EarlySound
import CompressionLean.Layer1.Encoding
import CompressionLean.Layer1.EvalK
import CompressionLean.Layer1.FiatShamir
import CompressionLean.Layer1.Gkr
import CompressionLean.Layer1.GkrLayer
import CompressionLean.Layer1.HonestFold
import CompressionLean.Layer1.HonestStage
import CompressionLean.Layer1.KeyStructure
import CompressionLean.Layer1.Late
import CompressionLean.Layer1.Linear
import CompressionLean.Layer1.Lookups
import CompressionLean.Layer1.Model
import CompressionLean.Layer1.Queries
import CompressionLean.Layer1.RingRows
import CompressionLean.Layer1.Soundness
import CompressionLean.Layer1.Stages
import CompressionLean.Layer1.Weights

/-!
`Layer1/` owns the soundness of the layer-1 verifier
(`crates/neo-spartan/src/verifier.rs`) as a game: the model of the key, the
statement and the setup (`Model`), each stage and what it forces (`Early`,
`Linear`, `HonestStage`, `Late`, `Stages`), the whole game and its accepting
paths (`Soundness`), its cost (`Cost`), and the closed
forms that the bridge reads (`ClosedForms`, `EvalK`, `KeyStructure`).

It does not own the replay (`Replay/`), the circuit (`Circuit/`), the link
between them (`Bridge/`) or the budget at Rust's split (`Final/Budget.lean`).
-/
