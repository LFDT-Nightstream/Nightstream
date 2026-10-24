import CompressionLean.Bridge.Early
import CompressionLean.Bridge.EvalK
import CompressionLean.Bridge.Field
import CompressionLean.Bridge.Formulas
import CompressionLean.Bridge.Gkr
import CompressionLean.Bridge.Honest
import CompressionLean.Bridge.Late
import CompressionLean.Bridge.Linear
import CompressionLean.Bridge.Main
import CompressionLean.Bridge.Openings
import CompressionLean.Bridge.Relation
import CompressionLean.Bridge.Ring
import CompressionLean.Bridge.Run
import CompressionLean.Bridge.Setup

/-!
`Bridge/` owns the link from the replay to the layer-1 model: replay words
and extension values as field values (`Field`), the facts of a successful
replay run (`Run`), and, stage by stage, that an accepted replay is a path
of the model's game (`Main`, `replay_accepts_implies_path`).

It does not own the replay (`Ports/Replay/`), the model (`Layer1/`), or the
model's values of the replay's that the claims name (`Spec/`).
-/
