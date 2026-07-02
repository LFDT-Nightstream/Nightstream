import SuperNeo.SumCheck.Core
import SuperNeo.SumCheck.Paper
import SuperNeo.SumCheck.Extension
import SuperNeo.SumCheck.ExtensionPaper
import SuperNeo.SumCheck.General
import SuperNeo.SumCheck.PrefixSoundnessEndpoint

/-! Section 4, Definition 6 (sum-check) barrel.

`Core`/`Paper` own the protocol scaffold and the Definition-6 paper closure;
`Extension`/`ExtensionPaper` mirror that surface over `KExt`; `Defs` through
`General`/`PrefixSoundnessEndpoint` own the round-by-round soundness
development and its endpoints. -/
