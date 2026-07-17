import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Exact.Carrier
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Exact.Schedule
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Exact.Refinement
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Exact.CompleteSchedule
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Exact.HonestProver

/-!
Stable facade for the exact typed `Pi_CCS` message language, canonical
FE-to-NC sub-schedule, complete binding-to-catch-up schedule, and honest
semantic construction.

Owns: only the public import boundary.

Does not own: authority of outer binding fields, paper-joint/Split-NC
equivalence, Fiat--Shamir probability, Rust/R1CS refinement, constraints,
costs, or removals.

| Child | Mathematical ownership | Emits constraints? |
|---|---|---|
| `Exact.Carrier` | exact FE/NC counts, widths, FE-initial binding, and lossless codec | no |
| `Exact.Schedule` | exact FE execution followed directly by exact NC execution | no |
| `Exact.Refinement` | exact carrier serialization equals the typed FE/NC checker adapters | no |
| `Exact.CompleteSchedule` | binding, concrete coins, exact FE/NC replay, and catch-up | no |
| `Exact.HonestProver` | paper obligations construct an accepted, source-bound exact execution | no |
-/
