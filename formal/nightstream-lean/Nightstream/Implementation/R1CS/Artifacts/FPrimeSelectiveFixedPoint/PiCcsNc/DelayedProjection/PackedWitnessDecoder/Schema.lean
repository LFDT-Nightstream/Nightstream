import Nightstream.SuperNeo.Concrete.Parameters

/-!
Schema for the compact production full-`Z` decoder artifact.

Owns: one proof-free record describing whether a Boolean lane reads the
corresponding packed-witness row or is verifier-computed zero.

Does not own: generated dimensions, witness values, commitment binding,
combined-NC acceptance, transcript scheduling, rows, costs, or row removal.

Emits constraints: none; artifact schema only.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `pi_ccs_nc.full_z_decoder.schema.lane_source` | distinguish one direct witness lane from one computed-zero lane | organizational schema |
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessDecoder

/-- `some lane` denotes a direct packed-witness row; `none` denotes a
computed-zero virtual lane. -/
structure LaneSourceRecord where
  booleanLane : Nat
  witnessLane : Option Nat
deriving DecidableEq, Repr, Inhabited

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessDecoder
