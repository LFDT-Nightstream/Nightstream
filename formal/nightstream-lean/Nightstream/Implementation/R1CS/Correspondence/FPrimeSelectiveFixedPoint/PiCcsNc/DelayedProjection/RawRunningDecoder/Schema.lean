import Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Schema

/-!
Correspondence-layer facade for the neutral raw running-decoder schema.

Owns: no definitions; it exposes the neutral typed coordinate contract at the
stable correspondence path.

Does not own: generated columns, assignment decoding, protocol acceptance,
rows, costs, or authority.

Emits constraints: none.

| Stage path | Mathematical obligation | Authority class |
|---|---|---|
| `nifs.pi_ccs.nc.delayed.raw_decoder.schema_facade` | expose the neutral schema without reversing artifact dependencies | organizational |
-/
