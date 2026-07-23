/-
Generated file: source-provenance payload; do not hand-edit.

Owns: the exact ordered payload stored in this generated module.

Does not own: decoding, assignment satisfaction, semantic authority, security
events, or permission to remove rows.

Emits constraints: no.

| Artifact leaf | Mathematical obligation | Authority class |
|---|---|---|
| generated payload | exact Rust-rendered list data and order | computed |
-/

import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.SourceSchema

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Generated.SourceColumns.RewriteSteps39

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized

set_option maxRecDepth 100000 in
def values : List RawRewriteStep :=
  [
    { emittedRow := 14944189, rewriteId := 133069, kind := .productSum, sourceRows := [{ start := 7961549, stop := 7961554 }], output := .source { constant := 0, terms := [{ column := 7716892, coefficient := 1 }] }, base := { constant := 0, terms := [] }, previous := none, factors := [{ left := { constant := 0, terms := [{ column := 7716887, coefficient := 1 }] }, right := { constant := 1, terms := [{ column := 7644454, coefficient := 1 }, { column := 7644319, coefficient := 1 }] }, coefficient := 1 }, { left := { constant := 0, terms := [{ column := 7716888, coefficient := 1 }] }, right := { constant := 0, terms := [{ column := 7644455, coefficient := 1 }, { column := 7644320, coefficient := 1 }] }, coefficient := 7 }] },
    { emittedRow := 14944190, rewriteId := 133069, kind := .productSum, sourceRows := [{ start := 7961549, stop := 7961554 }], output := .source { constant := 0, terms := [{ column := 7716893, coefficient := 1 }] }, base := { constant := 0, terms := [] }, previous := none, factors := [{ left := { constant := 0, terms := [{ column := 7716887, coefficient := 1 }] }, right := { constant := 0, terms := [{ column := 7644455, coefficient := 1 }, { column := 7644320, coefficient := 1 }] }, coefficient := 1 }, { left := { constant := 0, terms := [{ column := 7716888, coefficient := 1 }] }, right := { constant := 1, terms := [{ column := 7644454, coefficient := 1 }, { column := 7644319, coefficient := 1 }] }, coefficient := 1 }] }
  ]

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Generated.SourceColumns.RewriteSteps39
