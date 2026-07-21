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
    { emittedRow := 14946857, rewriteId := 144247, kind := .productSum, sourceRows := [{ start := 8010324, stop := 8010329 }], output := .source { constant := 0, terms := [{ column := 7765657, coefficient := 1 }] }, base := { constant := 0, terms := [] }, previous := none, factors := [{ left := { constant := 0, terms := [{ column := 7765652, coefficient := 1 }] }, right := { constant := 1, terms := [{ column := 7693219, coefficient := 1 }, { column := 7693084, coefficient := 1 }] }, coefficient := 1 }, { left := { constant := 0, terms := [{ column := 7765653, coefficient := 1 }] }, right := { constant := 0, terms := [{ column := 7693220, coefficient := 1 }, { column := 7693085, coefficient := 1 }] }, coefficient := 7 }] },
    { emittedRow := 14946858, rewriteId := 144247, kind := .productSum, sourceRows := [{ start := 8010324, stop := 8010329 }], output := .source { constant := 0, terms := [{ column := 7765658, coefficient := 1 }] }, base := { constant := 0, terms := [] }, previous := none, factors := [{ left := { constant := 0, terms := [{ column := 7765652, coefficient := 1 }] }, right := { constant := 0, terms := [{ column := 7693220, coefficient := 1 }, { column := 7693085, coefficient := 1 }] }, coefficient := 1 }, { left := { constant := 0, terms := [{ column := 7765653, coefficient := 1 }] }, right := { constant := 1, terms := [{ column := 7693219, coefficient := 1 }, { column := 7693084, coefficient := 1 }] }, coefficient := 1 }] }
  ]

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Generated.SourceColumns.RewriteSteps39
