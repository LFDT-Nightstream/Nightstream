# Packed Boolean Source Companions

**Status:** Accepted

## Problem

The fresh witness already has signed-unit coordinates. A separate quadratic
equation for each Boolean source repeats a range restriction that can use
this existing norm.

## SuperNeo

SuperNeo v1.2 supplies the fresh-witness norm relation. It does not select
the application's scalar encoding or remove its Boolean equations.

## Decision

Use explicit source companions and ordinary packed rows as specified in
[SPEC.md section 7.9](../SPEC.md#79-packed-boolean-checks). Keep the existing
slot decoder, four matrices and folding profile. The map is part of the
native application identity.

This choice uses the actual fresh norm and the exact one-coordinate source
layout. Native block outputs and terminal source checks do not have the
required representation and retain their checks. The deterministic
implication and honest completion are proved in
`Export.Generic.PackedBoolean` and
`Export.BoundedDepth.PackedBooleanCompleteness`.
