# Native guard fusion

**Status:** Proposed

## Problem

The native WASM application has separate constraints for inactive operation
cases. Some rows share complementary or mutually exclusive guards.

## SuperNeo

SuperNeo v1.2 fixes the folding relation and fresh-witness norm. It does not
select an application's R1CS row construction.

## Decision

Combine exact complementary guards and disjoint subsets of a certified
one-hot family. Use source equations and checked packed seeds as certificates.
Keep the certificate rows and every original application record. Bind the
original record indices, exact source forms, and replacement rows into the
experimental application identity. Keep the four matrices and the profile in
[SPEC.md section 7.7](../SPEC.md#77-native-pieces).

The algebraic equivalence and certificate premises are in
`Export.Generic.GuardFusion`. The complete Rust compiler and slot bridge
remain outside that proof, as stated in
[the security model](../formal/nightstream-fprime/SECURITY_MODEL.md).
