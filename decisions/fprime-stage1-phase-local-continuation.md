# F′ Stage 1 Phase-Local Continuation

**Status:** Accepted

## Problem

PiCCS cannot receive its final status until the canonical Stage 1 package,
application, key, and recursive fixed point exist. Requiring formal PiCCS,
PiRLC, or PiDEC closure before constructing those objects creates a circular
work-order dependency.

The earlier
[PiCCS phase-local order decision](piccs-phase-local-conformance-order.md)
authorizes PiRLC work. A later owner waiver authorizes PiDEC work. The
remaining Stage 1 phases need the same explicit work-order authority.

## SuperNeo

SuperNeo fixes the PiCCS, PiRLC, and PiDEC relations and their composition. It
does not fix Nightstream's implementation review order or status vocabulary.
Changing the work order does not change the paper relation, transcript,
profile, or proof bytes.

## Decision

Accumulator, application, terminal, and Stage 1 assembler work may proceed on
phase-local evidence while PiCCS, PiRLC, PiDEC, and the running-instance branch
remain formally status open.

No named phase may be called Conformance-closed until the final canonical
package identity exists and every required Lean, exact matrix, independent
assignment, complete nonzero parity, mutation, loader, identity, and exact-cut
external-review gate has run again on that identity.

This decision changes work order only. It does not weaken an acceptance
criterion, authorize Stage 2, authorize a proof backend, make a digest
authoritative, or make backend acceptance evidence for the Lean relation or
the Rust assignment.
