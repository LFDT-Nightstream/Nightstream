# Nightstream semantic requirement graph

This file is generated from `src/requirements/*.jsonl`. The rule text
stays in the ordered modules under `src/normative/`.

Requirements: **104**. Direct dependency edges: **170**.

Longest declared path:

```text
SN-FND-FIELD -> SN-FND-RING -> SN-FND-DIMENSIONS -> SN-EMBED-COEFFICIENTS -> SN-REL-CCS -> SN-PICCS-POLYNOMIAL -> SN-PICCS-TARGET -> SN-PICCS-EXECUTION -> SN-PICCS-OUTPUT -> SN-PICCS-CHARACTERIZATION -> SN-PICCS-IDENTITY -> NS-PICCS-VARIANT -> NS-PICCS-COINS -> NS-PICCS-SUMCHECK -> NS-PICCS-TERMINAL -> NS-PICCS-NORM-BINDING -> NS-PICCS-NO-COLUMN -> NS-AUTH-CLAIM -> NS-AUTH-DERIVED -> NS-RUST-EVIDENCE-ORIGIN -> NS-RUST-EVIDENCE-CONTENT -> NS-CIRCUIT-MANIFEST -> NS-CIRCUIT-SOUNDNESS -> NS-DECIDER-CORRESPONDENCE -> NS-SEC-REDUCTION -> NS-SEC-COMPOSITION -> NS-RELEASE-PRODUCTION
```

## Assembly operations

| Rule | Operation | Replaces |
|---|---|---|
| `SN-FND-FIELD` | adopt | — |
| `SN-FND-RING` | adopt | — |
| `SN-FND-DIMENSIONS` | adopt | — |
| `SN-NORM-CENTERED` | adopt | — |
| `SN-NORM-BOUNDS` | adopt | — |
| `SN-SPLIT-ABSTRACT` | adopt | — |
| `SN-EMBED-COEFFICIENTS` | adopt | — |
| `SN-EMBED-MODULE-ACTION` | adopt | — |
| `SN-REL-STRUCTURE` | adopt | — |
| `SN-REL-CCS` | adopt | — |
| `SN-REL-CE` | adopt | — |
| `SN-GLOBAL-STRONG-SET` | adopt | — |
| `SN-GLOBAL-NORM-GUARD` | adopt | — |
| `SN-GLOBAL-COMMITMENT` | adopt | — |
| `SN-STRONGSET-DIVISOR` | adopt | — |
| `SN-STRONGSET-DIFFERENCE` | adopt | — |
| `SN-STRONGSET-EXPANSION` | adopt | — |
| `SN-MSIS-PARAMETERS` | adopt | — |
| `SN-RED-STAGE` | adopt | — |
| `SN-RED-KNOWLEDGE` | adopt | — |
| `SN-RED-SEQUENTIAL` | adopt | — |
| `SN-RED-RELATIONS` | adopt | — |
| `SN-RED-PROJECTION` | adopt | — |
| `SN-RED-STRONG-CONDITIONS` | adopt | — |
| `SN-RED-WEAK-CONDITIONS` | adopt | — |
| `SN-SUMCHECK-CLAIM` | adopt | — |
| `SN-SUMCHECK-ROUNDS` | adopt | — |
| `SN-SUMCHECK-SOUNDNESS` | adopt | — |
| `SN-PICCS-POLYNOMIAL` | adopt | — |
| `SN-PICCS-TARGET` | adopt | — |
| `SN-PICCS-EXECUTION` | adopt | — |
| `SN-PICCS-OUTPUT` | adopt | — |
| `SN-PICCS-CHARACTERIZATION` | adopt | — |
| `SN-PICCS-IDENTITY` | adopt | — |
| `SN-PICCS-DEGREE` | adopt | — |
| `SN-PICCS-LOSSES` | adopt | — |
| `SN-PICCS-EXTRACTOR-FLOW` | adopt | — |
| `SN-PICCS-EXTRACTOR-TARGET` | adopt | — |
| `SN-PIRLC-DOMAIN` | adopt | — |
| `SN-PIRLC-EQUATIONS` | adopt | — |
| `SN-PIRLC-OUTPUT` | adopt | — |
| `SN-PIRLC-FORK-LOSS` | adopt | — |
| `SN-PIRLC-FORK-SET` | adopt | — |
| `SN-PIRLC-AGREEMENT` | adopt | — |
| `SN-PIDEC-SPLIT` | adopt | — |
| `SN-PIDEC-EQUATIONS` | adopt | — |
| `SN-PIDEC-OUTPUT` | adopt | — |
| `SN-COMP-ORDER` | adopt | — |
| `SN-COMP-BINDING` | adopt | — |
| `SN-FOLD-TYPE` | adopt | — |
| `SN-FOLD-PROOF` | adopt | — |
| `SN-SEC-ABSTRACT` | adopt | — |
| `NS-ALGEBRA-PROFILE` | add | — |
| `NS-SHAPE-LOGICAL` | add | — |
| `NS-SHAPE-PADDING` | add | — |
| `NS-SHAPE-IDENTITY` | add | — |
| `NS-SHAPE-POLYNOMIAL` | add | — |
| `NS-PUBLIC-CARRIER` | add | — |
| `NS-SPLIT-BINARY` | add | — |
| `NS-COMMITMENT-PROFILE` | add | — |
| `NS-PICCS-VARIANT` | add | — |
| `NS-PICCS-PADDING-EQUIVALENCE` | add | — |
| `NS-PICCS-COINS` | add | — |
| `NS-PICCS-SUMCHECK` | add | — |
| `NS-PICCS-TERMINAL` | add | — |
| `NS-PICCS-NORM-BINDING` | add | — |
| `NS-PICCS-NO-COLUMN` | add | — |
| `NS-PICCS-CENSUS` | add | — |
| `NS-PIRLC-PROFILE` | add | — |
| `NS-PIDEC-PROFILE` | add | — |
| `NS-RED-PADDED-RELATIONS` | add | — |
| `NS-RED-COMPOSITION` | add | — |
| `NS-AUTH-STRUCTURE` | add | — |
| `NS-AUTH-CLAIM` | add | — |
| `NS-AUTH-DERIVED` | add | — |
| `NS-ENC-BASE` | add | — |
| `NS-ENC-EXTENSION` | add | — |
| `NS-ENC-RING` | add | — |
| `NS-ENC-CONTAINER` | add | — |
| `NS-ENC-STRUCTURE` | add | — |
| `NS-ENC-COMMITMENT` | add | — |
| `NS-POSEIDON-PARAMETERS` | add | — |
| `NS-TRANSCRIPT-SPONGE` | add | — |
| `NS-TRANSCRIPT-FRAMING` | add | — |
| `NS-VERIFIER-KEY-DIGEST` | add | — |
| `NS-TRANSCRIPT-ORDER` | add | — |
| `NS-CHALLENGE-EXTENSION` | add | — |
| `NS-SAMPLER-CANDIDATES` | add | — |
| `NS-SAMPLER-REPETITIONS` | add | — |
| `NS-SAMPLER-LOSS` | add | — |
| `NS-SECURITY-POLICY` | add | — |
| `NS-DECIDER-PROFILE` | add | — |
| `NS-RUST-EVIDENCE-ORIGIN` | add | — |
| `NS-RUST-EVIDENCE-CONTENT` | add | — |
| `NS-CIRCUIT-MANIFEST` | add | — |
| `NS-CIRCUIT-COMPLETENESS` | add | — |
| `NS-CIRCUIT-SOUNDNESS` | add | — |
| `NS-CIRCUIT-PUBLIC-INPUT` | add | — |
| `NS-CIRCUIT-LOWERING` | add | — |
| `NS-DECIDER-CORRESPONDENCE` | add | — |
| `NS-SEC-REDUCTION` | add | — |
| `NS-SEC-COMPOSITION` | add | — |
| `NS-RELEASE-IMPLEMENTATION` | add | — |
| `NS-RELEASE-PRODUCTION` | add | — |

## Direct dependencies and blockers

| Rule | Kind | Depends on | Decision dependencies |
|---|---|---|---|
| `SN-FND-FIELD` | definition | — | — |
| `SN-FND-RING` | definition | `SN-FND-FIELD` | — |
| `SN-FND-DIMENSIONS` | definition | `SN-FND-RING` | — |
| `SN-NORM-CENTERED` | definition | `SN-FND-FIELD` | — |
| `SN-NORM-BOUNDS` | definition | `SN-NORM-CENTERED` | — |
| `SN-SPLIT-ABSTRACT` | algorithm | `SN-NORM-BOUNDS` | — |
| `SN-EMBED-COEFFICIENTS` | encoding | `SN-FND-DIMENSIONS` | — |
| `SN-EMBED-MODULE-ACTION` | encoding | `SN-EMBED-COEFFICIENTS` | — |
| `SN-REL-STRUCTURE` | relation | `SN-FND-DIMENSIONS` | — |
| `SN-REL-CCS` | relation | `SN-NORM-BOUNDS`, `SN-EMBED-COEFFICIENTS`, `SN-REL-STRUCTURE` | — |
| `SN-REL-CE` | relation | `SN-NORM-BOUNDS`, `SN-EMBED-COEFFICIENTS`, `SN-REL-STRUCTURE` | — |
| `SN-GLOBAL-STRONG-SET` | profile | `SN-FND-RING` | — |
| `SN-GLOBAL-NORM-GUARD` | profile | `SN-NORM-BOUNDS`, `SN-GLOBAL-STRONG-SET` | — |
| `SN-GLOBAL-COMMITMENT` | assumption | `SN-FND-RING` | — |
| `SN-STRONGSET-DIVISOR` | profile | `SN-GLOBAL-STRONG-SET` | — |
| `SN-STRONGSET-DIFFERENCE` | profile | `SN-STRONGSET-DIVISOR`, `SN-NORM-CENTERED` | — |
| `SN-STRONGSET-EXPANSION` | profile | `SN-STRONGSET-DIFFERENCE` | — |
| `SN-MSIS-PARAMETERS` | assumption | `SN-GLOBAL-COMMITMENT`, `SN-GLOBAL-NORM-GUARD`, `SN-FND-DIMENSIONS` | — |
| `SN-RED-STAGE` | reduction | `SN-FND-FIELD` | — |
| `SN-RED-KNOWLEDGE` | reduction | `SN-RED-STAGE` | — |
| `SN-RED-SEQUENTIAL` | reduction | `SN-RED-KNOWLEDGE` | — |
| `SN-RED-RELATIONS` | reduction | `SN-REL-CCS`, `SN-REL-CE` | — |
| `SN-RED-PROJECTION` | reduction | `SN-RED-RELATIONS`, `SN-GLOBAL-COMMITMENT` | — |
| `SN-RED-STRONG-CONDITIONS` | reduction | `SN-RED-PROJECTION` | — |
| `SN-RED-WEAK-CONDITIONS` | reduction | `SN-RED-PROJECTION` | — |
| `SN-SUMCHECK-CLAIM` | protocol | `SN-FND-FIELD` | — |
| `SN-SUMCHECK-ROUNDS` | protocol | `SN-SUMCHECK-CLAIM` | — |
| `SN-SUMCHECK-SOUNDNESS` | security | `SN-SUMCHECK-ROUNDS` | — |
| `SN-PICCS-POLYNOMIAL` | protocol | `SN-REL-CCS`, `SN-REL-CE`, `SN-SPLIT-ABSTRACT` | — |
| `SN-PICCS-TARGET` | protocol | `SN-PICCS-POLYNOMIAL` | — |
| `SN-PICCS-EXECUTION` | protocol | `SN-PICCS-TARGET`, `SN-SUMCHECK-ROUNDS` | — |
| `SN-PICCS-OUTPUT` | protocol | `SN-PICCS-EXECUTION` | — |
| `SN-PICCS-CHARACTERIZATION` | relation | `SN-PICCS-OUTPUT` | — |
| `SN-PICCS-IDENTITY` | relation | `SN-PICCS-CHARACTERIZATION` | — |
| `SN-PICCS-DEGREE` | protocol | `SN-PICCS-POLYNOMIAL` | — |
| `SN-PICCS-LOSSES` | security | `SN-PICCS-DEGREE`, `SN-PICCS-CHARACTERIZATION`, `SN-SUMCHECK-SOUNDNESS` | — |
| `SN-PICCS-EXTRACTOR-FLOW` | security | `SN-PICCS-LOSSES`, `SN-RED-STRONG-CONDITIONS` | — |
| `SN-PICCS-EXTRACTOR-TARGET` | security | `SN-PICCS-EXTRACTOR-FLOW` | — |
| `SN-PIRLC-DOMAIN` | protocol | `SN-PICCS-OUTPUT` | — |
| `SN-PIRLC-EQUATIONS` | protocol | `SN-PIRLC-DOMAIN`, `SN-EMBED-MODULE-ACTION`, `SN-GLOBAL-STRONG-SET` | — |
| `SN-PIRLC-OUTPUT` | protocol | `SN-PIRLC-EQUATIONS`, `SN-GLOBAL-NORM-GUARD` | — |
| `SN-PIRLC-FORK-LOSS` | security | `SN-PIRLC-EQUATIONS` | — |
| `SN-PIRLC-FORK-SET` | protocol | `SN-PIRLC-FORK-LOSS` | — |
| `SN-PIRLC-AGREEMENT` | protocol | `SN-PIRLC-FORK-SET`, `SN-RED-WEAK-CONDITIONS` | — |
| `SN-PIDEC-SPLIT` | algorithm | `SN-REL-CE`, `SN-SPLIT-ABSTRACT` | — |
| `SN-PIDEC-EQUATIONS` | protocol | `SN-PIDEC-SPLIT` | — |
| `SN-PIDEC-OUTPUT` | protocol | `SN-PIDEC-EQUATIONS` | — |
| `SN-COMP-ORDER` | reduction | `SN-PIRLC-OUTPUT`, `SN-PIDEC-OUTPUT` | — |
| `SN-COMP-BINDING` | reduction | `SN-COMP-ORDER`, `SN-RED-PROJECTION` | — |
| `SN-FOLD-TYPE` | definition | `SN-COMP-BINDING` | — |
| `SN-FOLD-PROOF` | reduction | `SN-FOLD-TYPE`, `SN-RED-STRONG-CONDITIONS`, `SN-RED-WEAK-CONDITIONS`, `SN-RED-SEQUENTIAL` | — |
| `SN-SEC-ABSTRACT` | security | `SN-FOLD-PROOF`, `SN-MSIS-PARAMETERS`, `SN-PICCS-EXTRACTOR-TARGET`, `SN-PIRLC-AGREEMENT` | — |
| `NS-ALGEBRA-PROFILE` | profile | `SN-FND-RING`, `SN-NORM-BOUNDS` | — |
| `NS-SHAPE-LOGICAL` | profile | `NS-ALGEBRA-PROFILE`, `SN-REL-STRUCTURE` | — |
| `NS-SHAPE-PADDING` | profile | `NS-SHAPE-LOGICAL` | — |
| `NS-SHAPE-IDENTITY` | relation | `NS-SHAPE-PADDING` | — |
| `NS-SHAPE-POLYNOMIAL` | profile | `NS-SHAPE-IDENTITY` | — |
| `NS-PUBLIC-CARRIER` | profile | `NS-SHAPE-LOGICAL`, `SN-EMBED-COEFFICIENTS` | — |
| `NS-SPLIT-BINARY` | algorithm | `SN-SPLIT-ABSTRACT`, `NS-ALGEBRA-PROFILE` | — |
| `NS-COMMITMENT-PROFILE` | assumption | `SN-MSIS-PARAMETERS`, `NS-SHAPE-LOGICAL` | — |
| `NS-PICCS-VARIANT` | protocol | `SN-PICCS-IDENTITY`, `NS-SHAPE-POLYNOMIAL` | — |
| `NS-PICCS-PADDING-EQUIVALENCE` | relation | `NS-PICCS-VARIANT` | — |
| `NS-PICCS-COINS` | protocol | `NS-PICCS-VARIANT` | — |
| `NS-PICCS-SUMCHECK` | protocol | `NS-PICCS-COINS`, `SN-PICCS-DEGREE` | — |
| `NS-PICCS-TERMINAL` | protocol | `NS-PICCS-SUMCHECK` | — |
| `NS-PICCS-NORM-BINDING` | relation | `NS-PICCS-TERMINAL`, `NS-PICCS-PADDING-EQUIVALENCE` | — |
| `NS-PICCS-NO-COLUMN` | protocol | `NS-PICCS-NORM-BINDING`, `NS-PUBLIC-CARRIER` | — |
| `NS-PICCS-CENSUS` | profile | `NS-PICCS-NORM-BINDING` | — |
| `NS-PIRLC-PROFILE` | profile | `NS-PICCS-TERMINAL`, `SN-PIRLC-DOMAIN` | — |
| `NS-PIDEC-PROFILE` | profile | `NS-SPLIT-BINARY`, `SN-PIDEC-OUTPUT`, `NS-PUBLIC-CARRIER` | — |
| `NS-RED-PADDED-RELATIONS` | reduction | `NS-PICCS-NORM-BINDING`, `SN-RED-PROJECTION` | — |
| `NS-RED-COMPOSITION` | reduction | `NS-RED-PADDED-RELATIONS`, `SN-FOLD-PROOF`, `NS-PICCS-NO-COLUMN` | — |
| `NS-AUTH-STRUCTURE` | verifier | `NS-COMMITMENT-PROFILE` | — |
| `NS-AUTH-CLAIM` | verifier | `NS-PICCS-NO-COLUMN` | — |
| `NS-AUTH-DERIVED` | verifier | `NS-AUTH-STRUCTURE`, `NS-AUTH-CLAIM` | — |
| `NS-ENC-BASE` | encoding | `NS-ALGEBRA-PROFILE` | — |
| `NS-ENC-EXTENSION` | encoding | `NS-ENC-BASE` | — |
| `NS-ENC-RING` | encoding | `NS-ENC-EXTENSION` | — |
| `NS-ENC-CONTAINER` | encoding | `NS-ENC-BASE`, `NS-PICCS-TERMINAL`, `NS-PIDEC-PROFILE` | — |
| `NS-ENC-STRUCTURE` | encoding | `NS-ENC-BASE`, `NS-AUTH-STRUCTURE` | — |
| `NS-ENC-COMMITMENT` | encoding | `NS-ENC-RING` | — |
| `NS-POSEIDON-PARAMETERS` | transcript | `NS-ENC-BASE` | — |
| `NS-TRANSCRIPT-SPONGE` | transcript | `NS-POSEIDON-PARAMETERS` | — |
| `NS-TRANSCRIPT-FRAMING` | transcript | `NS-TRANSCRIPT-SPONGE` | — |
| `NS-VERIFIER-KEY-DIGEST` | transcript | `NS-ENC-STRUCTURE`, `NS-TRANSCRIPT-FRAMING` | — |
| `NS-TRANSCRIPT-ORDER` | transcript | `NS-VERIFIER-KEY-DIGEST`, `NS-PIRLC-PROFILE`, `NS-PIDEC-PROFILE` | — |
| `NS-CHALLENGE-EXTENSION` | transcript | `NS-TRANSCRIPT-ORDER`, `NS-ENC-EXTENSION` | — |
| `NS-SAMPLER-CANDIDATES` | sampler | `NS-TRANSCRIPT-ORDER`, `SN-STRONGSET-DIVISOR` | — |
| `NS-SAMPLER-REPETITIONS` | sampler | `NS-SAMPLER-CANDIDATES` | — |
| `NS-SAMPLER-LOSS` | sampler | `NS-SAMPLER-REPETITIONS` | — |
| `NS-SECURITY-POLICY` | definition | `NS-PICCS-CENSUS`, `SN-SEC-ABSTRACT`, `NS-SAMPLER-LOSS` | — |
| `NS-DECIDER-PROFILE` | decider | `NS-ENC-CONTAINER`, `NS-TRANSCRIPT-ORDER` | — |
| `NS-RUST-EVIDENCE-ORIGIN` | conformance | `NS-AUTH-DERIVED`, `NS-ENC-CONTAINER`, `NS-TRANSCRIPT-ORDER` | — |
| `NS-RUST-EVIDENCE-CONTENT` | conformance | `NS-RUST-EVIDENCE-ORIGIN` | — |
| `NS-CIRCUIT-MANIFEST` | circuit | `NS-RUST-EVIDENCE-CONTENT`, `NS-DECIDER-PROFILE` | — |
| `NS-CIRCUIT-COMPLETENESS` | circuit | `NS-CIRCUIT-MANIFEST` | — |
| `NS-CIRCUIT-SOUNDNESS` | security | `NS-CIRCUIT-MANIFEST` | — |
| `NS-CIRCUIT-PUBLIC-INPUT` | circuit | `NS-CIRCUIT-MANIFEST` | — |
| `NS-CIRCUIT-LOWERING` | circuit | `NS-CIRCUIT-MANIFEST` | — |
| `NS-DECIDER-CORRESPONDENCE` | decider | `NS-CIRCUIT-SOUNDNESS`, `NS-CIRCUIT-PUBLIC-INPUT` | — |
| `NS-SEC-REDUCTION` | security | `NS-RED-COMPOSITION`, `NS-SECURITY-POLICY`, `NS-DECIDER-CORRESPONDENCE` | — |
| `NS-SEC-COMPOSITION` | security | `NS-SEC-REDUCTION` | — |
| `NS-RELEASE-IMPLEMENTATION` | release | `NS-SAMPLER-LOSS`, `NS-RED-COMPOSITION`, `NS-DECIDER-PROFILE` | — |
| `NS-RELEASE-PRODUCTION` | release | `NS-RELEASE-IMPLEMENTATION`, `NS-CIRCUIT-COMPLETENESS`, `NS-CIRCUIT-LOWERING`, `NS-SEC-COMPOSITION` | — |
