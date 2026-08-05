## 2. Reviewed reduction framework

### SN-RED-STAGE — Interactive reduction interface

Each reduction stage MUST be one tuple `(G,K,P,V)`, and `K` MUST be
deterministic. All composed stages MUST use the same generator and encoder.

Source: PAPER-RED-001 and PAPER-RED-002.

### SN-RED-KNOWLEDGE — Reduction-of-knowledge conditions

A stage that claims a reduction of knowledge MUST establish completeness,
knowledge soundness, and public coin. All verifier randomness MUST appear in
uniform verifier messages.

Source: PAPER-RED-001 and PAPER-RED-002.

### SN-RED-SEQUENTIAL — Sequential composition boundary

Sequential composition MUST bind the first stage output to the second stage
input under one shared setup and encoder.

Source: PAPER-RED-002 and PAPER-RED-006.

### SN-RED-RELATIONS — Paper strong and weak relation pairs

The paper composition uses these exact relation pairs:

```text
PiCCS: CCS(b,L)^K_fresh * CE(b,L)^k
        -> CE(b,L)^(K_fresh+k), ambient CE(B_amb,L)^(K_fresh+k)
PiRLC: CE(b,L)^(K_fresh+k), ambient CE(B_amb,L)^(K_fresh+k)
        -> CE(B,L).
```

Source: PAPER-RED-003 through PAPER-RED-005, PAPER-PICCS-006,
PAPER-PIRLC-002, and ERR-AMBIENT.

### SN-RED-PROJECTION — Shared commitment projection

Both stages MUST use the same function `phi`, equal to the ordered commitment
projection of their instances.

Source: PAPER-RED-003 through PAPER-RED-005.

### SN-RED-STRONG-CONDITIONS — PiCCS strong conditions

PiCCS MUST preserve the same `phi` image across two independent prover runs
with probability one, and its relaxed extractor MUST target the ambient
PiCCS relation from SN-RED-RELATIONS. Under the strong definition's output-
witness agreement premise, extraction MUST recover an input witness with
probability at least relaxed success minus a negligible term.

Source: PAPER-RED-003, PAPER-RED-005, and PAPER-PICCS-006.

### SN-RED-WEAK-CONDITIONS — PiRLC weak conditions

PiRLC MUST have an extractor for its ambient input relation with probability
at least adversary success minus a negligible term. For two input instances
with the same `phi` image, its extracted witnesses MUST agree except with
negligible probability.

Source: PAPER-RED-004, PAPER-RED-005, and PAPER-PIRLC-002.
