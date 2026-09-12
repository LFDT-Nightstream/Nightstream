

# Neo and SuperNeo: Post-quantum folding with pay-per-bit costs over small fields

Wilson Nguyen  
Microsoft Research

Srinath Setty  
Microsoft Research

# Abstract.

We construct the first folding scheme that simultaneously achieves six desirable properties: plausible post-quantum security, pay-per-bit commitment costs, field-native arithmetic (the sum-check and norm checks run purely over a small field), support for general (non-SIMD) constraint systems, small-field support (e.g., Goldilocks), and low recursion overheads. No existing scheme satisfies all six: group-based schemes (e.g., HyperNova) lack post-quantum security and are tied to large elliptic-curve fields; lattice-based schemes (e.g., LatticeFold) require expensive ring arithmetic, lose pay-per-bit costs, and impose SIMD constraints; and hash-based schemes (e.g., Arc) incur large verifier circuits.

We present two lattice-based folding schemes for CCS—an NP-complete relation generalizing R1CS, Plonkish, and AIR—called Neo and SuperNeo. Neo satisfies five of the six properties but requires SIMD constraint systems; SuperNeo removes this restriction and satisfies all six. SuperNeo also natively supports CCS relations over arbitrary extension fields of the field underlying the Ajtai commitment, without relying on an NTT embedding. Both run a single invocation of the sum-check protocol over a small field extension and achieve pay-per-bit costs via new folding-friendly instantiations of Ajtai commitments under the Module-SIS assumption. At the core of our constructions are two new norm-preserving embeddings of field vectors into ring vectors that respect an evaluation homomorphism required for folding. We also introduce *interactive reductions*, a framework that generalizes reductions of knowledge and enables modular security proofs for composed lattice-based protocols.

# Table of Contents

|       |                                                                  |    |
|-------|------------------------------------------------------------------|----|
| 1     | Introduction .....                                               | 3  |
| 1.1   | Six desiderata for a practical folding scheme.....               | 3  |
| 1.2   | Our work: Neo and SuperNeo .....                                 | 5  |
| 1.2.1 | Challenges and prior solutions .....                             | 6  |
| 1.2.2 | Contributions of our work.....                                   | 8  |
| 1.3   | Related works .....                                              | 12 |
| 2     | Technical overview .....                                         | 13 |
| 2.1   | Breaking down HyperNova .....                                    | 13 |
| 2.2   | The Neo embedding .....                                          | 14 |
| 2.3   | The SuperNeo embedding .....                                     | 14 |
| 2.4   | Proving the security with interactive reductions .....           | 16 |
| 3     | Overview of the following sections .....                         | 17 |
| 4     | Preliminaries .....                                              | 17 |
| 5     | Embeddings and Evaluation Homomorphism .....                     | 22 |
| 6     | Strong and weak interactive reductions .....                     | 27 |
| 7     | SuperNeo’s folding scheme for CCS .....                          | 28 |
| 7.1   | Relations .....                                                  | 28 |
| 7.2   | A folding scheme for CCS via interactive reductions .....        | 29 |
| 7.3   | Interactive reduction for CCS – $\Pi_{\text{CCS}}$ .....         | 30 |
| 7.4   | Random linear combination reduction – $\Pi_{\text{RLC}}$ .....   | 32 |
| 7.5   | Decomposition reduction – $\Pi_{\text{DEC}}$ .....               | 33 |
| 8     | Concrete parameters .....                                        | 34 |
| 8.1   | Almost Goldilocks: $(2^{64} - 2^{32} + 1) - 32$ .....            | 34 |
| 8.2   | Goldilocks: $(2^{64} - 2^{32} + 1)$ .....                        | 34 |
| 8.3   | Mersenne 61: $2^{61} - 1$ .....                                  | 34 |
|       | <b>References</b> .....                                          | 35 |
| A     | AI Disclaimer .....                                              | 40 |
| B     | Deferred theorems and proofs .....                               | 40 |
| B.1   | Proof of Composition Theorem (Theorem 12) .....                  | 40 |
| B.2   | Proofs for $\Pi_{\text{CCS}}$ .....                              | 42 |
| B.3   | Proofs for $\Pi_{\text{RLC}}$ .....                              | 48 |
| B.4   | $\Pi_{\text{DEC}}$ is a Reduction of Knowledge (Theorem 13)..... | 54 |
| B.5   | Hardness and Inversion Bound calculation scripts .....           | 56 |
| B.6   | Lattice Estimator Script .....                                   | 58 |

