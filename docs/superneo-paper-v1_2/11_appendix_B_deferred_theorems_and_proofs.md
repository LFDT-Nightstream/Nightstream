# B Deferred theorems and proofs

## B.1 Proof of Composition Theorem (Theorem 12)

*Proof.* Consider an arbitrary expected polynomial time adversary  $(\mathcal{A}, \mathcal{P}^*)$  for the composition  $\Pi := \Pi_2 \circ \Pi_1$  with success probability  $\epsilon(\mathcal{A}, \mathcal{P}^*) \geq 1/\text{poly}(\lambda)$ . Without loss of generality, the adversary  $\mathcal{P}^*$  can be split into two adversaries  $(\mathcal{P}_1^*, \mathcal{P}_2^*)$  such that given  $\text{pp} \leftarrow \mathcal{G}(1^\lambda, \text{sz})$ ,  $(\mathbf{s}, u_1, \text{st}_1) \leftarrow \mathcal{A}(\text{pp})$ , and  $(\text{pk}, \text{vk}) \leftarrow \mathcal{K}(\text{pp}, \mathbf{s})$ ,

- $\langle \mathcal{P}_1^*, \mathcal{V}_1 \rangle((\text{pk}, \text{vk}), u_1, \text{st}_1) \rightarrow (u_2, \text{st}_2)$
- $\langle \mathcal{P}_2^*, \mathcal{V}_2 \rangle((\text{pk}, \text{vk}), u_2, \text{st}_2) \rightarrow (u_3, w_3)$

Furthermore, we assume that  $\mathcal{A}$  outputs  $\text{st}_1$  which contains  $(\mathbf{s}, \text{pp}, u_1)$ ; otherwise, we could trivially construct an adversary  $\mathcal{A}'$  with an identical distribution of prior outputs that does so. First, we construct an adversary  $\mathcal{A}_2 := (\mathcal{B}_2, \mathcal{B}'_2)$  for  $\Pi_2$ :

```

 $\mathcal{B}_2(\text{pp}) \rightarrow (\mathbf{s}, \text{st}_1)$  :
  1.  $(\mathbf{s}, u_1, \text{st}_1) \leftarrow \mathcal{A}(\text{pp})$ .
  2. Output  $(\mathbf{s}, \text{st}_1)$ .

 $\mathcal{B}'_2(\text{st}_1) \rightarrow (u_2, \text{st}_2)$  :
  1. Parse  $\text{st}_1$  to obtain  $(\mathbf{s}, \text{pp}, u_1)$ .
  2.  $(\text{pk}, \text{vk}) \leftarrow \mathcal{K}(\text{pp}, \mathbf{s})$ .
  3. Simulate  $(u_2, \text{st}_2) \leftarrow \langle \mathcal{P}_1^*, \mathcal{V}_1 \rangle((\text{pk}, \text{vk}), u_1, \text{st}_1)$ .
  4. Output  $(u_2, \text{st}_2)$ .

 $\mathcal{A}_2(\text{pp}) \rightarrow (\mathbf{s}, u_2, \text{st}_2)$  :
  1.  $(\mathbf{s}, \text{st}_1) \leftarrow \mathcal{B}_2(\text{pp})$ .
  2.  $(u_2, \text{st}_2) \leftarrow \mathcal{B}'_2(\text{st}_1)$ .
  3. Output  $(\mathbf{s}, u_2, \text{st}_2)$ .
  
```

The adversary  $\mathcal{A}_2$  runs in expected polynomial time since  $\mathcal{A}$  and  $\mathcal{P}_1^*$  do, and all remaining computations run in polynomial time. Observe that, by construction, the success probability  $\epsilon(\mathcal{A}_2, \mathcal{P}_2^*)$  of adversary  $(\mathcal{A}_2, \mathcal{P}_2^*)$  for  $\Pi_2$  is equal to the success probability  $\epsilon(\mathcal{A}, \mathcal{P}^*)$  of adversary  $(\mathcal{A}, \mathcal{P}^*)$  for  $\Pi$ . By construction,  $\mathcal{B}_2$  samples the input and state according to  $\mathcal{A}$ , while each invocation of  $\mathcal{B}'_2$  simulates one execution of  $\langle \mathcal{P}_1^*, \mathcal{V}_1 \rangle$  on that input and state. Thus, the distribution of  $(u_2, u'_2)$  in the following experiment is identical to that in condition (i) for  $\Pi_1$  with respect to adversary  $(\mathcal{A}, \mathcal{P}_1^*)$ . By assumption,  $\Pi_1$  is a strong interactive reduction (Definition 18). As such, it satisfies condition (i); thus, we must have

$$\Pr \left[ \begin{array}{c} u_2, u'_2 \neq \perp \\ \Downarrow \\ \phi(u_2) = \phi(u'_2) \end{array} \middle| \begin{array}{l} \text{pp} \leftarrow \mathcal{G}(1^\lambda, \text{sz}) \\ (\mathbf{s}, \text{st}_1) \leftarrow \mathcal{B}_2(\text{pp}) \\ (u_2, \text{st}_2) \leftarrow \mathcal{B}'_2(\text{st}_1) \\ (u'_2, \text{st}'_2) \leftarrow \mathcal{B}'_2(\text{st}_1) \end{array} \right] = 1, \quad (1)$$

By assumption,  $\Pi_2$  is a weak interactive reduction (Definition 17). Thus, with respect to adversary  $(\mathcal{A}_2, \mathcal{P}_2^*)$ , there exists an expected polynomial time extractor  $\mathcal{E}_2$  such that, since  $\epsilon(\mathcal{A}_2, \mathcal{P}_2^*) = \epsilon(\mathcal{A}, \mathcal{P}^*) \geq 1/\text{poly}(\lambda)$ ,

$$\Pr \left[ (\text{pp}, \mathbf{s}, u_2, w_2) \in \mathcal{R}'_2 \middle| \begin{array}{l} \text{pp} \leftarrow \mathcal{G}(1^\lambda, \text{sz}) \\ (\mathbf{s}, u_2, \text{st}_2) \leftarrow \mathcal{A}_2(\text{pp}) \\ (\text{pk}, \text{vk}) \leftarrow \mathcal{K}(\text{pp}, \mathbf{s}) \\ w_2 \leftarrow \mathcal{E}_2(\text{pp}, \mathbf{s}, u_2, \text{st}_2) \end{array} \right] \geq \epsilon(\mathcal{A}, \mathcal{P}^*) - \text{negl}(\lambda) \quad (2)$$

Moreover, since  $\mathcal{A}_2 = (\mathcal{B}_2, \mathcal{B}'_2)$  satisfies (1), the same extractor  $\mathcal{E}_2$  satisfies condition (ii) of weak interactive reductions (Definition 17). Let  $\mathcal{O}_{\text{weak}}$  be the oracle from Definition 17 instantiated with  $\mathcal{B}'_2$  and  $\mathcal{E}_2$ . Thus, for any uniqueness adversary  $\mathcal{W}_{\text{uniq}}^{\mathcal{O}_{\text{weak}}}$ ,

$$\Pr \left[ \begin{array}{l} w_2, w'_2 \neq \perp \\ \wedge w_2 \neq w'_2 \end{array} \middle| \begin{array}{l} \mathbf{pp} \leftarrow \mathcal{G}(1^\lambda, \mathbf{sz}) \\ (\mathbf{s}, \mathbf{st}_1) \leftarrow \mathcal{B}_2(\mathbf{pp}) \\ \mathbf{in} \leftarrow (\mathbf{pp}, \mathbf{s}, \mathbf{st}_1) \\ (u_2, w_2), (u'_2, w'_2) \leftarrow \mathcal{W}_{\text{uniq}}^{\mathcal{O}_{\text{weak}}(\mathbf{in})}(\mathbf{in}) \end{array} \right] \leq \text{negl}(\lambda) \quad (3)$$

Next, we will construct an adversary  $\mathcal{P}_1^{**}$  for  $\Pi_1$ . Recall that  $\mathcal{V}_1$  is the verifier of  $\Pi_1$ . During the interaction  $\langle \mathcal{P}_1^{**}, \mathcal{V}_1 \rangle$ ,  $\mathcal{P}_1^{**}$  internally runs  $\mathcal{P}_1^*$  and relays messages between  $\mathcal{P}_1^*$  and  $\mathcal{V}_1$ :

$\mathcal{P}_1^{**}(\mathbf{pk}, u_1, \mathbf{st}_1) \rightarrow w_2$  :

1. Parse  $\mathbf{st}_1$  to obtain  $(\mathbf{s}, \mathbf{pp})$ .
2.  $(\mathbf{pk}, \mathbf{vk}) \leftarrow \mathcal{K}(\mathbf{pp}, \mathbf{s})$ .
3. Run  $\mathcal{P}_1^*(\mathbf{pk}, u_1, \mathbf{st}_1)$ , relaying each message from  $\mathcal{V}_1$  to  $\mathcal{P}_1^*$  and each response from  $\mathcal{P}_1^*$  back to  $\mathcal{V}_1$ . Let  $\mathbf{tr}$  be the resulting transcript and  $\mathbf{st}_2$  the final state of  $\mathcal{P}_1^*$ .
4. Since  $\Pi_1$  is public coin, compute the verifier's output  $u_2$  by emulating  $\mathcal{V}_1(\mathbf{vk}, u_1)$  on  $\mathbf{tr}$ .
5.  $w_2 \leftarrow \mathcal{E}_2(\mathbf{pp}, \mathbf{s}, u_2, \mathbf{st}_2)$ .
6. Output  $w_2$ .

The adversary  $\mathcal{P}_1^{**}$  runs in expected polynomial time since  $\mathcal{P}_1^*$  and  $\mathcal{E}_2$  do, and all remaining computations run in polynomial time. Conditioned on  $(\mathbf{pp}, \mathbf{s}, u_1, \mathbf{st}_1)$ ,  $\mathcal{B}'_2(\mathbf{st}_1)$  and the interaction  $\langle \mathcal{P}_1^{**}, \mathcal{V}_1 \rangle$  generate  $(u_2, \mathbf{st}_2)$  according to the same distribution: both run  $\mathcal{P}_1^*$  against  $\mathcal{V}_1$  using the same keys produced by the deterministic encoder. Moreover, since  $\Pi_1$  is public coin,  $\mathcal{P}_1^{**}$  can reconstruct from  $\mathbf{tr}$  the output  $u_2$  produced by  $\mathcal{V}_1$ . Hence, the event in (2) is exactly the event defining  $\epsilon'(\mathcal{A}, \mathcal{P}_1^{**})$ , and therefore  $\epsilon'(\mathcal{A}, \mathcal{P}_1^{**}) \geq \epsilon(\mathcal{A}, \mathcal{P}^*) - \text{negl}(\lambda) \geq 1/\text{poly}(\lambda)$ .

Next, consider any uniqueness adversary  $\mathcal{S}_{\text{uniq}}^{\mathcal{O}_{\text{str}}}$  (Definition 16), where  $\mathcal{O}_{\text{str}}$  is the oracle from condition (ii) of strong interactive reductions (Definition 18) instantiated with  $\mathcal{P}_1^{**}$ . Recall that  $\mathbf{st}_1$  contains  $(\mathbf{s}, \mathbf{pp}, u_1)$ . Define  $\mathbf{in}_{\text{weak}} := (\mathbf{pp}, \mathbf{s}, \mathbf{st}_1)$  and  $\mathbf{in}_{\text{str}} := (\mathbf{pp}, \mathbf{s}, u_1, \mathbf{st}_1)$ , where  $u_1$  is parsed from  $\mathbf{st}_1$ . Define

$$\mathcal{W}_{\text{uniq}}^{\mathcal{O}_{\text{weak}}(\mathbf{in}_{\text{weak}})}(\mathbf{in}_{\text{weak}}) := \mathcal{S}_{\text{uniq}}^{\mathcal{O}_{\text{weak}}(\mathbf{in}_{\text{weak}})}(\mathbf{in}_{\text{str}}).$$

In particular,  $\mathcal{W}_{\text{uniq}}$  runs the code of  $\mathcal{S}_{\text{uniq}}$ , but answers each oracle query using  $\mathcal{O}_{\text{weak}}(\mathbf{in}_{\text{weak}})$ . By definition,

$$\begin{array}{ll} \mathcal{O}_{\text{weak}}(\mathbf{in}_{\text{weak}}) \rightarrow (u_2, w_2) : & \mathcal{O}_{\text{str}}(\mathbf{in}_{\text{str}}) \rightarrow (u_2, w_2) : \\ 1. (u_2, \mathbf{st}_2) \leftarrow \mathcal{B}'_2(\mathbf{st}_1). & 1. (\mathbf{pk}, \mathbf{vk}) \leftarrow \mathcal{K}(\mathbf{pp}, \mathbf{s}). \\ 2. w_2 \leftarrow \mathcal{E}_2(\mathbf{pp}, \mathbf{s}, u_2, \mathbf{st}_2). & 2. (u_2, w_2) \leftarrow \langle \mathcal{P}_1^{**}, \mathcal{V}_1 \rangle((\mathbf{pk}, \mathbf{vk}), u_1, \mathbf{st}_1). \\ 3. \text{Output } (u_2, w_2). & 3. \text{Output } (u_2, w_2). \end{array}$$

By construction,  $\mathcal{B}'_2$  runs  $\mathcal{P}_1^*$  against  $\mathcal{V}_1$ . The prover  $\mathcal{P}_1^{**}$  runs the same interaction, reconstructs  $u_2$ , and runs  $\mathcal{E}_2$  on  $(\mathbf{pp}, \mathbf{s}, u_2, \mathbf{st}_2)$ . Since the encoder is deterministic, the two oracles have the same output distribution. Since each oracle call uses fresh independent randomness, the view and output of  $\mathcal{S}_{\text{uniq}}$  have the same distribution under either oracle. Furthermore,  $\mathcal{W}_{\text{uniq}}^{\mathcal{O}_{\text{weak}}}$  runs in expected polynomial time. Since  $\mathcal{S}_{\text{uniq}}^{\mathcal{O}_{\text{str}}}$  is EPT, its computation outside oracle calls and its expected number of oracle calls are polynomial. The adversary  $\mathcal{W}_{\text{uniq}}$  runs the same computation but replaces each call to  $\mathcal{O}_{\text{str}}$  with one call to  $\mathcal{O}_{\text{weak}}$ . By the definitions of  $\mathcal{B}'_2$  and  $\mathcal{P}_1^{**}$ , each replacement runs the same interaction between  $\mathcal{P}_1^*$  and  $\mathcal{V}_1$  followed by  $\mathcal{E}_2$ , with only polynomial time differences. Thus,  $\mathcal{W}_{\text{uniq}}^{\mathcal{O}_{\text{weak}}}$  is EPT. Thus, the following experiment has the same distribution as the experiment in (3); by (3), we must have

$$\Pr \left[ \begin{array}{l} w_2, w'_2 \neq \perp \\ \wedge \\ w_2 \neq w'_2 \end{array} \middle| \begin{array}{l} \mathbf{pp} \leftarrow \mathcal{G}(1^\lambda, \mathbf{sz}) \\ (\mathbf{s}, u_1, \mathbf{st}_1) \leftarrow \mathcal{A}(\mathbf{pp}) \\ \mathbf{in}_{\text{str}} \leftarrow (\mathbf{pp}, \mathbf{s}, u_1, \mathbf{st}_1) \\ (u_2, w_2), (u'_2, w'_2) \leftarrow \mathcal{S}_{\text{uniq}}^{\mathcal{O}_{\text{str}}(\mathbf{in}_{\text{str}})}(\mathbf{in}_{\text{str}}) \end{array} \right] \leq \text{negl}(\lambda) \quad (4)$$

Since  $\mathcal{S}_{\text{uniq}}$  was arbitrary, (4) holds for any uniqueness adversary. Thus, by (4), the bound  $\epsilon'(\mathcal{A}, \mathcal{P}_1^{**}) \geq 1/\text{poly}(\lambda)$  established above, and condition (ii) of Definition 18 for  $\Pi_1$ , there exists an expected polynomial time extractor  $\mathcal{E}_1$  such that

$$\Pr \left[ (\mathbf{pp}, \mathbf{s}, u_1, w_1) \in \mathcal{R}_1 \left| \begin{array}{l} \mathbf{pp} \leftarrow \mathcal{G}(1^\lambda, \mathbf{sz}) \\ (\mathbf{s}, u_1, \mathbf{st}_1) \leftarrow \mathcal{A}(\mathbf{pp}) \\ (\mathbf{pk}, \mathbf{vk}) \leftarrow \mathcal{K}(\mathbf{pp}, \mathbf{s}) \\ w_1 \leftarrow \mathcal{E}_1(\mathbf{pp}, \mathbf{s}, u_1, \mathbf{st}_1) \end{array} \right. \right] \geq \epsilon'(\mathcal{A}, \mathcal{P}_1^{**}) - \text{negl}(\lambda) \quad (5)$$

By (2) and the preceding argument,

$$\epsilon'(\mathcal{A}, \mathcal{P}_1^{**}) \geq \epsilon(\mathcal{A}, \mathcal{P}^*) - \text{negl}(\lambda).$$

Therefore, combining these inequalities and absorbing the sum of the two negligible functions into  $\text{negl}(\lambda)$ , we have

$$\Pr \left[ (\mathbf{pp}, \mathbf{s}, u_1, w_1) \in \mathcal{R}_1 \left| \begin{array}{l} \mathbf{pp} \leftarrow \mathcal{G}(1^\lambda, \mathbf{sz}) \\ (\mathbf{s}, u_1, \mathbf{st}_1) \leftarrow \mathcal{A}(\mathbf{pp}) \\ (\mathbf{pk}, \mathbf{vk}) \leftarrow \mathcal{K}(\mathbf{pp}, \mathbf{s}) \\ w_1 \leftarrow \mathcal{E}_1(\mathbf{pp}, \mathbf{s}, u_1, \mathbf{st}_1) \end{array} \right. \right] \geq \epsilon(\mathcal{A}, \mathcal{P}^*) - \text{negl}(\lambda).$$

In conclusion, for the arbitrary adversary  $(\mathcal{A}, \mathcal{P}^*)$  fixed above, set  $\mathcal{E} := \mathcal{E}_1$ . The preceding inequality shows that  $\mathcal{E}$  extracts a valid witness for  $\mathcal{R}_1$  with probability at least  $\epsilon(\mathcal{A}, \mathcal{P}^*) - \text{negl}(\lambda)$ . Thus,  $\Pi := \Pi_2 \circ \Pi_1$  is knowledge sound. Since  $\Pi_1$  and  $\Pi_2$  are complete and public coin, their sequential composition  $\Pi$  is also complete and public coin. Therefore,  $\Pi$  is a reduction of knowledge (Definition 9).  $\square$

## B.2 Proofs for $\Pi_{\text{CCS}}$

We first provide a lemma that will be helpful for both the security and completeness of the interactive reduction.

**Lemma 9.** *Consider the following arbitrary items:*

$$\begin{aligned} & \text{structure } \mathbf{s}, \quad \text{vectors } z_1, \dots, z_{K+k} \in \mathbb{L}^{n_L}, \\ & \text{point } r \in \mathbb{K}^{\log m}, \quad \text{evaluations } (y_i \in \mathbb{R}_{\mathbb{K}}, \{y_{i,j} \in \mathbb{R}_{\mathbb{A}}\}_{j \in [t]})_{i=K+1}^{K+k} \end{aligned}$$

Recall from Definition 21 the matrix  $\text{Pad} \in \mathbb{F}^{m \times n_{\mathbb{F}}}$ . For every  $i \in [K+k]$ , define

$$\begin{aligned} z_i^b &:= \text{Flat}_{\mathbb{L}}(z_i) \in \mathbb{F}^{n_{\mathbb{F}}} \\ z_i^{b, \text{pad}} &:= \text{Pad } z_i^b \in \mathbb{F}^m \\ \mathbf{z}_i &:= \text{Pack}_{\mathbb{L}}(z_i) = \text{Pack}_{\mathbb{F}}(z_i^b) \in \mathbb{R}_{\mathbb{F}}^{n_{\mathbb{R}}} \\ h_i &:= \text{Emb}_{\mathbb{F}}(\text{Trans}_{\mathbb{F}}(\text{Pad})) \cdot \mathbf{z}_i \in \mathbb{R}_{\mathbb{F}}^m \end{aligned}$$

For every  $i \in [K+k]$  and  $j \in [t]$ , define

$$h_{i,j} := \text{Emb}_{\mathbb{L}}(\text{Trans}_{\mathbb{L}}(M_j)) \cdot \mathbf{z}_i \in \mathbb{R}_{\mathbb{L}}^m$$

Similarly to  $\Pi_{\text{CCS}}$  (Section 7.3), define

$$\begin{aligned} \mathbf{F}(\vec{X}, C) &:= \sum_{i=1}^K C^{i-1} \cdot f(\widetilde{M_1 z_i}, \dots, \widetilde{M_t z_i}) \\ \mathbf{NC}(\vec{X}, C) &:= \sum_{i=1}^{K+k} C^{i-1} \cdot \prod_{a=-b+1}^{b-1} \left( \widetilde{z_i^{b, \text{pad}}} - a \right) \end{aligned}$$

$$\begin{aligned}
\text{Eval}_{\mathbb{K}}(\vec{X}, C) &:= \text{eq}(\vec{X}, r) \cdot \sum_{i=K+1}^{K+k} \sum_{\ell=1}^d C^{\mathbb{I}_{\mathbb{K}}(i, \ell)} \cdot \widetilde{\text{cf}(h_i)}_{\ell} \\
\text{Eval}_{\mathbb{A}}(\vec{X}, C) &:= \text{eq}(\vec{X}, r) \cdot \sum_{i=K+1}^{K+k} \sum_{j=1}^t \sum_{\ell=1}^d C^{\mathbb{I}_{\mathbb{A}}(i, j, \ell)} \cdot \widetilde{\text{cf}(h_{i,j})}_{\ell} \\
Q(\vec{X}, \vec{A}, C) &:= \text{Eval}_{\mathbb{K}}(\vec{X}, C) + C^{kd} \cdot \text{Eval}_{\mathbb{A}}(\vec{X}, C) + C^{kd(t+1)} \cdot \text{eq}(\vec{X}, \vec{A}) \left( \mathbb{F}(\vec{X}, C) + C^K \cdot \text{NC}(\vec{X}, C) \right) \\
T(C) &:= \sum_{i=K+1}^{K+k} \sum_{\ell=1}^d C^{\mathbb{I}_{\mathbb{K}}(i, \ell)} \cdot \text{cf}(y_i)_{\ell} + C^{kd} \cdot \sum_{i=K+1}^{K+k} \sum_{j=1}^t \sum_{\ell=1}^d C^{\mathbb{I}_{\mathbb{A}}(i, j, \ell)} \cdot \text{cf}(y_{i,j})_{\ell}
\end{aligned}$$

where  $\vec{X} := (X_1, \dots, X_{\log m})$ , and the challenges  $\alpha \in \mathbb{K}^{\log m}$  and  $\gamma \in \mathbb{K}$  are replaced by the indeterminates  $\vec{A} := (A_1, \dots, A_{\log m})$  and  $C$ , respectively.

We must have  $T(C) = \sum_{\vec{x} \in \{0,1\}^{\log m}} Q(\vec{x}, \vec{A}, C)$  if and only if

1.  $f(\widetilde{M_1 z_i}, \dots, \widetilde{M_t z_i}) \in \text{ZS}_{\log m}(\mathbb{L})$  for all  $i \in [K]$ ,
2.  $\|z_i^b\|_{\infty} < b$  for all  $i \in [K+k]$ ,
3.  $y_i = \widetilde{h_i}(r)$  for all  $i \in [K+1, K+k]$ ,
4.  $y_{i,j} = \widetilde{h_{i,j}}(r)$  for all  $i \in [K+1, K+k]$  and  $j \in [t]$ .

*Proof.* By definition of  $T(C)$ ,

$$T(C) = \sum_{\vec{x} \in \{0,1\}^{\log m}} Q(\vec{x}, \vec{A}, C)$$

if and only if

$$\sum_{i=K+1}^{K+k} \sum_{\ell=1}^d C^{\mathbb{I}_{\mathbb{K}}(i, \ell)} \cdot \text{cf}(y_i)_{\ell} + C^{kd} \cdot \sum_{i=K+1}^{K+k} \sum_{j=1}^t \sum_{\ell=1}^d C^{\mathbb{I}_{\mathbb{A}}(i, j, \ell)} \cdot \text{cf}(y_{i,j})_{\ell} = \sum_{\vec{x} \in \{0,1\}^{\log m}} Q(\vec{x}, \vec{A}, C) \quad (6)$$

Since powers of  $C$  are linearly independent, Equation (6) occurs if and only if

$$\forall i \in [K], \quad 0 = \sum_{\vec{x} \in \{0,1\}^{\log m}} \text{eq}(\vec{x}, \vec{A}) \cdot f(\widetilde{M_1 z_i}(\vec{x}), \dots, \widetilde{M_t z_i}(\vec{x})), \quad (7)$$

$$\forall i \in [K+k], \quad 0 = \sum_{\vec{x} \in \{0,1\}^{\log m}} \text{eq}(\vec{x}, \vec{A}) \cdot \prod_{a=-b+1}^{b-1} \left( \widetilde{z_i^{b, \text{pad}}}(\vec{x}) - a \right), \quad (8)$$

$$\forall i \in [K+1, K+k], \quad \forall \ell \in [d], \quad \text{cf}(y_i)_{\ell} = \sum_{\vec{x} \in \{0,1\}^{\log m}} \text{eq}(\vec{x}, r) \cdot \widetilde{\text{cf}(h_i)}_{\ell}(\vec{x}) \quad (9)$$

$$\forall i \in [K+1, K+k], \quad \forall j \in [t], \quad \forall \ell \in [d], \quad \text{cf}(y_{i,j})_{\ell} = \sum_{\vec{x} \in \{0,1\}^{\log m}} \text{eq}(\vec{x}, r) \cdot \widetilde{\text{cf}(h_{i,j})}_{\ell}(\vec{x}) \quad (10)$$

By Lemma 3, Equation (7) and Equation (8) occur if and only if

1.  $f(\widetilde{M_1 z_i}, \dots, \widetilde{M_t z_i}) \in \text{ZS}_{\log m}(\mathbb{L})$  for all  $i \in [K]$  (Item 1),
2.  $\prod_{a=-b+1}^{b-1} \left( \widetilde{z_i^{b, \text{pad}}} - a \right) \in \text{ZS}_{\log m}(\mathbb{F})$  for all  $i \in [K+k]$ .

Since  $z_i^{b,\text{pad}} = \text{Pad } z_i^b$  only appends zero coordinates and  $0 \in \{-b+1, \dots, b-1\}$ , Item 2 holds if and only if  $\|z_i^b\|_\infty < b$  for all  $i \in [K+k]$  (Item 2). By definition of multilinear extension and bijectivity of the coefficient map, Equation (9) and Equation (10) hold if and only if

$$\begin{aligned} y_i &= \widetilde{h_i}(r) && \text{for all } i \in [K+1, K+k] && \text{(Item 3),} \\ y_{i,j} &= \widetilde{h_{i,j}}(r) && \text{for all } i \in [K+1, K+k], j \in [t] && \text{(Item 4).} \end{aligned}$$

In conclusion, we have shown

$$T(C) = \sum_{\vec{x} \in \{0,1\}^{\log m}} Q(\vec{x}, \vec{A}, C)$$

if and only if (Item 1), (Item 2), (Item 3), and (Item 4).  $\square$

**Lemma 10.** *The interactive reduction  $\Pi_{\text{CCS}} : \text{CCS}(b, \mathcal{L})^K \times \text{CE}(b, \mathcal{L})^k \rightarrow \text{CE}(b, \mathcal{L})^{K+k}$  is **complete** and **public coin**.*

*Proof. Completeness.* Assume the original input tuples belong to relations  $\text{CCS}(b, \mathcal{L})$  (Definition 20) and  $\text{CE}(b, \mathcal{L})$  (Definition 21). We will first argue that the sum-check verifier in step 2 passes. Then, we will argue that the evaluation claim check in step 4 passes. Finally, we will argue that output tuples belong to  $\text{CE}(b, \mathcal{L})^{K+k}$ .

By the definition of relations  $\text{CCS}(b, \mathcal{L})$  (Definition 20) and  $\text{CE}(b, \mathcal{L})$  (Definition 21), we must have that (Item 1), (Item 2), (Item 3), and (Item 4) from Lemma 9 hold. Therefore, we must have that

$$T(C) = \sum_{\vec{x} \in \{0,1\}^{\log m}} Q(\vec{x}, \vec{A}, C).$$

Thus, for any choice of challenges  $\alpha \in \mathbb{K}^{\log m}$  and  $\gamma \in \mathbb{K}$  chosen in step 1,

$$T(\gamma) = \sum_{\vec{x} \in \{0,1\}^{\log m}} Q(\vec{x}, \alpha, \gamma).$$

Thus, by the completeness of the sum-check protocol (Definition 11), we must have that the sum-check verifier (step 2) always passes.

By step 3 and Theorem 10, we must have that

$$\begin{aligned} \text{ct}(y'_i) &= \widetilde{z_i^{b,\text{pad}}}(r') && \text{for all } i \in [K+k], \\ \text{ct}(y'_{i,j}) &= \widetilde{M_j z_i}(r') && \text{for all } i \in [K+k], j \in [t]. \end{aligned}$$

Finally, by the same theorem, we must have that

$$\begin{aligned} \text{cf}(y'_i)_\ell &= \widetilde{\text{cf}(h_i)_\ell}(r') && \text{for all } i \in [K+1, K+k], \ell \in [d], \\ \text{cf}(y'_{i,j})_\ell &= \widetilde{\text{cf}(h_{i,j})_\ell}(r') && \text{for all } i \in [K+1, K+k], j \in [t], \ell \in [d]. \end{aligned}$$

By definition of  $Q(\vec{X})$  in step 2, we must have that

$$Q(r') = E_{\mathbb{K}} + \gamma^{kd} \cdot E_{\mathbb{A}} + \gamma^{kd(t+1)} \cdot \text{eq}(r', \alpha)(F + \gamma^K \cdot N)$$

for values  $F, N, E_{\mathbb{K}}$ , and  $E_{\mathbb{A}}$  derived in step 4. Since the honest sum-check execution yields  $v = Q(r')$ , the verifier check in step 4 passes.

Observe that  $\Pi_{\text{CCS}}$  outputs exactly the original structure  $s$ , commitments  $(c_i)_{i \in [K+k]}$ , vectors  $(z_i)_{i \in [K+k]}$ , and instances  $(x_i)_{i \in [K+k]}$ . Thus, by the definition of  $\text{CCS}(b, \mathcal{L})$ , we must have immediately that every condition in  $\text{CE}(b, \mathcal{L})$  is satisfied for all the  $K+k$  tuples, except that

$$\begin{aligned} y'_i &= \widetilde{h_i}(r') && \text{for all } i \in [K+k], \\ y'_{i,j} &= \widetilde{h_{i,j}}(r') && \text{for all } i \in [K+k], j \in [t]. \end{aligned}$$

However, these are exactly the evaluations computed by the honest prover in step 3. Therefore, the output tuples do belong to  $\text{CE}(b, \mathcal{L})^{K+k}$  as required.

**Public coin.** The sum-check protocol itself is a public-coin protocol. The remaining randomness from the verifier are the challenges  $\alpha \in \mathbb{K}^{\log m}$ ,  $\gamma \in \mathbb{K}$ , which are sampled uniformly at random and sent to the prover.  $\square$

We prove conditions (i) and (ii) of strong interactive reductions (Definition 18).

*Proof.*

**Proof of (i)** By construction, the verifier trivially sets the commitments in the output instance  $u_2$  to be the original commitments  $(c_i)_{i \in [K+k]}$  from the input instance  $u_1$ . Hence, for repeated executions with respect to the same input instance  $u_1$  with output instances  $u_2, u'_2$ , the commitments in these output instances must be the same.

**Proof of (ii)** Consider an arbitrary expected polynomial-time adversary  $(\mathcal{A}, \mathcal{P}^*)$ , such that the relaxed success probability of the adversary  $\epsilon'(\mathcal{A}, \mathcal{P}^*) \geq 1/\text{poly}(\lambda)$  and, letting  $\mathcal{O}_{\text{str}}$  be the oracle from condition (ii) of strong interactive reductions (Definition 18) instantiated with  $\mathcal{P}^*$ , for any uniqueness adversary  $\mathcal{S}_{\text{uniq}}^{\mathcal{O}_{\text{str}}}$  (Definition 16),

$$\Pr \left[ \begin{array}{c} w_2, w'_2 \neq \perp \\ \wedge \\ w_2 \neq w'_2 \end{array} \middle| \begin{array}{l} \text{pp} \leftarrow \mathcal{G}(1^\lambda, \text{sz}) \\ (\mathbf{s}, u_1, \text{st}) \leftarrow \mathcal{A}(\text{pp}) \\ \text{in} \leftarrow (\text{pp}, \mathbf{s}, u_1, \text{st}) \\ (u_2, w_2), (u'_2, w'_2) \leftarrow \mathcal{S}_{\text{uniq}}^{\mathcal{O}_{\text{str}}(\text{in})}(\text{in}) \end{array} \right] \leq \text{negl}(\lambda) \quad (11)$$

then we will show that there exists an expected polynomial-time extractor  $\mathcal{E}$  such that

$$\Pr \left[ (\text{pp}, \mathbf{s}, u_1, w_1) \in \mathcal{R}_1 \middle| \begin{array}{l} \text{pp} \leftarrow \mathcal{G}(1^\lambda, \text{sz}) \\ (\mathbf{s}, u_1, \text{st}) \leftarrow \mathcal{A}(\text{pp}) \\ (\text{pk}, \text{vk}) \leftarrow \mathcal{K}(\text{pp}, \mathbf{s}) \\ w_1 \leftarrow \mathcal{E}(\text{pp}, \mathbf{s}, u_1, \text{st}) \end{array} \right] \geq \epsilon'(\mathcal{A}, \mathcal{P}^*) - \text{negl}(\lambda).$$

Namely, the following extractor  $\mathcal{E}$ ,

$\mathcal{E}(\text{pp}, \mathbf{s}, u_1, \text{st}) \rightarrow w_1$  :

1.  $(\text{pk}, \text{vk}) \leftarrow \mathcal{K}(\text{pp}, \mathbf{s})$ .
2. Simulate  $(u_2, w_2) \leftarrow \langle \mathcal{P}^*, \mathcal{V} \rangle((\text{pk}, \text{vk}), u_1, \text{st})$  with fresh randomness.
3. If  $(\text{pp}, \mathbf{s}, u_2, w_2) \notin \mathcal{R}'_2$ , then output  $\perp$ .
4. Parse  $(z_1, \dots, z_K, z_{K+1}, \dots, z_{K+k}) \leftarrow w_2$ .
5. For all  $i \in [K]$ , assign  $w_i^{\text{CCS}} \leftarrow z_i[n_{L, \text{in}} : ]$ .
6. Output  $w_1 := (w_1^{\text{CCS}}, \dots, w_K^{\text{CCS}}, z_{K+1}, \dots, z_{K+k})$ .

**Extractor runtime.** The extractor makes one call to the expected polynomial-time prover  $\mathcal{P}^*$  and otherwise performs only polynomial-time computations. Hence, it runs in expected polynomial time.

**Extractor success probability.** Let  $\text{in} := (\text{pp}, \mathbf{s}, u_1, \text{st})$  denote everything fixed before the protocol execution, and compute  $(\text{pk}, \text{vk}) \leftarrow \mathcal{K}(\text{pp}, \mathbf{s})$ . Conditioned on  $\text{in}$ , let  $(u_2, w_2)$  be the response from the extractor's interaction, and let  $(u'_2, w'_2)$  be one fresh response from  $\mathcal{O}_{\text{str}}(\text{in})$ :

$$\begin{aligned} (u_2, w_2) &\leftarrow \langle \mathcal{P}^*, \mathcal{V} \rangle((\text{pk}, \text{vk}), u_1, \text{st}), \\ (u'_2, w'_2) &\leftarrow \mathcal{O}_{\text{str}}(\text{in}). \end{aligned}$$

Let  $w_1$  denote the extractor's output when its protocol call returns  $(u_2, w_2)$ . Define the following events:

- $\text{Succ}_1 := \{(\text{pp}, \mathbf{s}, u_2, w_2) \in \mathcal{R}'_2\}$ : relaxed success of the first response.

- $\text{Succ}_2 := \{(\text{pp}, \mathbf{s}, u'_2, w'_2) \in \mathcal{R}'_2\}$ : relaxed success of the fresh response.
- $\text{Err}_1 := \text{Succ}_1 \cap \{(\text{pp}, \mathbf{s}, u_1, w_1) \notin \mathcal{R}_1\}$ : extraction error for the first response.
- $\text{Agree} := \{w'_2 = w_2\}$ : agreement between the two witnesses.<sup>10</sup>

Thus, conditioned on  $\text{in}$ ,  $\text{Succ}_1$  and  $\text{Succ}_2$  each occur with probability

$$p_{\text{in}} := \Pr[\text{Succ}_1 \mid \text{in}] = \Pr[\text{Succ}_2 \mid \text{in}],$$

and  $\text{Err}_1$  occurs with probability

$$a_{\text{in}} := \Pr[\text{Err}_1 \mid \text{in}].$$

Since  $\text{Err}_1 \subseteq \text{Succ}_1$ , we have  $0 \leq a_{\text{in}} \leq p_{\text{in}}$ . By the law of total probability,

$$\epsilon'(\mathcal{A}, \mathcal{P}^*) = \mathbb{E}_{\text{in}}[\Pr[\text{Succ}_1 \mid \text{in}]] = \mathbb{E}_{\text{in}}[p_{\text{in}}]. \quad (12)$$

Consider the following uniqueness adversary:

$\mathcal{S}_{\text{uniq}}^{\mathcal{O}_{\text{str}}(\text{in})}(\text{in})$  :

1. Parse  $(\text{pp}, \mathbf{s}, u_1, \text{st}) \leftarrow \text{in}$  and query  $\mathcal{O}_{\text{str}}(\text{in})$  to obtain  $(u_2, w_2)$ .
2. If  $(\text{pp}, \mathbf{s}, u_2, w_2) \notin \mathcal{R}'_2$ , output  $\perp$ .
3. Parse  $(z_1, \dots, z_K, z_{K+1}, \dots, z_{K+k}) \leftarrow w_2$ . For all  $i \in [K]$ , assign  $w_i^{\text{CCS}} \leftarrow z_i[n_{L_{\text{in}}}]$ , and assign  $w_1 \leftarrow (w_1^{\text{CCS}}, \dots, w_K^{\text{CCS}}, z_{K+1}, \dots, z_{K+k})$ .
4. If  $(\text{pp}, \mathbf{s}, u_1, w_1) \in \mathcal{R}_1$ , output  $\perp$ .
5. Repeatedly query  $\mathcal{O}_{\text{str}}(\text{in})$  until obtaining  $(u'_2, w'_2)$  such that  $(\text{pp}, \mathbf{s}, u'_2, w'_2) \in \mathcal{R}'_2$ .
6. Output  $(u_2, w_2), (u'_2, w'_2)$ .

By construction, the uniqueness adversary reaches step 5 exactly when  $\text{Err}_1$  occurs. Let  $t_{\text{in}}$  be the expected running time of one oracle call together with the relation checks and witness construction, conditioned on  $\text{in}$ . If  $p_{\text{in}} = 0$ , then  $a_{\text{in}} = 0$ , so step 5 is never reached. The expected running time conditioned on  $\text{in}$  is at most  $t_{\text{in}}$ . Otherwise, fresh oracle calls imply that step 5 terminates with probability one and has expected running time  $t_{\text{in}}/p_{\text{in}}$ . Since step 5 is reached with probability  $a_{\text{in}}$  and  $a_{\text{in}} \leq p_{\text{in}}$ , the expected running time conditioned on  $\text{in}$  is at most

$$t_{\text{in}} + \frac{a_{\text{in}}}{p_{\text{in}}} t_{\text{in}}.$$

Since  $\mathcal{P}^*$  is EPT and all other computations run in polynomial time,  $\mathbb{E}_{\text{in}}[t_{\text{in}}]$  is polynomial. Thus,  $\mathcal{S}_{\text{uniq}}^{\mathcal{O}_{\text{str}}}$  runs in expected polynomial time, and (11) gives

$$\epsilon_{\text{uniq}} := \Pr \left[ \begin{array}{c} w_2, w'_2 \neq \perp \\ \wedge w_2 \neq w'_2 \end{array} \middle| \begin{array}{l} \text{pp} \leftarrow \mathcal{G}(1^\lambda, \text{sz}) \\ (\mathbf{s}, u_1, \text{st}) \leftarrow \mathcal{A}(\text{pp}) \\ \text{in} \leftarrow (\text{pp}, \mathbf{s}, u_1, \text{st}) \\ (u_2, w_2), (u'_2, w'_2) \leftarrow \mathcal{S}_{\text{uniq}}^{\mathcal{O}_{\text{str}}(\text{in})}(\text{in}) \end{array} \right] \leq \text{negl}(\lambda).$$

To bound the probability that  $\mathcal{S}_{\text{uniq}}$  returns equal witnesses, fix  $\text{in}$  and a first response  $(u_2, w_2)$  for which  $\text{Err}_1$  occurs. Consider one call  $(u'_2, w'_2) \leftarrow \mathcal{O}_{\text{str}}(\text{in})$  made in step 5. We first bound  $\text{Succ}_2 \cap \text{Agree}$  for this call. By definition of  $\mathcal{R}'_2 = \text{CE}(q/2, \mathcal{L})^{K+k}$ , for all  $i \in [K+k]$ , define

$$\mathbf{x}_i := \text{Pack}_{\mathbb{L}}(x_i), \quad \mathbf{z}_i := \text{Pack}_{\mathbb{L}}(z_i).$$

We must have

$$\mathbf{x}_i = \mathcal{L}_{\text{in}}(\mathbf{z}_i) \quad \text{and} \quad c_i = \mathcal{L}(\mathbf{z}_i). \quad (13)$$

By the definitions of  $\text{Pack}_{\mathbb{L}}$  (Definition 14) and  $\mathcal{L}_{\text{in}}$  (Definition 22), together with  $n_{\mathbb{F}, \text{in}} = \tau n_{L, \text{in}} = dn_{R, \text{in}}$  (Definition 1), Equation (13) also implies that for all  $i \in [K+k]$ , the first  $n_{L, \text{in}}$  entries of  $z_i$  are equal to  $x_i$  (from input instance  $u_1$ ).

<sup>10</sup> Here, in  $\text{Agree} := \{w'_2 = w_2\}$ ,  $(u'_2, w'_2)$  denotes the response from the single call to  $\mathcal{O}_{\text{str}}(\text{in})$  above.

Since  $\text{Err}_1$  occurs for the fixed first response,  $(\text{pp}, \mathbf{s}, u_1, w_1) \notin \mathcal{R}_1$ . Recall that  $\mathcal{R}_1 := \text{CCS}(b, \mathcal{L})^K \times \text{CE}(b, \mathcal{L})^k$ . By Equation (13), we know that all conditions in  $\text{CCS}(b, \mathcal{L})$  and  $\text{CE}(b, \mathcal{L})$ , except for the norm bound  $\|z^b\|_\infty < b$ , the evaluations  $y = \widetilde{h}(r)$  and  $y_j = \widetilde{h}_j(r)$  for  $j \in [t]$ , or the CCS requirements  $f(\widetilde{M}_1 z, \dots, \widetilde{M}_t z) \in \text{ZS}_{\log m}(\mathbb{L})$ , are satisfied. Thus, using notation from Lemma 9,  $(\text{pp}, \mathbf{s}, u_1, w_1) \notin \mathcal{R}_1$  implies that either:

1. there exists an  $i \in [K]$ ,  $f(\widetilde{M}_1 z_i, \dots, \widetilde{M}_t z_i) \notin \text{ZS}_{\log m}(\mathbb{L})$ ,
2. OR there exists an  $i \in [K+k]$ ,  $\|z_i^b\|_\infty \geq b$ ,
3. OR there exists an  $i \in [K+1, K+k]$ ,  $y_i \neq \widetilde{h}_i(r)$ ,
4. OR there exists an  $i \in [K+1, K+k]$  and  $j \in [t]$ ,  $y_{i,j} \neq \widetilde{h}_{i,j}(r)$ .

By Lemma 9, we must have

$$T(C) \neq \sum_{\vec{x} \in \{0,1\}^{\log m}} Q(\vec{x}, \vec{A}, C).$$

Now focus on the oracle call, whose randomness is independent of the fixed first response. Write  $\alpha', \gamma', r', v', y'_i$ , and  $y'_{i,j}$  for its challenges, sum-check point and value, and claimed evaluations.

On **Agree**,  $w_2 = w'_2$ . Consequently, the oracle call uses the same vectors  $z_i$ , and hence the same  $h_i$  and  $h_{i,j}$ , as the fixed first response.

On **Succ<sub>2</sub>**, the oracle call has relaxed success, so  $(\text{pp}, \mathbf{s}, u'_2, w'_2) \in \mathcal{R}'_2$ . Therefore, the verifier did not abort in protocol steps 2 and 4, and the prover's claimed evaluations in protocol step 3 are true. Namely,

$$\begin{aligned} y'_i &= \widetilde{h}_i(r') && \text{for all } i \in [K+k], \\ y'_{i,j} &= \widetilde{h}_{i,j}(r') && \text{for all } i \in [K+k], j \in [t]. \end{aligned}$$

Thus, by Theorem 10, the definition of  $Q$ , and the construction of the verifier's checks in protocol step 4, we must have that the oracle call's sum-check evaluation claim  $v' = Q(r', \alpha', \gamma')$  is true.

Thus, in order for the sum-check verifier to have passed in the oracle call, either the adversary  $\mathcal{P}^*$

- succeeded in the sum-check protocol, despite  $T(\gamma') \neq \sum_{\vec{x} \in \{0,1\}^{\log m}} Q(\vec{x}, \alpha', \gamma')$  (in other words, violated the soundness of the sum-check protocol)
- OR the non-zero polynomial

$$P(C, \vec{A}) := T(C) - \sum_{\vec{x} \in \{0,1\}^{\log m}} Q(\vec{x}, \vec{A}, C)$$

evaluated to zero on the random point  $(\gamma', \alpha') \in \mathbb{K}^{1+\log m}$  of the oracle call.

By the soundness error of the sum-check protocol (Definition 11), the first event occurs with probability at most  $\epsilon_{\text{SC}} := \max(u, 2b, 2) \cdot \log m / |\mathbb{K}|$ , where  $u$  is the strict total-degree bound for  $f$ ,  $b$  is the norm bound, and 2 comes from  $\text{Eval}_{\mathbb{K}}(\vec{X})$  and  $\text{Eval}_{\mathbb{A}}(\vec{X})$ . By the Schwartz–Zippel lemma (Lemma 2), the second event occurs with probability at most  $\epsilon_{\text{SZ}} := (kd(t+1) + 2K + k - 1 + \log m) / |\mathbb{K}|$ . By Definition 1,  $1/|\mathbb{K}| = \text{negl}(\lambda)$ . Since  $u, b, t, k, K, d$ , and  $\log m$  are polynomially bounded in  $\lambda$ , both  $\epsilon_{\text{SC}}$  and  $\epsilon_{\text{SZ}}$  are negligible. For the fixed  $\text{in}$  and first response for which  $\text{Err}_1$  occurs, the union bound over the randomness of the oracle call gives

$$\Pr[\text{Succ}_2 \cap \text{Agree}] \leq \epsilon_{\text{SC}} + \epsilon_{\text{SZ}}. \quad (14)$$

Let  $\epsilon_{\text{test}} := \epsilon_{\text{SC}} + \epsilon_{\text{SZ}}$ . For fixed  $\text{in}$  and first response in  $\text{Err}_1$ , the response returned by step 5<sup>11</sup> is distributed as one oracle call conditioned on  $\text{Succ}_2$ . By Equation (14) and conditional probability,

$$\Pr[\text{Agree} \mid \text{Succ}_2] = \frac{\Pr[\text{Succ}_2 \cap \text{Agree}]}{\Pr[\text{Succ}_2]} \leq \frac{\epsilon_{\text{test}}}{p_{\text{in}}}.$$

<sup>11</sup> Hereafter, in  $\text{Agree} := \{w'_2 = w_2\}$ ,  $(u'_2, w'_2)$  denotes the response returned by step 5.

If  $p_{\text{in}} = 0$ , then  $a_{\text{in}} = 0$ , so step 5 is never reached and

$$\Pr[\text{Err}_1 \cap \text{Agree} \mid \text{in}] = 0.$$

If  $p_{\text{in}} > 0$ , then

$$\Pr[\text{Err}_1 \cap \text{Agree} \mid \text{in}] \leq a_{\text{in}} \frac{\epsilon_{\text{test}}}{p_{\text{in}}} \leq \epsilon_{\text{test}},$$

since  $a_{\text{in}} \leq p_{\text{in}}$ . Thus, the bound holds for every  $\text{in}$ . Averaging over  $\text{in}$  gives

$$\Pr[\text{Err}_1 \cap \text{Agree}] \leq \epsilon_{\text{test}}.$$

By construction,  $\mathcal{S}_{\text{uniq}}$  reaches step 5 exactly when  $\text{Err}_1$  occurs, and step 5 terminates with probability one. Its returned witnesses either agree or differ. Therefore,

$$\Pr[\text{Err}_1] = \Pr[\text{Err}_1 \cap \text{Agree}] + \Pr[\text{Err}_1 \cap \text{Agree}^c].$$

The first term is at most  $\epsilon_{\text{test}}$  by the bound above. The second term equals  $\epsilon_{\text{uniq}}$ , since the uniqueness adversary returns different witnesses exactly when  $\text{Err}_1 \cap \text{Agree}^c$  occurs. Hence,

$$\mathbb{E}_{\text{in}}[a_{\text{in}}] = \Pr[\text{Err}_1] \leq \epsilon_{\text{test}} + \epsilon_{\text{uniq}}.$$

For every fixed  $\text{in}$ , relaxed success occurs with probability  $p_{\text{in}}$  and extraction error occurs with probability  $a_{\text{in}}$ . The extractor outputs a valid witness exactly when relaxed success occurs without an extraction error, namely on  $\text{Succ}_1 \setminus \text{Err}_1$ . Since  $\text{Err}_1 \subseteq \text{Succ}_1$ , this event has probability  $p_{\text{in}} - a_{\text{in}}$ . Consequently, by the law of total probability, linearity of expectation, and Equation (12),

$$\begin{aligned} \Pr[(\text{pp}, \mathbf{s}, u_1, w_1) \in \mathcal{R}_1] &= \Pr[\text{Succ}_1 \setminus \text{Err}_1] \\ &= \mathbb{E}_{\text{in}}[p_{\text{in}} - a_{\text{in}}] \\ &= \mathbb{E}_{\text{in}}[p_{\text{in}}] - \mathbb{E}_{\text{in}}[a_{\text{in}}] \\ &= \epsilon'(\mathcal{A}, \mathcal{P}^*) - \mathbb{E}_{\text{in}}[a_{\text{in}}] \\ &\geq \epsilon'(\mathcal{A}, \mathcal{P}^*) - \epsilon_{\text{test}} - \epsilon_{\text{uniq}}. \end{aligned}$$

□

## B.3 Proofs for $\Pi_{\text{RLC}}$

**Lemma 11.** *The interactive reduction  $\Pi_{\text{RLC}} : \text{CE}(b, \mathcal{L})^{K+k} \rightarrow \text{CE}(B, \mathcal{L})$  is **complete** and **public coin**.*

*Proof. Completeness.* By definition of  $\text{CE}(b, \mathcal{L})$  (Definition 21), the input tuples satisfy the conditions in Theorem 11 instantiated with  $(\mathbb{B}, M, z_i, y_i) = (\mathbb{F}, \text{Pad}, z_i^b, y_i)$  and, for every  $j \in [t]$ , with  $(\mathbb{B}, M, z_i, y_i) = (\mathbb{L}, M_j, z_i, y_{i,j})$ . Since the protocol decodes  $x$  and  $z$  from their packed linear combinations, we have

$$\begin{aligned} c &= \mathcal{L}(z), \quad x = \mathcal{L}_{\text{in}}(z), \\ y &= \tilde{h}(r), \quad y_j = \tilde{h}_j(r) \quad \text{for every } j \in [t]. \end{aligned}$$

Here,  $h := \text{Emb}_{\mathbb{F}}(\text{Trans}_{\mathbb{F}}(\text{Pad})) \cdot z$  and  $h_j := \text{Emb}_{\mathbb{L}}(\text{Trans}_{\mathbb{L}}(M_j)) \cdot z$  for every  $j \in [t]$ . Therefore, the output tuple satisfies every requirement of  $\text{CE}(B, \mathcal{L})$  except  $\|z^b\|_{\infty} < B = b^k$ .

However, we show that this bound follows from the expansion factor  $T$  of  $\mathcal{C}$  chosen in Definition 22:

$$\begin{aligned} \|z^b\|_{\infty} &= \|z\|_{\infty} = \left\| \sum_{i=1}^{K+k} \rho_i z_i \right\|_{\infty} \\ &\leq \sum_{i=1}^{K+k} \|\rho_i z_i\|_{\infty} \leq \sum_{i=1}^{K+k} T \|z_i\|_{\infty} \\ &= \sum_{i=1}^{K+k} T \|z_i^b\|_{\infty} \leq (K+k)T(b-1) < B. \end{aligned}$$

The equalities  $\|z^b\|_\infty = \|z\|_\infty$  and  $\|z_i^b\|_\infty = \|z_i\|_\infty$  follow from the definitions of  $\text{Pack}_F$  and the norm. The first inequality is the triangle inequality, the second inequality follows from the expansion factor of  $\mathcal{C}$  being  $T$ , the third inequality follows from the definition of  $\text{CE}(b, \mathcal{L})$ , which enforces a norm bound of  $b$ , and the last inequality holds by assumption (Definition 22). Hence, the output tuple must belong to  $\text{CE}(B, \mathcal{L})$ .

**Public coin.** The verifier's randomness consists of challenges  $\rho_1, \dots, \rho_{k+K}$ , which are sampled uniformly at random from  $\mathcal{C}$  and sent to the prover.  $\square$

We prove the conditions of weak interactive reductions (Definition 17).

*Proof.* Consider an arbitrary expected-polynomial time adversary  $(\mathcal{A}, \mathcal{P}^*)$  for  $\Pi_{\text{RLC}}$ . First, we can construct an adversary and verification function for Theorem 7,

$A_{(\text{pp}, \mathbf{s}, u_1, \text{st})}(\vec{c})$  :

1. If  $u_1 = \perp$ , output  $\perp$ .
2. Execute encoder  $(\text{pk}, \text{vk}) \leftarrow \mathcal{K}(\text{pp}, \mathbf{s})$ .
3. Simulate  $(u_2, w_2) \leftarrow \langle \mathcal{P}^*(\text{pk}, u_1, \text{st}), \mathcal{V}(\text{vk}, u_1) \rangle$  with verifier randomness  $\vec{c}$ .
4. Output  $w_2$

$V_{(\text{pp}, \mathbf{s}, u_1, \text{st})}(\vec{c}, w_2) \rightarrow \{0, 1\}$  :

1. If  $u_1 = \perp$ , output reject.
2. Execute encoder  $(\text{pk}, \text{vk}) \leftarrow \mathcal{K}(\text{pp}, \mathbf{s})$ .
3. Simulate  $(u_2, \cdot) \leftarrow \langle \mathcal{P}^*(\text{pk}, u_1, \text{st}), \mathcal{V}(\text{vk}, u_1) \rangle$  with verifier randomness  $\vec{c}$ .
4. Output accept if and only if  $(u_2, w_2) \in \text{CE}(B, \mathcal{L})$ .

Let  $E_{(\text{pp}, \mathbf{s}, u_1, \text{st})}$  be the corresponding extractor from Theorem 7. We define  $E(\text{pp}, \mathbf{s}, u_1, \text{st})$  as the trivial algorithm that executes  $E_{(\text{pp}, \mathbf{s}, u_1, \text{st})}$  by simulating calls to  $A_{(\text{pp}, \mathbf{s}, u_1, \text{st})}$ . Without loss of generality,  $E$  outputs  $\perp$  whenever its output does not satisfy the three conditions in Theorem 7. We construct an extractor for adversary  $(\mathcal{A}, \mathcal{P}^*)$  as follows:

$\mathcal{E}(\text{pp}, \mathbf{s}, u_1, \text{st})$  :

1.  $\text{result} \leftarrow E(\text{pp}, \mathbf{s}, u_1, \text{st})$ .
2. If  $u_1 = \perp$  or  $\text{result} = \perp$ , output  $\perp$ .
3. Parse  $(\vec{c}, w'), (\vec{c}_1, w'_1), \dots, (\vec{c}_{K+k}, w'_{K+k}) \leftarrow \text{result}$ .
4. Parse  $z \leftarrow w'$  and  $\rho_1, \dots, \rho_{K+k} \leftarrow \vec{c}$ .
5. For  $i \in [K+k]$ ,
  - (a) Parse  $z^{(i)} \leftarrow w'_i$  and  $\rho_1^{(i)}, \dots, \rho_{K+k}^{(i)} \leftarrow \vec{c}_i$ .
  - (b) Assign  $z_i \leftarrow (\rho_i - \rho_i^{(i)})^{-1} \cdot (z - z^{(i)})$ .
  - (c) Assign  $z_i \leftarrow \text{Pack}_L^{-1}(z_i)$ .
6. Parse  $(c_i, x_i, r, y_i, \{y_{i,j}\}_{j \in [t]})_{i \in [K+k]} \leftarrow u_1$ .
7. Output  $w_1 := (z_i)_{i \in [K+k]}$  if and only if
$$(\mathbf{s}; c_i, x_i, r, y_i, \{y_{i,j}\}_{j \in [t]}; z_i)_{i \in [K+k]} \in \text{CE}(q/2, \mathcal{L})^{K+k}$$

**Extractor runtime.** By Theorem 7, we are guaranteed  $E_{(\text{pp}, \mathbf{s}, u_1, \text{st})}$  makes in expectation at most  $(K+k)+1$  calls to  $A_{(\text{pp}, \mathbf{s}, u_1, \text{st})}$ . Hence, our overall extractor  $\mathcal{E}$  runs in expected polynomial time.

**Extractor success probability.** If  $u_1 = \perp$ , then  $A_{(\text{pp}, \mathbf{s}, u_1, \text{st})}$  outputs  $\perp$  and  $V_{(\text{pp}, \mathbf{s}, u_1, \text{st})}$  rejects, so both the conditional success probability and  $\Pr[\text{result} \neq \perp \mid \text{pp}, \mathbf{s}, u_1, \text{st}]$  are zero. For every fixed  $(\text{pp}, \mathbf{s}, u_1, \text{st})$ , by Theorem 7, we are guaranteed that  $E(\text{pp}, \mathbf{s}, u_1, \text{st})$  outputs  $(K+k)+1$  pairs  $(\vec{c}, w'), (\vec{c}_1, w'_1), \dots, (\vec{c}_{K+k}, w'_{K+k})$  such that

- $V_{(\text{pp}, \mathbf{s}, u_1, \text{st})}(\vec{c}, w') = 1$ ,
- for all  $i \in [K+k]$ ,  $V_{(\text{pp}, \mathbf{s}, u_1, \text{st})}(\vec{c}_i, w'_i) = 1$ , and

- $(\vec{c}, \vec{c}_1, \dots, \vec{c}_{K+k}) \in \text{SS}(\mathcal{C}, K+k)$

with probability at least  $\epsilon^{V_{(\text{pp}, \mathbf{s}, u_1, \text{st})}(A_{(\text{pp}, \mathbf{s}, u_1, \text{st})})} - \frac{K+k}{|\mathcal{C}|}$ . Hence, for every fixed  $(\text{pp}, \mathbf{s}, u_1, \text{st})$ ,

$$\Pr[\text{result} \neq \perp \mid \text{pp}, \mathbf{s}, u_1, \text{st}] \geq \epsilon^{V_{(\text{pp}, \mathbf{s}, u_1, \text{st})}(A_{(\text{pp}, \mathbf{s}, u_1, \text{st})})} - \frac{K+k}{|\mathcal{C}|}.$$

Since the verifier's output  $u_2$  in  $\Pi_{\text{RLC}}$  is determined by  $u_1$  and the challenge vector  $\vec{c}$ , for every fixed  $(\text{pp}, \mathbf{s}, u_1, \text{st})$ ,

$$\epsilon^{V_{(\text{pp}, \mathbf{s}, u_1, \text{st})}(A_{(\text{pp}, \mathbf{s}, u_1, \text{st})})} = \Pr[(\text{pp}, \mathbf{s}, u_2, w_2) \in \text{CE}(B, \mathcal{L}) \mid \text{pp}, \mathbf{s}, u_1, \text{st}].$$

By definition,  $\epsilon(\mathcal{A}, \mathcal{P}^*)$  is the overall probability that the output belongs to  $\text{CE}(B, \mathcal{L})$ . Therefore, by the law of total probability and the fixed-tuple equality above,

$$\begin{aligned} \epsilon(\mathcal{A}, \mathcal{P}^*) &= \Pr[(\text{pp}, \mathbf{s}, u_2, w_2) \in \text{CE}(B, \mathcal{L})] \\ &= \mathbb{E}_{(\text{pp}, \mathbf{s}, u_1, \text{st})} [\Pr[(\text{pp}, \mathbf{s}, u_2, w_2) \in \text{CE}(B, \mathcal{L}) \mid \text{pp}, \mathbf{s}, u_1, \text{st}]] \\ &= \mathbb{E}_{(\text{pp}, \mathbf{s}, u_1, \text{st})} [\epsilon^{V_{(\text{pp}, \mathbf{s}, u_1, \text{st})}(A_{(\text{pp}, \mathbf{s}, u_1, \text{st})})}]. \end{aligned} \quad (15)$$

Averaging this pointwise guarantee over the distribution of  $(\text{pp}, \mathbf{s}, u_1, \text{st})$  induced by  $\text{pp} \leftarrow \mathcal{G}(1^\lambda, \mathbf{sz})$  and  $(\mathbf{s}, u_1, \text{st}) \leftarrow \mathcal{A}(\text{pp})$  gives

$$\begin{aligned} \Pr[\text{result} \neq \perp] &= \mathbb{E}_{(\text{pp}, \mathbf{s}, u_1, \text{st})} [\Pr[\text{result} \neq \perp \mid \text{pp}, \mathbf{s}, u_1, \text{st}]] \\ &\geq \mathbb{E}_{(\text{pp}, \mathbf{s}, u_1, \text{st})} \left[ \epsilon^{V_{(\text{pp}, \mathbf{s}, u_1, \text{st})}(A_{(\text{pp}, \mathbf{s}, u_1, \text{st})})} - \frac{K+k}{|\mathcal{C}|} \right] \\ &= \mathbb{E}_{(\text{pp}, \mathbf{s}, u_1, \text{st})} \left[ \epsilon^{V_{(\text{pp}, \mathbf{s}, u_1, \text{st})}(A_{(\text{pp}, \mathbf{s}, u_1, \text{st})})} \right] - \frac{K+k}{|\mathcal{C}|} \\ &= \epsilon(\mathcal{A}, \mathcal{P}^*) - \frac{K+k}{|\mathcal{C}|}. \end{aligned} \quad (16)$$

The first equality follows from the law of total probability, the inequality from the fixed-tuple guarantee above, and the next equality from linearity of expectation. The final equality follows from Equation (15). Assume that  $\text{result} \neq \perp$ . Since  $V_{(\text{pp}, \mathbf{s}, u_1, \text{st})}(\vec{c}, w') = 1$  and  $V_{(\text{pp}, \mathbf{s}, u_1, \text{st})}$  executes  $\Pi_{\text{RLC}}$ 's  $\mathcal{V}$  and outputs accept if and only if  $(u_2, w_2) \in \text{CE}(B, \mathcal{L})$ , we must have for  $\mathbf{x} := \sum_{i=1}^{K+k} \rho_i \mathbf{x}_i$  and  $x := \text{Pack}_{\mathbb{L}}^{-1}(\mathbf{x})$  that

$$\left( \mathbf{s}; \begin{array}{l} c := \sum_{i=1}^{K+k} \rho_i c_i, \\ x, r, y := \sum_{i=1}^{K+k} \rho_i y_i, \quad ; \quad z \\ \{y_j := \sum_{i=1}^{K+k} \rho_i y_{i,j}\}_{j \in [t]} \end{array} \right) \in \text{CE}(B, \mathcal{L}) \quad (17)$$

where  $(c_i, x_i, r, y_i, \{y_{i,j}\}_j)_i$  are the instance elements in  $u_1$  (parsed in step 6) and  $z \leftarrow w'$  and  $(\rho_i)_i \leftarrow \vec{c}$  are the elements parsed in step 4.

For all  $i \in [K+k]$ , we will make a similar argument to the one directly above. Namely, since

$$V_{(\text{pp}, \mathbf{s}, u_1, \text{st})}(\vec{c}_i, w'_i) = 1,$$

we must have, for

$$\mathbf{x}^{(i)} := \sum_{a=1}^{K+k} \rho_a^{(i)} \mathbf{x}_a, \quad x^{(i)} := \text{Pack}_{\mathbb{L}}^{-1}(\mathbf{x}^{(i)}),$$

that

$$\left( \mathbf{s}; \begin{array}{l} c^{(i)} := \sum_{a=1}^{K+k} \rho_a^{(i)} c_a, \\ x^{(i)}, r, y^{(i)} := \sum_{a=1}^{K+k} \rho_a^{(i)} y_a, \quad ; \quad z^{(i)} \\ \{y_j^{(i)} := \sum_{a=1}^{K+k} \rho_a^{(i)} y_{a,j}\}_{j \in [t]} \end{array} \right) \in \text{CE}(B, \mathcal{L}) \quad (18)$$

where  $(c_i, x_i, r, y_i, \{y_{i,j}\}_j)_i$  are in  $u_1$  and  $z^{(i)} \leftarrow w'_i$  and  $(\rho_a^{(i)})_a \leftarrow \vec{c}_i$  are the elements parsed in step 5a. By definition of  $\text{CE}(B, \mathcal{L})$  (Definition 21), we must have

$$c = \mathcal{L}(\mathbf{z}), \quad c^{(i)} = \mathcal{L}(\mathbf{z}^{(i)}), \quad \mathbf{x} = \mathcal{L}_{\text{in}}(\mathbf{z}), \quad \mathbf{x}^{(i)} = \mathcal{L}_{\text{in}}(\mathbf{z}^{(i)}) \quad (19)$$

Since  $(\vec{c}, \vec{c}_1, \dots, \vec{c}_{K+k}) \in \text{SS}(\mathcal{C}, K+k)$ , we must have for all  $i \in [K+k]$  that

$$(\rho_1, \dots, \rho_{K+k}) \equiv_i (\rho_1^{(i)}, \dots, \rho_{K+k}^{(i)}) \quad (20)$$

which means the challenges differ only on index  $i$ . By definition of strong sampling set (Definition 6), we must have  $(\rho_i - \rho_i^{(i)})$  is invertible for all  $i \in [K+k]$ .

Thus, by Equation (19) and Equation (20), we have for all  $i \in [K+k]$ ,

$$\begin{aligned} c - c^{(i)} &= \mathcal{L}(\mathbf{z}) - \mathcal{L}(\mathbf{z}^{(i)}) \\ \mathbf{x} - \mathbf{x}^{(i)} &= \mathcal{L}_{\text{in}}(\mathbf{z}) - \mathcal{L}_{\text{in}}(\mathbf{z}^{(i)}) \\ \sum_{a=1}^{K+k} \rho_a c_a - \sum_{a=1}^{K+k} \rho_a^{(i)} c_a &= \mathcal{L}(\mathbf{z}) - \mathcal{L}(\mathbf{z}^{(i)}), \\ \sum_{a=1}^{K+k} \rho_a \mathbf{x}_a - \sum_{a=1}^{K+k} \rho_a^{(i)} \mathbf{x}_a &= \mathcal{L}_{\text{in}}(\mathbf{z}) - \mathcal{L}_{\text{in}}(\mathbf{z}^{(i)}) \end{aligned} \quad (21)$$

$$\begin{aligned} (\rho_i - \rho_i^{(i)}) \cdot c_i &= \mathcal{L}(\mathbf{z}) - \mathcal{L}(\mathbf{z}^{(i)}), \\ (\rho_i - \rho_i^{(i)}) \cdot \mathbf{x}_i &= \mathcal{L}_{\text{in}}(\mathbf{z}) - \mathcal{L}_{\text{in}}(\mathbf{z}^{(i)}) \end{aligned} \quad (22)$$

$$\begin{aligned} c_i &= \mathcal{L}\left((\rho_i - \rho_i^{(i)})^{-1} \cdot (\mathbf{z} - \mathbf{z}^{(i)})\right), \\ \mathbf{x}_i &= \mathcal{L}_{\text{in}}\left((\rho_i - \rho_i^{(i)})^{-1} \cdot (\mathbf{z} - \mathbf{z}^{(i)})\right) \end{aligned} \quad (23)$$

$$c_i = \mathcal{L}(\mathbf{z}_i), \quad \mathbf{x}_i = \mathcal{L}_{\text{in}}(\mathbf{z}_i) \quad (24)$$

where equation (21) follows from (17), (18), and Equation (19). Equation (22) follows from the equivalence in Equation (20). Equation (23) follows from  $\mathcal{L}, \mathcal{L}_{\text{in}}$  being  $\mathbb{R}_{\mathbb{F}}$ -module homomorphisms and  $\mathcal{C}$  being a strong sampling set (Definition 6) which because  $\rho_i \neq \rho_i^{(i)}$  (guaranteed by (20)) means  $\rho_i - \rho_i^{(i)}$  is invertible. Equation (24) follows from the packed extraction in step 5b and the subsequent assignment  $z_i := \text{Pack}_{\mathbb{L}}^{-1}(\mathbf{z}_i)$ .

We make a similar argument for the evaluations. Define

$$\begin{aligned} h &:= \text{Emb}_{\mathbb{F}}(\text{Trans}_{\mathbb{F}}(\text{Pad})) \cdot \mathbf{z}, & h^{(i)} &:= \text{Emb}_{\mathbb{F}}(\text{Trans}_{\mathbb{F}}(\text{Pad})) \cdot \mathbf{z}^{(i)}, \\ h_j &:= \text{Emb}_{\mathbb{L}}(\text{Trans}_{\mathbb{L}}(M_j)) \cdot \mathbf{z}, & h_j^{(i)} &:= \text{Emb}_{\mathbb{L}}(\text{Trans}_{\mathbb{L}}(M_j)) \cdot \mathbf{z}^{(i)} \quad \text{for every } j \in [t]. \end{aligned}$$

In particular, by the definition of  $\text{CE}(B, \mathcal{L})$  (Definition 21), Equation (17), and Equation (18), we must have

$$y = \tilde{h}(r), \quad y^{(i)} = \tilde{h}^{(i)}(r), \quad y_j = \tilde{h}_j(r), \quad y_j^{(i)} = \tilde{h}_j^{(i)}(r) \quad \text{for every } j \in [t]. \quad (25)$$

Thus, we must have for all  $i \in [K+k]$ ,

$$y - y^{(i)} = \tilde{h}(r) - \tilde{h}^{(i)}(r) \quad (26)$$

$$\sum_{a=1}^{K+k} \rho_a y_a - \sum_{a=1}^{K+k} \rho_a^{(i)} y_a = \overline{\text{Emb}_{\mathbb{F}}(\text{Trans}_{\mathbb{F}}(\text{Pad})) \cdot (\mathbf{z} - \mathbf{z}^{(i)})(r)} \quad (27)$$

$$(\rho_i - \rho_i^{(i)}) \cdot y_i = \overline{\text{Emb}_{\mathbb{F}}(\text{Trans}_{\mathbb{F}}(\text{Pad})) \cdot (\mathbf{z} - \mathbf{z}^{(i)})(r)} \quad (28)$$

$$y_i = \overline{\text{Emb}_{\mathbb{F}}(\text{Trans}_{\mathbb{F}}(\text{Pad})) \cdot \left((\rho_i - \rho_i^{(i)})^{-1} \cdot (\mathbf{z} - \mathbf{z}^{(i)})\right)(r)} \quad (29)$$

$$= \widetilde{h_i}(r)$$

where Equation (26) follows from Equation (25), Equation (27) follows from Equation (17) and Equation (18), the definitions of  $h$  and  $h^{(i)}$ , and the linearity of multilinear evaluation, Equation (28) follows from the equivalence (20), and (29) follows from  $\mathcal{C}$  being a strong sampling set (Definition 6) which because  $\rho_i \neq \rho_i^{(i)}$  (guaranteed by (20)) means  $\rho_i - \rho_i^{(i)}$  is invertible, together with the  $\mathbb{R}_F$ -linearity of the transform and multilinear evaluation. The final equality follows from the construction of  $\mathbf{z}_i$  and the definition of  $h_i$ .

Thus, we must have for all  $i \in [K + k]$  and  $j \in [t]$ ,

$$y_j - y_j^{(i)} = \widetilde{h_j}(r) - \widetilde{h_j^{(i)}}(r) \quad (30)$$

$$\sum_{a=1}^{K+k} \rho_a y_{a,j} - \sum_{a=1}^{K+k} \rho_a^{(i)} y_{a,j} = \overline{\text{Emb}_{\mathbb{L}}(\text{Trans}_{\mathbb{L}}(M_j)) \cdot (\mathbf{z} - \mathbf{z}^{(i)})(r)} \quad (31)$$

$$(\rho_i - \rho_i^{(i)}) \cdot y_{i,j} = \overline{\text{Emb}_{\mathbb{L}}(\text{Trans}_{\mathbb{L}}(M_j)) \cdot (\mathbf{z} - \mathbf{z}^{(i)})(r)} \quad (32)$$

$$\begin{aligned} y_{i,j} &= \overline{\text{Emb}_{\mathbb{L}}(\text{Trans}_{\mathbb{L}}(M_j)) \cdot \left( (\rho_i - \rho_i^{(i)})^{-1} \cdot (\mathbf{z} - \mathbf{z}^{(i)}) \right)(r)} \\ &= \widetilde{h_{i,j}}(r) \end{aligned} \quad (33)$$

where Equation (30) follows from Equation (25), Equation (31) follows from Equation (17) and Equation (18), the definitions of  $h_j$  and  $h_j^{(i)}$ , and the linearity of multilinear evaluation, Equation (32) follows from the equivalence (20), and (33) follows from  $\mathcal{C}$  being a strong sampling set (Definition 6) which because  $\rho_i \neq \rho_i^{(i)}$  (guaranteed by (20)) means  $\rho_i - \rho_i^{(i)}$  is invertible, together with the  $\mathbb{R}_F$ -linearity of matrix multiplication and multilinear evaluation. The final equality follows from the construction of  $\mathbf{z}_i$  and the definition of  $h_{i,j}$ .

Therefore, by Equation (16), Equation (24), Equation (29), and Equation (33), we must have with probability at least  $\epsilon(\mathcal{A}, \mathcal{P}^*) - (K + k)/|\mathcal{C}|$ , the extractor outputs witness elements  $z_1, \dots, z_{K+k}$  such that

$$(\mathbf{s}; c_i, x_i, r, y_i, \{y_{i,j}\}_{j \in [t]}; z_i)_{i \in [K+k]} \in \text{CE}(q/2, \mathcal{L})^{K+k},$$

The norm bound  $q/2$  is trivial because every coordinate of  $z_i^b \in \mathbb{F}^{n_F}$  has norm at most  $(q - 1)/2 < q/2$ . By Definition 22,  $1/|\mathcal{C}| = \text{negl}(\lambda)$ , so this probability is at least  $\epsilon(\mathcal{A}, \mathcal{P}^*) - \text{negl}(\lambda)$ . Thus, the extractor satisfies condition (i) of weak interactive reductions (Definition 17).

Now, assume that  $\mathcal{A} := (\mathcal{B}, \mathcal{B}')$  such that

$$\Pr \left[ \begin{array}{c} u_1, u'_1 \neq \perp \\ \Downarrow \\ \phi(u_1) = \phi(u'_1) \end{array} \middle| \begin{array}{l} \mathbf{pp} \leftarrow \mathcal{G}(1^\lambda, \mathbf{sz}) \\ (\mathbf{s}, \mathbf{st}^*) \leftarrow \mathcal{B}(\mathbf{pp}) \\ (u_1, \mathbf{st}) \leftarrow \mathcal{B}'(\mathbf{st}^*) \\ (u'_1, \mathbf{st}') \leftarrow \mathcal{B}'(\mathbf{st}^*) \end{array} \right] = 1 \quad (34)$$

Let  $\mathcal{O}_{\text{weak}}$  be the oracle from condition (ii) of weak interactive reductions (Definition 17), instantiated with  $\mathcal{B}'$  and  $\mathcal{E}$ . Consider any uniqueness adversary  $\mathcal{W}_{\text{unq}}^{\mathcal{O}_{\text{weak}}}$ . We will show that

$$\Pr \left[ \begin{array}{c} w_1, w'_1 \neq \perp \\ \wedge w_1 \neq w'_1 \end{array} \middle| \begin{array}{l} \mathbf{pp} \leftarrow \mathcal{G}(1^\lambda, \mathbf{sz}) \\ (\mathbf{s}, \mathbf{st}^*) \leftarrow \mathcal{B}(\mathbf{pp}) \\ \mathbf{in} \leftarrow (\mathbf{pp}, \mathbf{s}, \mathbf{st}^*) \\ (u_1, w_1), (u'_1, w'_1) \leftarrow \mathcal{W}_{\text{unq}}^{\mathcal{O}_{\text{weak}}(\mathbf{in})}(\mathbf{in}) \end{array} \right] \leq \text{negl}(\lambda) \quad (35)$$

To bound the probability in Equation (35), consider a relaxed-binding adversary  $\mathcal{A}_{\text{rlx}}$ . The generator  $\mathcal{G}(1^\lambda, n_{\text{R}})$  outputs  $\mathbf{pp} \leftarrow \text{Setup}(1^\lambda, n_{\text{R}})$ , so  $\mathcal{A}_{\text{rlx}}$  receives public parameters with the same distribution as in Equation (35).

$\mathcal{A}_{\text{rlx}}(\text{pp})$  :

1.  $(s, \text{st}^*) \leftarrow \mathcal{B}(\text{pp})$  and  $\text{in} \leftarrow (\text{pp}, s, \text{st}^*)$ .
2. Run  $\mathcal{W}_{\text{uniq}}^{\mathcal{O}_{\text{weak}}(\text{in})}(\text{in})$ . For every query to  $\mathcal{O}_{\text{weak}}$ :
  - (a) Compute  $(u, \text{st}) \leftarrow \mathcal{B}'(\text{st}^*)$ .
  - (b) Run  $\mathcal{E}(\text{pp}, s, u, \text{st})$  to obtain  $w$ , retaining its internal value **result**.
  - (c) Return  $(u, w)$  and store **result** for this call.
3. If  $\mathcal{W}_{\text{uniq}}$  outputs  $\perp$ , a  $\perp$  witness, or equal witnesses, output  $\perp$ .
4. Otherwise, let  $(u_1, w_1)$  and  $(u'_1, w'_1)$  be the selected responses, and retrieve **result** and **result'** stored for the two calls that returned them.
5. Parse  $w_1 = (z_i)_{i \in [K+k]}$  and  $w'_1 = (z'_i)_{i \in [K+k]}$ , and choose  $i \in [K+k]$  such that  $z_i \neq z'_i$ .
6. Parse **result** and **result'** to obtain  $\rho_i, \rho_i^{(i)}, z, z^{(i)}$  and  $\rho'_i, \rho_i^{(i)'}, z', z^{(i)'}$ .
7. Assign

$$\Delta_1 \leftarrow \rho_i - \rho_i^{(i)}, \quad v_1 \leftarrow \text{Pack}_{\mathbb{L}}(z) - \text{Pack}_{\mathbb{L}}(z^{(i)}),$$

$$\Delta_2 \leftarrow \rho'_i - \rho_i^{(i)'}, \quad v_2 \leftarrow \text{Pack}_{\mathbb{L}}(z') - \text{Pack}_{\mathbb{L}}(z^{(i)'})$$

8. Parse  $c_i$  from  $u_1$  and output  $(c_i, \Delta_1, \Delta_2, v_1, v_2)$ .

Assume that, during an execution of  $\mathcal{A}_{\text{rlx}}$ ,  $\mathcal{W}_{\text{uniq}}$  returns  $(u_1, w_1), (u'_1, w'_1)$  such that the event in Equation (35) occurs. Every call to  $\mathcal{B}'(\text{st}^*)$  that  $\mathcal{A}_{\text{rlx}}$  makes to answer  $\mathcal{W}_{\text{uniq}}$ 's oracle queries uses fresh randomness. Therefore, by Equation (34), any two such outputs  $u_1, u'_1 \neq \perp$  satisfy  $\phi(u_1) = \phi(u'_1)$  with probability one. In particular,

1. Since  $w_1, w'_1 \neq \perp$ , we have  $u_1, u'_1 \neq \perp$ . Hence,  $\phi(u_1) = \phi(u'_1)$ , which guarantees that the selected instances share identical commitments  $(c_i)_{i \in [K+k]}$ .
2. Write  $w_1 = (z_i)_{i \in [K+k]}$  and  $w'_1 = (z'_i)_{i \in [K+k]}$ . Since  $w_1 \neq w'_1$ , there exists  $i \in [K+k]$  such that  $z_i \neq z'_i$ .

The values parsed by  $\mathcal{A}_{\text{rlx}}$  from **result** and **result'** for the two selected calls satisfy the following

$$\begin{aligned} z_i \neq z'_i &\iff \text{Pack}_{\mathbb{L}}(z_i) \neq \text{Pack}_{\mathbb{L}}(z'_i) \\ &\iff (\rho_i - \rho_i^{(i)})^{-1} \cdot (\text{Pack}_{\mathbb{L}}(z) - \text{Pack}_{\mathbb{L}}(z^{(i)})) \\ &\neq (\rho'_i - \rho_i^{(i)'})^{-1} \cdot (\text{Pack}_{\mathbb{L}}(z') - \text{Pack}_{\mathbb{L}}(z^{(i)'})) \end{aligned} \quad (36)$$

$$\|\text{Flat}_{\mathbb{L}}(z)\|_{\infty} \left\| \text{Flat}_{\mathbb{L}}(z^{(i)}) \right\|_{\infty} \|\text{Flat}_{\mathbb{L}}(z')\|_{\infty} \left\| \text{Flat}_{\mathbb{L}}(z^{(i)'}) \right\|_{\infty} < B \quad (37)$$

Equation (36) follows from (Item 2), the bijectivity of  $\text{Pack}_{\mathbb{L}}$ , and the construction of  $\mathcal{E}$ . Equation (37) follows from the construction of  $\mathcal{E}$ , which only outputs  $w_1, w'_1 \neq \perp$  when the internal extractor  $E$  (from Theorem 7) succeeds in the two selected calls. In particular, the internal extractor  $E$  succeeding guarantees that the verification functions  $V_{(\text{pp}, s, u_1, \text{st})}$  and  $V_{(\text{pp}, s, u'_1, \text{st}')}$  accept. These verification functions check that output tuples (corresponding to Equation (17) and Equation (18)) belong to  $\text{CE}(B, \mathcal{L})$ , which exactly checks the required norm bound on the flattened vectors. Since  $\rho_i, \rho_i^{(i)}, \rho'_i, \rho_i^{(i)'} \in \mathcal{C}$ , we have  $\Delta_1, \Delta_2 \in \mathcal{C} - \mathcal{C}$ . By Equation (37), the definitions of  $\text{Pack}_{\mathbb{L}}$  and the norm, and the triangle inequality,

$$\begin{aligned} \|v_1\|_{\infty} &\leq \|\text{Pack}_{\mathbb{L}}(z)\|_{\infty} + \left\| \text{Pack}_{\mathbb{L}}(z^{(i)}) \right\|_{\infty} < 2B, \\ \|v_2\|_{\infty} &\leq \|\text{Pack}_{\mathbb{L}}(z')\|_{\infty} + \left\| \text{Pack}_{\mathbb{L}}(z^{(i)'}) \right\|_{\infty} < 2B. \end{aligned}$$

Applying (23) to the two selected calls and using Item 1, we must have

$$c_i = \mathcal{L}(\Delta_1^{-1} \cdot v_1) = \mathcal{L}(\Delta_2^{-1} \cdot v_2).$$

Thus, since  $\mathcal{L}$  is a  $\mathbb{R}_{\mathbb{F}}$ -module homomorphism, we have

$$\Delta_1 \cdot c_i = \mathcal{L}(v_1) \wedge \Delta_2 \cdot c_i = \mathcal{L}(v_2) \quad (38)$$

Moreover, since  $\Delta_1$  and  $\Delta_2$  are invertible in the commutative ring  $R_{\mathbb{F}}$ , multiplication by  $\Delta_1\Delta_2$  is injective. Therefore, Equation (36) implies  $\Delta_1v_2 \neq \Delta_2v_1$ . All together, by (36), (37), and (38), we have that  $(c_i, \Delta_1, \Delta_2, v_1, v_2)$  is a  $(2B, \mathcal{C})$ -relaxed binding collision (Definition 7). Furthermore,  $\mathcal{A}_{\text{rlx}}$  runs in expected polynomial time. The execution of  $\mathcal{W}_{\text{uniq}}^{\mathcal{O}_{\text{weak}}}$ , including its oracle calls, is EPT. Recording each result and processing the two selected results add only polynomial time computations. Thus,  $\mathcal{A}_{\text{rlx}}$  is EPT.

By assumption (Definition 22), the ring commitment scheme defining  $\mathcal{L}$  satisfies  $(2B, \mathcal{C})$ -relaxed binding. Since  $\mathcal{A}_{\text{rlx}}$  is EPT and outputs a valid relaxed-binding collision whenever the event in Equation (35) occurs,

$$\Pr \left[ \begin{array}{l} w_1, w'_1 \neq \perp \\ \wedge w_1 \neq w'_1 \end{array} \middle| \begin{array}{l} \text{pp} \leftarrow \mathcal{G}(1^\lambda, \text{sz}) \\ (\text{s}, \text{st}^*) \leftarrow \mathcal{B}(\text{pp}) \\ \text{in} \leftarrow (\text{pp}, \text{s}, \text{st}^*) \\ (u_1, w_1), (u'_1, w'_1) \leftarrow \mathcal{W}_{\text{uniq}}^{\mathcal{O}_{\text{weak}}(\text{in})}(\text{in}) \end{array} \right] \leq \epsilon_{\text{rlx}}(2B, \mathcal{C}) \leq \text{negl}(\lambda).$$

Thus, the extractor satisfies condition (ii) of weak interactive reductions (Definition 17).  $\square$

## B.4 $\Pi_{\text{DEC}}$ is a Reduction of Knowledge (Theorem 13)

*Proof. Completeness:* First, we show that the verifier's checks in step 2 pass. Then, we will show that the output tuples belong to  $\text{CE}(b, \mathcal{L})^k$ .

By the definition of  $\text{CE}(B, \mathcal{L})$ , let  $z^b := \text{Flat}_{\mathbb{L}}(z)$  and  $x^b := \text{Flat}_{\mathbb{L}}(x)$ . We must have  $\|z^b\|_{\infty} < B = b^k$  (Definition 22), so  $\text{split}_b(z^b)$  is defined and  $z^b = \sum_{i=1}^k b^{i-1} \cdot z_i^b$ . Moreover,  $\text{Pack}_{\mathbb{L}}(x) = \mathcal{L}_{\text{in}}(\text{Pack}_{\mathbb{L}}(z))$ . Since  $\mathcal{L}_{\text{in}}$  projects the first  $n_{\text{R}, \text{in}}$  packed coordinates,  $x^b$  is the corresponding prefix of  $z^b$ . Therefore,  $\|x^b\|_{\infty} \leq \|z^b\|_{\infty} < B$ , so  $\text{split}_b(x^b)$  is also defined. Since  $\text{split}_b$  is coordinatewise, each  $x_i^b$  is the corresponding prefix of  $z_i^b$ . Hence,  $\text{Pack}_{\mathbb{L}}(x_i) = \mathcal{L}_{\text{in}}(\text{Pack}_{\mathbb{L}}(z_i))$  for every  $i \in [k]$ . By linearity,

$$\begin{aligned} z^b &= \sum_{i=1}^k b^{i-1} \cdot z_i^b \\ z &= \sum_{i=1}^k b^{i-1} \cdot z_i \\ \text{Pack}_{\mathbb{L}}(z) &= \sum_{i=1}^k b^{i-1} \cdot \text{Pack}_{\mathbb{L}}(z_i) \\ \mathcal{L}(\text{Pack}_{\mathbb{L}}(z)) &= \mathcal{L}(\sum_{i=1}^k b^{i-1} \cdot \text{Pack}_{\mathbb{L}}(z_i)) \end{aligned} \tag{39}$$

$$c = \sum_{i=1}^k b^{i-1} \cdot \mathcal{L}(\text{Pack}_{\mathbb{L}}(z_i)), \tag{40}$$

$$c = \sum_{i=1}^k b^{i-1} \cdot c_i, \tag{41}$$

Equation (40) follows directly from  $\mathcal{L}$  being a  $R_{\mathbb{F}}$ -module homomorphism. Equation (41) follows by construction of  $c_i \leftarrow \mathcal{L}(\text{Pack}_{\mathbb{L}}(z_i))$  in step 1. Therefore, the verifier's commitment check passes. Starting from equation (39), we must have

$$\begin{aligned} \text{Pack}_{\mathbb{L}}(z) &= \sum_{i=1}^k b^{i-1} \cdot \text{Pack}_{\mathbb{L}}(z_i) \\ \text{Emb}_{\mathbb{F}}(\text{Trans}_{\mathbb{F}}(\text{Pad})) \cdot \text{Pack}_{\mathbb{L}}(z) &= \sum_{i=1}^k b^{i-1} \cdot \text{Emb}_{\mathbb{F}}(\text{Trans}_{\mathbb{F}}(\text{Pad})) \cdot \text{Pack}_{\mathbb{L}}(z_i) \\ h &= \sum_{i=1}^k b^{i-1} \cdot h_i \\ \widetilde{h}(r) &= \sum_{i=1}^k b^{i-1} \cdot \widetilde{h}_i(r) \end{aligned} \tag{42}$$

$$y = \sum_{i=1}^k b^{i-1} \cdot y_i \tag{43}$$

Similarly, for every  $j \in [t]$ ,

$$\begin{aligned} \text{Pack}_{\mathbb{L}}(z) &= \sum_{i=1}^k b^{i-1} \cdot \text{Pack}_{\mathbb{L}}(z_i) \\ \text{Emb}_{\mathbb{L}}(\text{Trans}_{\mathbb{L}}(M_j)) \cdot \text{Pack}_{\mathbb{L}}(z) &= \sum_{i=1}^k b^{i-1} \cdot \text{Emb}_{\mathbb{L}}(\text{Trans}_{\mathbb{L}}(M_j)) \cdot \text{Pack}_{\mathbb{L}}(z_i) \\ h_j &= \sum_{i=1}^k b^{i-1} \cdot h_{i,j} \end{aligned}$$

$$\widetilde{h}_j(r) = \sum_{i=1}^k b^{i-1} \cdot \widetilde{h}_{i,j}(r) \quad (44)$$

$$y_j = \sum_{i=1}^k b^{i-1} \cdot y_{i,j} \quad (45)$$

Equation (42) and Equation (44) follow from the linearity of matrix multiplication and multilinear evaluation. Equation (43) and Equation (45) follow from the definition of  $\text{CE}(B, \mathcal{L})$  and the construction of  $y_i$  and  $y_{i,j}$  in step 1. Thus, by (41), (43), and (45), the verifier's checks pass.

Next, we show that the output tuple,  $(\mathbf{s}; \{c_i, x_i, r, y_i, \{y_{i,j}\}_{j \in [t]}\}_{i \in [k]}; \{z_i\}_{i \in [k]})$ , belongs to  $\text{CE}(b, \mathcal{L})^k$ . By the definition of  $\text{split}_b$ , we must have that  $\|z_i^b\|_\infty < b$  for all  $i \in [k]$ . Since  $\mathcal{L}_{\text{in}}$  is the trivial  $\mathbb{R}_{\text{in}}$ -module homomorphism which projects the first  $n_{\text{R,in}}$  coordinates, and  $\text{split}_b$  is coordinatewise, we must have that, by construction in step 2,  $\text{Pack}_{\mathbb{L}}(x_i) = \mathcal{L}_{\text{in}}(\text{Pack}_{\mathbb{L}}(z_i))$  for all  $i \in [k]$ . Thus, in total, we must have, along with the construction of  $(c_i, y_i, \{y_{i,j}\}_{j \in [t]})_{i \in [k]}$  in step 1, that the output tuples belong to  $\text{CE}(b, \mathcal{L})^k$ .

**Public coin:** The verifier uses no randomness in this protocol. Thus, the protocol is trivially public coin.

**Knowledge soundness:** Consider an arbitrary expected-polynomial time adversary  $(\mathcal{A}, \mathcal{P}^*)$  for  $\Pi_{\text{DEC}}$  with success probability,  $\epsilon(\mathcal{A}, \mathcal{P}^*) \geq 1/\text{poly}(\lambda)$ . We construct an extractor  $\mathcal{E}$  for  $\Pi_{\text{DEC}}$  as follows,

$\mathcal{E}(\text{pp}, \mathbf{s}, u_1 := (c, x, r, y, (y_j)_{j \in [t]}), \text{st}):$

1. Execute encoder  $(\text{pk}, \text{vk}) \leftarrow \mathcal{K}(\text{pp}, \mathbf{s})$ .
2. Simulate  $(u_2, w_2) \leftarrow \langle \mathcal{P}^*(\text{pk}, u_1, \text{st}), \mathcal{V}(\text{vk}, u_1) \rangle$ .
3. If  $u_2 = \perp$ , output  $\perp$ .
4. Parse  $(z_1, \dots, z_k) \leftarrow w_2$ .
5. Output  $w_1 := \sum_{i=1}^k b^{i-1} z_i$ .

**Extractor runtime:** The extractor runs in expected polynomial time, since it simulates only one execution between the adversary  $\mathcal{P}^*$  and verifier  $\mathcal{V}$ , which both run in expected polynomial time.

**Extractor success probability:** Assume that the simulated adversary  $(\mathcal{A}, \mathcal{P}^*)$  succeeds in convincing the verifier  $\mathcal{V}$  and the parties jointly output  $(\mathbf{s}, u_2, w_2) \in \text{CE}(b, \mathcal{L})^k$ ; note that this occurs with probability  $\epsilon(\mathcal{A}, \mathcal{P}^*)$ . Define

$$(c_i, x_i, r, y_i, (y_{i,j})_{j \in [t]})_{i \in [k]} := u_2 \text{ and } z_1, \dots, z_k := w_2.$$

By the definition of  $\text{CE}(b, \mathcal{L})$ , for every  $i \in [k]$ , define  $z_i^b := \text{Flat}_{\mathbb{L}}(z_i)$ ,  $h_i := \text{Emb}_{\mathbb{F}}(\text{Trans}_{\mathbb{F}}(\text{Pad})) \cdot \text{Pack}_{\mathbb{L}}(z_i)$ , and  $h_{i,j} := \text{Emb}_{\mathbb{L}}(\text{Trans}_{\mathbb{L}}(M_j)) \cdot \text{Pack}_{\mathbb{L}}(z_i)$  for every  $j \in [t]$ . We must have

$$\begin{aligned} c_i &= \mathcal{L}(\text{Pack}_{\mathbb{L}}(z_i)), \\ \text{Pack}_{\mathbb{L}}(x_i) &= \mathcal{L}_{\text{in}}(\text{Pack}_{\mathbb{L}}(z_i)), \\ \|z_i^b\|_\infty &< b, \\ y_i &= \widetilde{h}_i(r), \\ \forall j \in [t], \quad y_{i,j} &= \widetilde{h}_{i,j}(r) \end{aligned} \quad (46)$$

Since the adversary convinces the verifier, we must have

$$\begin{aligned} c &= \sum_{i=1}^k b^{i-1} \cdot c_i, \\ y &= \sum_{i=1}^k b^{i-1} \cdot y_i, \\ \forall j \in [t], \quad y_j &= \sum_{i=1}^k b^{i-1} \cdot y_{i,j} \end{aligned} \quad (47)$$

By construction in step 2 (i.e. definition of  $\text{split}_b$ ), we also must have  $x = \sum_{i=1}^k b^{i-1} \cdot x_i$ . By defining  $z := \sum_{i=1}^k b^{i-1} z_i$ , observe that  $x = \sum_{i=1}^k b^{i-1} \cdot x_i$ , (46), and (47) satisfy the remaining conditions stated in

Theorem 11, applied first with  $\mathbb{B} = \mathbb{F}$ ,  $M = \text{Pad}$ , vectors  $(z_i^b)_{i \in [k]}$ , and evaluations  $(y_i)_{i \in [k]}$ , and then, for every  $j \in [t]$ , with  $\mathbb{B} = \mathbb{L}$ ,  $M = M_j$ , vectors  $(z_i)_{i \in [k]}$ , and evaluations  $(y_{i,j})_{i \in [k]}$ . In both applications, the scalars are  $(b^{i-1})_{i \in [k]}$ , where each  $b^{i-1}$  is treated as a constant element of  $R_{\mathbb{F}}$ . We must have  $c = \mathcal{L}(\text{Pack}_{\mathbb{L}}(z))$ ,  $\text{Pack}_{\mathbb{L}}(x) = \mathcal{L}_{\text{in}}(\text{Pack}_{\mathbb{L}}(z))$ ,  $y = \tilde{h}(r)$ , and, for all  $j \in [t]$ ,  $y_j = \tilde{h}_j(r)$ , where  $h := \text{Emb}_{\mathbb{F}}(\text{Trans}_{\mathbb{F}}(\text{Pad})) \cdot \text{Pack}_{\mathbb{L}}(z)$  and  $h_j := \text{Emb}_{\mathbb{L}}(\text{Trans}_{\mathbb{L}}(M_j)) \cdot \text{Pack}_{\mathbb{L}}(z)$ . Since in (46), we have  $\|z_i^b\|_{\infty} < b$  for every  $i \in [k]$ , by the  $\mathbb{F}$ -linearity of  $\text{Flat}_{\mathbb{L}}$ , we must also have  $z^b = \sum_{i=1}^k b^{i-1} z_i^b$  and  $\|z^b\|_{\infty} \leq \sum_{i=1}^k b^{i-1}(b-1) = b^k - 1 < B$ . These are exactly the conditions for  $(s; u_1 := \{c, x, r, y, \{y_j\}_{j \in [t]}\}; w_1 := z)$  to belong to  $\text{CE}(B, \mathcal{L})$ . The simulated interaction in  $\mathcal{E}$  has the same distribution as the interaction defining  $\epsilon(\mathcal{A}, \mathcal{P}^*)$ . Whenever this interaction succeeds, the preceding argument shows that the extracted witness  $w_1$  satisfies the input relation. Therefore,  $\mathcal{E}$  outputs a satisfying witness with probability at least  $\epsilon(\mathcal{A}, \mathcal{P}^*)$ .  $\square$

## B.5 Hardness and Inversion Bound calculation scripts

We defer to the lattice estimator script for hardness. The inversion bounds are calculated with the following script.

```
# [LS18, eprint 2017-523] pg 6
# m is the cyclotomic polynomial index
def tau(m):
    return m if (m % 2) != 0 else m / 2

# [LS18, eprint 2017-523] Thm 1.1, pg 4
# m is the cyclotomic polynomial index
# p is the prime
# z is any divisor of m
# This tests for the condition for thm 1.1 to hold
def thm1_1_cond(m, p, z):
    cond1 = (p % z) == 1
    cond2 = Mod(p,m).multiplicative_order() == m/z
    return cond1 and cond2

# [LS18, eprint 2017-523] Thm 1.1, pg 4
# p is the prime
# z is any divisor of m
# lInf bound for elements to be invertible
# given that m,p,z satisfy thm 1.1 cond
def thm1_1_inv_bound(p, z):
    return (1/s1(z)*p^(1/euler_phi(z))).n()

def thm1_1_num_factors(z):
    return euler_phi(z)

# Output divisors of m
def divisors(m):
    zs = list()
    for i in range(1,m+1):
        if m % i == 0:
            zs.append(i)
    return zs

# [LS18, eprint 2017-523] pg 6, pg 9
# We only consider prime power cyclotomics
# m is the cyclotomic polynomial index
def s1(m):
    return sqrt(tau(m))
```

```

# checks if cyclotomic index m is power of two
def is_pow2(m):
    return sum(m.digits(2)) == 1

# [AL21] eprint Prop 2. 2021/202
# for all u,v in R, |u*v| / |v| <= gamma*|u|
# outputs T = gamma * |u|
# assumes we are only testing prime powers
def expansion_factor(m, norm):
    if is_pow2(m):
        return euler_phi(m) * norm
    else:
        return 2 * euler_phi(m) * norm

# p is prime
# max_idx is max cyclotomic index
# outputs list of (m, z)
def candidates(p, min_idx=10, max_idx=200):
    # prime powers
    possible_indices = [i for i in range(min_idx, max_idx) if len(factor(i)) == 1]
    c = list()
    for m in possible_indices:
        zs = divisors(m)
        for z in zs:
            if thm1_1_cond(m, p, z):
                c.append((Integer(m), Integer(z)))
    return c

def pre_filter(q, cyclotomic_index, z, chals):
    chals_max_diff = chals[-1] - chals[0]
    phi = cyclotomic_polynomial(cyclotomic_index) # index cyclotomic polynomial
    d = phi.degree() # degree of cyclotomic

    return chals_max_diff < thm1_1_inv_bound(q, z) and log(len(chals)^d,2).n() >= 120

def info(q, cyclotomic_index, z, chals):
    chals_norm = max({abs(c) for c in chals})
    chals_max_diff = chals[-1] - chals[0]
    phi = cyclotomic_polynomial(cyclotomic_index) # index cyclotomic polynomial
    d = phi.degree() # degree of cyclotomic
    T = expansion_factor(cyclotomic_index, chals_norm)

    print("####")
    print("Cyclotomic idx:", cyclotomic_index)
    print("Cyclotomic Poly:", phi)
    print("z:", z)
    print("Csmall norm is small enough?", chals_max_diff < thm1_1_inv_bound(q, z))
    print("Csmall large enough?", log(len(chals)^d,2).n() >= 120)
    print("Degree of Cyclotomic:", d)
    print("Expansion Factor T:", T)
    print("Invertible Norm bound:", thm1_1_inv_bound(q, z))
    print("log(|C_Small|):", log(len(chals)^d,2).n())
    print("Factors of Cyclotomic:", thm1_1_num_factors(z))
    print()

def possible_settings(q, chals):

```

```

for (cyclotomic_index, z) in candidates(q):
    if pre_filter(q, cyclotomic_index, z, chals):
        info(q, cyclotomic_index, z, chals)
    else:
        d = cyclotomic_polynomial(cyclotomic_index).degree()
        print(
            "[Does not satisfy challenge-set requirements] "
            "index: {}, degree: {}, z: {}, "
            "log(|C_Small|): {}, Invertible Norm bound: {}".format(
                cyclotomic_index,
                d,
                z,
                log(len(chals)^d,2).n(),
                thm1_1_inv_bound(q, z),
            )
        )

# Primes:
GL = 2^64 - 2^32 + 1
AGL = GL - 32
print("#####")
print("AGL #####")
print("#####")
# Small Challenge set
chals = [-1, 0, 1, 2]
possible_settings(AGL, chals)
print("#####")
print("M61 #####")
print("#####")
# Small Challenge set
chals = [-2, -1, 0, 1, 2]
possible_settings(2^61-1, chals)
print("#####")
print("GL #####")
print("#####")
# Small Challenge set
chals = [-2, -1, 0, 1, 2]
possible_settings(GL, chals)
print("#####")

```

## B.6 Lattice Estimator Script

```

from pathlib import Path
import sys

# Initialize this dependency with: git submodule update --init
sys.path.insert(0, str(Path(__file__).resolve().parent / "lattice-estimator"))

from estimator import *
Logging.set_level(Logging.LEVEL0)

M61 = 2^61 -1
GL = 2^64 - 2^32 +1
AGL = GL - 32
b = 2

kappa = 15

```

```

d = 64
k = 13
K = 50
B = b^k
n_R = 2^27
n_F = 2^33
assert n_F == n_R*d

T = 128
q = AGL

n_sis = kappa*d
m_sis = n_R*d
B_l2 = sqrt(n_R*d)*(8*T*B)

params = SIS.Parameters(n=n_sis, q=q, m=m_sis,length_bound=B_l2, norm=2)
_ = SIS.estimate(params)
print((K+k)*T*(b-1) < B)

kappa = 18
d = 54
k = 14
K = 61
B = b^k
n_R = 19884107
n_F = 2^30 - 46
assert n_F == n_R*d

T=216
q = GL

n_sis = kappa*d
m_sis = n_R*d
B_l2 = sqrt(n_R*d)*(8*T*B)

params = SIS.Parameters(n=n_sis, q=q, m=m_sis,length_bound=B_l2, norm=2)
_ = SIS.estimate(params)
print((K+k)*T*(b-1) < B)

kappa = 18
d = 54
k = 14
K = 61
B = b^k
n_R = 4971026
n_F = 2^28 - 52
assert n_F == n_R*d

T = 216
q = M61

n_sis = kappa*d
m_sis = n_R*d
B_l2 = sqrt(n_R*d)*(8*T*B)

params = SIS.Parameters(n=n_sis, q=q, m=m_sis,length_bound=B_l2, norm=2)

```

```
_ = SIS.estimate(params)
print((K+k)*T*(b-1) < B)
```