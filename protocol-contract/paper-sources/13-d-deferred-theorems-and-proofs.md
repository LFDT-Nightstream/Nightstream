## D Deferred theorems and proofs

### D.1 Proof of Matrix-Vector Product Transformation (Theorem 4)

*Proof.* Let  $M_1, \dots, M_m \in \mathbb{F}^{n_{\mathbb F}}$  be the rows of  $M$ . Define  $z_1, \dots, z_{n_R} \in \mathbb{F}^d$  and  $M_{i,1}, \dots, M_{i,n_R} \in \mathbb{F}^d$  (for all  $i \in [m]$ ) to be the partitions of vector  $z$  and row  $M_i$  into  $d$ -sized sub-vectors, respectively. Let  $\mathbf z_j$  be the coefficient embedding of  $z_j$ , and let  $\overline{\mathbf M}_{i,j}$  be the coefficient embedding of the transformed block  $\overline{M_{i,j}}$ . For all  $i \in [m]$ ,

$$\operatorname{ct}\!\left(\sum_{j\in[n_R]}\overline{\mathbf M}_{i,j}\mathbf z_j\right)
=\sum_{j\in[n_R]}\operatorname{ct}(\overline{\mathbf M}_{i,j}\mathbf z_j)
=\sum_{j\in[n_R]}\langle M_{i,j},z_j\rangle
=\langle M_i,z\rangle.$$

The first equality uses linearity of the constant-term map. The second uses the one-sided transform identity from Theorem 3. The third follows because  $(M_{i,j})_j$  and  $(z_j)_j$  are aligned partitions of  $M_i$  and  $z$ . Applying the equality to every row gives  $Mz=\operatorname{ct}(\bar M\mathbf z)$ .  $\square$

### D.2 Proof of Evaluation Homomorphism (Theorem 5)

*Proof.* Since  $\mathcal{L}$  and  $\mathcal L_{\mathrm{in}}$  are  $\mathbb{R}_F$ -module homomorphisms,

$$c = \sum_{i \in [\ell]} \rho_i c_i = \sum_{i \in [\ell]} \rho_i \mathcal{L}(\mathbf z_i) = \mathcal{L} \left( \sum_{i \in [\ell]} \rho_i \mathbf z_i \right) = \mathcal{L}(\mathbf z),$$

and, by the same argument,

$$\mathbf x=\sum_{i\in[\ell]}\rho_i\mathbf x_i
=\mathcal L_{\mathrm{in}}\!\left(\sum_{i\in[\ell]}\rho_i\mathbf z_i\right)
=\mathcal L_{\mathrm{in}}(\mathbf z).$$

Multilinear evaluation  $\mathbf z\mapsto\widehat{\bar M\mathbf z}(r)$  is an  $\mathbb R_{\mathbb K}$ -module homomorphism. Therefore,

$$\mathbf y=\sum_{i\in[\ell]}\rho_i\mathbf y_i
=\sum_{i\in[\ell]}\rho_i\widehat{\bar M\mathbf z_i}(r)
=\widehat{\bar M\!\left(\sum_{i\in[\ell]}\rho_i\mathbf z_i\right)}(r)
=\widehat{\bar M\mathbf z}(r).$$

Finally, Remark 2 gives  $\operatorname{ct}(\mathbf y_i)=\widetilde{Mz_i}(r)$  for every  $i$  and  $\operatorname{ct}(\mathbf y)=\widetilde{Mz}(r)$ , where  $z$  is the inverse coefficient embedding of  $\mathbf z$ .  $\square$

### D.3 Proof of Composition Theorem (Theorem 6)

*Proof.* Completeness follows by sequentially applying the completeness of  $\Pi_1$  and  $\Pi_2$ . Concatenating their public verifier messages preserves the public-coin property. It remains to prove knowledge soundness.

Consider an arbitrary expected polynomial-time adversary  $(\mathcal{A}, \mathcal{P}^*)$  for the composition  $\Pi := \Pi_2 \circ \Pi_1$  with success probability  $\epsilon(\mathcal{A}, \mathcal{P}^*) \ge 1/\text{poly}(\lambda)$ . Without loss of generality, the adversary  $\mathcal{P}^*$  can be split into two adversaries  $(\mathcal{P}_1^*, \mathcal{P}_2^*)$  such that given  $\text{pp} \leftarrow \mathcal{G}(1^\lambda,\mathrm{sz})$ ,  $(s, u_1, \text{st}_1) \leftarrow \mathcal{A}(\text{pp})$ , and  $(\text{pk}, \text{vk}) \leftarrow \mathcal{K}(\text{pp}, s)$ ,

$$\begin{aligned} & - \langle \mathcal{P}_1^*, \mathcal{V}_1 \rangle((\text{pk}, \text{vk}), u_1, \text{st}_1) \to (u_2, \text{st}_2) \\ & - \langle \mathcal{P}_2^*, \mathcal{V}_2 \rangle((\text{pk}, \text{vk}), u_2, \text{st}_2) \to (u_3, w_3) \end{aligned}$$

Define an adversary  $\mathcal A_1$  for  $\Pi_1$  that runs  $(s,u_1,\mathrm{st}_1^{\mathrm{orig}})\leftarrow\mathcal A(\mathrm{pp})$  and outputs

$$(s,u_1,\mathrm{st}_1):=(s,u_1,(\mathrm{pp},s,u_1,\mathrm{st}_1^{\mathrm{orig}})).$$

This state augmentation does not change any protocol output distribution. Next, construct an adversary  $\mathcal{A}_2 := (\mathcal{B}_2, \mathcal{B}'_2)$  for  $\Pi_2$ :

| |
|---|
| $\mathcal{B}_2(\mathrm{pp})\to(s,\mathrm{aux}_1)$ : |
| 1. $(s,u_1,\mathrm{st}_1)\leftarrow\mathcal A(\mathrm{pp})$ . |
| 2. Output $(s,\mathrm{aux}_1:=(\mathrm{pp},s,u_1,\mathrm{st}_1))$ . |
| $\mathcal{B}'_2(\mathrm{aux}_1)\to(u_2,\mathrm{st}_2)$ : |
| 1. Parse $(\mathrm{pp},s,u_1,\mathrm{st}_1)\leftarrow\mathrm{aux}_1$ . |
| 2. $(\mathrm{pk},\mathrm{vk})\leftarrow\mathcal K(\mathrm{pp},s)$ . |
| 3. Simulate $(u_2,\mathrm{st}_2)\leftarrow\langle\mathcal P_1^*,\mathcal V_1\rangle((\mathrm{pk},\mathrm{vk}),u_1,\mathrm{st}_1)$ . |
| 4. Output $(u_2,\mathrm{st}_2)$ . |
| $\mathcal A_2(\mathrm{pp})\to(s,u_2,\mathrm{st}_2)$ : |
| 1. $(s,\mathrm{aux}_1)\leftarrow\mathcal B_2(\mathrm{pp})$ . |
| 2. $(u_2,\mathrm{st}_2)\leftarrow\mathcal B'_2(\mathrm{aux}_1)$ . |
| 3. Output $(s,u_2,\mathrm{st}_2)$ . |

Observe that, by construction, the success probability  $\epsilon(\mathcal{A}_2, \mathcal{P}_2^*)$  of adversary  $(\mathcal{A}_2, \mathcal{P}_2^*)$  for  $\Pi_2$  is equal to the success probability  $\epsilon(\mathcal{A}, \mathcal{P}^*)$  of adversary  $(\mathcal{A}, \mathcal{P}^*)$  for  $\Pi$ . By condition (i) of strongness for  $\Pi_1$ , we must have

$$\Pr \left[ \begin{array}{c} u_2, u'_2 \neq \perp \\ \wedge \\ \phi(u_2) \neq \phi(u'_2) \end{array} \middle| \begin{array}{l} \text{pp} \leftarrow \mathcal{G}(1^\lambda, \text{sz}) \\ (s, \mathrm{aux}_1) \leftarrow \mathcal{B}_2(\text{pp}) \\ (u_2, \text{st}_2) \leftarrow \mathcal{B}'_2(\mathrm{aux}_1) \\ (u'_2, \text{st}'_2) \leftarrow \mathcal{B}'_2(\mathrm{aux}_1) \end{array} \right] = 0, \quad (1)$$

Thus, Equation (1) and the weak extraction and witness-uniqueness conditions for  $\Pi_2$  give an expected polynomial-time extractor  $\mathcal{E}_2$  such that

$$\Pr \left[ \begin{array}{l} (\text{pp}, \text{s}, u_2, w_2) \in \mathcal{R}'_2 \\ \left| \begin{array}{l} \text{pp} \leftarrow \mathcal{G}(1^\lambda, \text{sz}) \\ (\text{s}, u_2, \text{st}_2) \leftarrow \mathcal{A}_2(\text{pp}) \\ (\text{pk}, \text{vk}) \leftarrow \mathcal{K}(\text{pp}, \text{s}) \\ w_2 \leftarrow \mathcal{E}_2(\text{pp}, \text{s}, u_2, \text{st}_2) \end{array} \right. \end{array} \right] \ge \epsilon(\mathcal{A}, \mathcal{P}^*) - \text{negl}(\lambda) \quad (2)$$

$$\text{and} \quad \Pr \left[ \begin{array}{l} w_2, w'_2 \neq \perp \\ \wedge w_2 \neq w'_2 \\ \left| \begin{array}{l} \text{pp} \leftarrow \mathcal{G}(1^\lambda, \text{sz}) \\ (\text{s}, \mathrm{aux}_1) \leftarrow \mathcal{B}_2(\text{pp}) \\ (u_2, \text{st}_2) \leftarrow \mathcal{B}'_2(\mathrm{aux}_1) \\ w_2 \leftarrow \mathcal{E}_2(\text{pp}, \text{s}, u_2, \text{st}_2) \\ (u'_2, \text{st}'_2) \leftarrow \mathcal{B}'_2(\mathrm{aux}_1) \\ w'_2 \leftarrow \mathcal{E}_2(\text{pp}, \text{s}, u'_2, \text{st}'_2) \end{array} \right. \end{array} \right] \le \text{negl}(\lambda) \quad (3)$$

Next, construct an adversarial prover  $\mathcal{P}_1^{**}$  for  $\Pi_1$ . It must interact with the external verifier  $\mathcal V_1$ ; it must not simulate that verifier internally.

$\mathcal{P}_1^{**}(\mathrm{pk},u_1,\mathrm{st}_1)\to w_2$ :

1. Parse the augmented state to obtain  $(\mathrm{pp},s,u_1,\mathrm{st}_1^{\mathrm{orig}})$ .
2. Execute  $\mathcal P_1^*(\mathrm{pk},u_1,\mathrm{st}_1^{\mathrm{orig}})$ , forwarding its protocol messages to and from the external  $\mathcal V_1$ . At the end of the interaction, obtain the verifier output instance  $u_2$  and the continuation state  $\mathrm{st}_2$ .
3. Compute  $w_2\leftarrow\mathcal E_2(\mathrm{pp},s,u_2,\mathrm{st}_2)$ .
4. Output  $w_2$  as the output witness of the  $\Pi_1$  interaction.

By construction and Equation (2), the relaxed success probability  $\epsilon'(\mathcal{A}_1, \mathcal{P}_1^{**})$  for  $\Pi_1$  is at least  $\epsilon(\mathcal{A}, \mathcal{P}^*) - \text{negl}(\lambda) \ge 1/\text{poly}(\lambda)$ . Furthermore, by Equation (3) and the construction of  $(\mathcal{B}_2, \mathcal{B}'_2)$ ,

$$\Pr \left[ \begin{array}{l} w_2, w'_2 \neq \perp \\ \wedge \\ w_2 \neq w'_2 \\ \left| \begin{array}{l} \text{pp} \leftarrow \mathcal G(1^\lambda,\mathrm{sz}) \\ (\text{s}, u_1, \text{st}_1) \leftarrow \mathcal{A}_1(\text{pp}) \\ (\text{pk}, \text{vk}) \leftarrow \mathcal{K}(\text{pp}, \text{s}) \\ (u_2, w_2) \leftarrow \langle \mathcal{P}_1^{**}, \mathcal{V}_1 \rangle((\text{pk}, \text{vk}), u_1, \text{st}_1) \\ (u'_2, w'_2) \leftarrow \langle \mathcal{P}_1^{**}, \mathcal{V}_1 \rangle((\text{pk}, \text{vk}), u_1, \text{st}_1) \end{array} \right. \end{array} \right] \le \text{negl}(\lambda) \quad (4)$$

Thus, Equation (4) and condition (ii) of strongness for  $\Pi_1$  give an expected polynomial-time extractor  $\mathcal{E}_1$  such that

$$\Pr \left[ \begin{array}{l} (\text{pp}, \text{s}, u_1, w_1) \in \mathcal{R}_1 \\ \left| \begin{array}{l} \text{pp} \leftarrow \mathcal{G}(1^\lambda, \text{sz}) \\ (\text{s}, u_1, \text{st}_1) \leftarrow \mathcal{A}_1(\text{pp}) \\ (\text{pk}, \text{vk}) \leftarrow \mathcal{K}(\text{pp}, \text{s}) \\ w_1 \leftarrow \mathcal{E}_1(\text{pp}, \text{s}, u_1, \text{st}_1) \end{array} \right. \end{array} \right] \ge \epsilon(\mathcal{A}, \mathcal{P}^*) - \text{negl}(\lambda) \quad (5)$$

For the original adversary state, define

$$\mathcal E(\mathrm{pp},s,u_1,\mathrm{st}_1^{\mathrm{orig}})
:=\mathcal E_1(\mathrm{pp},s,u_1,(\mathrm{pp},s,u_1,\mathrm{st}_1^{\mathrm{orig}})).$$

Equation (5) then gives

$$\Pr \left[ \begin{array}{l} (\text{pp}, \text{s}, u_1, w_1) \in \mathcal{R}_1 \\ \left| \begin{array}{l} \text{pp} \leftarrow \mathcal{G}(1^\lambda, \text{sz}) \\ (\text{s}, u_1, \text{st}_1) \leftarrow \mathcal{A}(\text{pp}) \\ (\text{pk}, \text{vk}) \leftarrow \mathcal{K}(\text{pp}, \text{s}) \\ w_1 \leftarrow \mathcal{E}(\text{pp}, \text{s}, u_1, \text{st}_1) \end{array} \right. \end{array} \right] \ge \epsilon(\mathcal{A}, \mathcal{P}^*) - \text{negl}(\lambda).$$

Thus,  $\Pi := \Pi_2 \circ \Pi_1$  is a reduction of knowledge.  $\square$

### D.4 Proofs for $\Pi_{\mathrm{CCS}}$

We first provide a lemma that will be helpful for both the security and completeness of the interactive reduction.

**Lemma 7.** *Consider the following arbitrary items:*

$$\begin{aligned} & \text{structure } \mathbf{s}, \quad \text{vectors } z_1, \dots, z_{K+k} \in \mathbb{F}^{n_{\mathbb F}}, \\ & \text{point } r \in \mathbb{K}^{\log m}, \quad \text{evaluations } (\{y_{i,j} \in \mathbb{R}_{\mathbb{K}}\}_{j \in [t]})_{i=K+1}^{K+k}. \end{aligned}$$

For every  $i$ , let  $\mathbf z_i\in\mathbb R_{\mathbb F}^{n_R}$  be the coefficient embedding of  $z_i$ .

Define the scalar norm polynomial

$$P_b(Z) := \prod_{a=-(b-1)}^{b-1}\left(Z-\iota_q(a)\right).$$

Similarly to  $\Pi_{\text{CCS}}$  (Section 7.3), define

$$\begin{aligned}
F(\vec{X}, C) &:= \sum_{i=1}^{K} C^{i-1} \cdot f(\widetilde{M_1 z_i}(\vec{X}), \dots, \widetilde{M_t z_i}(\vec{X})), \\
\text{NC}(\vec{X}, C) &:= \sum_{i=1}^{K+k} C^{i-1} \cdot P_b(\widetilde{M_1z_i}(\vec{X})), \\
\text{Eval}(\vec{X}, C) &:= \text{eq}(\vec{X}, r) \cdot \sum_{i=K+1}^{K+k} \sum_{j=1}^{t} \sum_{\ell=1}^{d} C^{I(i,j,\ell)} \cdot \widetilde{\text{cf}(\bar M_j\mathbf z_i)_{\ell}}(\vec{X}), \\
Q(\vec{X}, \vec{A}, C) &:= \text{eq}(\vec{X}, \vec{A}) \cdot (F(\vec{X}, C) + C^K \cdot \text{NC}(\vec{X}, C)) + C^{2K+k} \cdot \text{Eval}(\vec{X}, C),
\end{aligned}$$

where

$$I(i,j,\ell) := (i-(K+1))+k(j-1)+kt(\ell-1).$$

Define the local and absolute carried targets by

$$T_{\mathrm{local}}(C) := \sum_{i=K+1}^{K+k} \sum_{j=1}^{t} \sum_{\ell=1}^{d} C^{I(i,j,\ell)} \cdot \text{cf}(y_{i,j})_{\ell},$$

$$T_{\mathrm{abs}}(C) := C^{2K+k}T_{\mathrm{local}}(C) = \sum_{i=K+1}^{K+k} \sum_{j=1}^{t} \sum_{\ell=1}^{d} C^{2K+k+I(i,j,\ell)} \cdot \text{cf}(y_{i,j})_{\ell}.$$

The challenges  $\alpha \in \mathbb{K}^{\log m}$  and  $\gamma \in \mathbb{K}$  are replaced above by indeterminates  $\vec{A}:=(A_1,\dots,A_{\log m})$  and  $C$ . The fresh, norm, and carried blocks occupy the disjoint  $C$ -exponent ranges

$$0,\dots,K-1, \qquad K,\dots,2K+k-1, \qquad 2K+k,\dots,2K+k+ktd-1.$$

We have

$$T_{\mathrm{abs}}(C) = \sum_{\vec{x}\in\{0,1\}^{\log m}} Q(\vec{x},\vec{A},C). \tag{6}$$

This identity holds if and only if

1.  $f(\widetilde{M_1 z_i},\dots,\widetilde{M_t z_i})\in\mathbb{ZS}_{\log m}$  for every  $i\in[K]$ ;
2.  $\|z_i\|_\infty<b$  for every  $i\in[K+k]$ ; and
3.  $y_{i,j}=\widehat{\bar M_j\mathbf z_i}(r)$  for every  $i\in\{K+1,\dots,K+k\}$  and  $j\in[t]$ .

*Proof.* Since the three exponent ranges are disjoint and powers of  $C$  are linearly independent over  $\mathbb{K}[\vec{A}]$ , the displayed polynomial identity holds if and only if

$$\forall i\in[K],\quad 0=\sum_{\vec{x}\in\{0,1\}^{\log m}}\text{eq}(\vec{x},\vec{A})\cdot f(\widetilde{M_1z_i}(\vec{x}),\dots,\widetilde{M_tz_i}(\vec{x})), \tag{7}$$

$$\forall i\in[K+k],\quad 0=\sum_{\vec{x}\in\{0,1\}^{\log m}}\text{eq}(\vec{x},\vec{A})\cdot P_b(\widetilde{M_1z_i}(\vec{x})), \tag{8}$$

and

$$\forall i\in\{K+1,\dots,K+k\},\ \forall j\in[t],\ \forall \ell\in[d],\quad
\text{cf}(y_{i,j})_\ell = \sum_{\vec{x}\in\{0,1\}^{\log m}}\text{eq}(\vec{x},r)\cdot\widetilde{\text{cf}(\bar M_j\mathbf z_i)_\ell}(\vec{x}). \tag{9}$$

By Lemma 6, Equation (7) is equivalent to Item 1. Equation (8) is equivalent to  $P_b(\widetilde{M_1z_i}(\vec{x}))=0$  at every Boolean point. At each such point,  $\widetilde{M_1z_i}(\vec{x})\in\mathbb{F}$ . Since  $M_1$  is the zero-padding injection, its Boolean evaluations list every coordinate of  $z_i$  and then zeros. The parameter restriction  $b\le B<q/2$  gives the exact scalar equivalence

$$P_b(a)=0 \quad\Longleftrightarrow\quad |\operatorname{ctr}_q(a)|<b, \qquad a\in\mathbb{F}.$$

Thus Equation (8) is equivalent to Item 2. Finally, the multilinear-extension identity in Equation (9), coefficient by coefficient, is equivalent to Item 3.  $\square$

**Lemma 8.** *The interactive reduction  $\Pi_{\text{CCS}} : \text{CCS}(b, \mathcal{L})^K \times \operatorname{BatchCE}_k(b, \mathcal{L}) \to \operatorname{BatchCE}_{K+k}(b, \mathcal{L})$  is complete and public coin.*

*Proof. Completeness.* Assume the original input tuples belong to relations  $\text{CCS}(b, \mathcal{L})$  (Definition 12) and  $\operatorname{BatchCE}_k(b, \mathcal{L})$  (Definition 13a). Thus, the carried components share one point and each component belongs to  $\operatorname{CE}(b, \mathcal{L})$ . We will first argue that the sum-check verifier in step 2 passes. Then, we will argue that the evaluation claim check in step 4 passes. Finally, we will argue that the output belongs to  $\operatorname{BatchCE}_{K+k}(b, \mathcal{L})$ .

By the definition of relations  $\text{CCS}(b, \mathcal{L})$  (Definition 12) and  $\text{CE}(b, \mathcal{L})$  (Definition 13), we must have that (Item 1), (Item 2), and (Item 3) from Lemma 7 hold. Therefore, we must have that

$$T_{\mathrm{abs}}(C) = \sum_{\vec{x} \in \{0,1\}^{\log m}} Q(\vec{x}, \vec{A}, C).$$

Thus, for any choice of challenges  $\alpha \in \mathbb{K}^{\log m}$  and  $\gamma \in \mathbb{K}$  chosen in step 1,

$$T_{\mathrm{abs}}(\gamma) = \sum_{\vec{x} \in \{0,1\}^{\log m}} Q(\vec{x}, \alpha, \gamma).$$

Thus, by the completeness of the sum-check protocol (Definition 6), we must have that the sum-check verifier (step 2) always passes.

By step 3 and Remark 2, we must have that

$$\text{ct}(y'_{i,j}) = \widetilde{M_j z_i}(r')$$

for all  $i \in [K+k]$  and  $j \in [t]$ . Since  $M_1$  is the zero-padding injection, we must have that

$$\text{ct}(y'_{i,1}) = \widetilde{M_1z_i}(r')$$

for all  $i \in [K+k]$ . Finally, by Remark 2, we must have that

$$\text{cf}(y'_{i,j})_\ell = \widetilde{\text{cf}(\bar M_j\mathbf z_i)_\ell}(r')$$

for all  $i \in \{K+1, \dots, K+k\}, j \in [t], \ell \in [d]$ . By definition of  $Q(\vec{X})$  in step 2, we must have that

$$Q(r') = \text{eq}(r', \alpha) \cdot (F + \gamma^K \cdot N) + \gamma^{2K+k} \cdot E$$

for values  $F, N, E$  derived in step 4. Thus, the verifier check in step 4 passes.

Observe that  $\Pi_{\text{CCS}}$  outputs exactly the original structure  $s$ , commitments  $(c_i)_{i \in [K+k]}$ , vectors  $(z_i)_{i \in [K+k]}$ , and instances  $(x_i)_{i \in [K+k]}$ . Thus, by the definition of  $\text{CCS}(b, \mathcal{L})$ , we must have immediately that every condition in  $\text{CE}(b, \mathcal{L})$  is satisfied for all the  $K+k$  tuples, except that

$$\forall i \in [K+k], j \in [t], \quad y'_{i,j} = \widehat{\bar M_j\mathbf z_i}(r').$$

However, this is exactly what is computed by the honest prover in step 3. Every output component uses the one verifier-selected point  $r'$ . Therefore, the output belongs to  $\operatorname{BatchCE}_{K+k}(b, \mathcal{L})$  as required.

**Public coin.** The sum-check protocol itself is a public-coin protocol. The remaining randomness from the verifier are the challenges  $\alpha \in \mathbb{K}^{\log m}$ ,  $\gamma \in \mathbb{K}$ , which are sampled uniformly at random and sent to the prover.  $\square$

We next prove Lemma 3 by analyzing conditions (i) and (ii) of strong interactive reductions (Definition 10).

**Proof of (i).** By construction, the verifier sets the commitments in the output instance  $u_2$  to the original commitments  $(c_i)_{i\in[K+k]}$  from the input instance  $u_1$ . Hence repeated executions on the same input have the same projected commitment value.

**Proof of (ii).** Consider an arbitrary expected polynomial-time adversary  $(\mathcal A,\mathcal P^*)$  satisfying the two premises in Definition 10(ii). For a fixed extractor input

$$x:=(\mathrm{pp},\mathbf{s},u_1,\mathrm{st}),$$

let  $\operatorname{Run}_x$  denote one fresh execution of  $\langle\mathcal P^*,\mathcal V\rangle$ . Let  $A$  be the event that this execution outputs  $(u_2,W)$  with

$$(\mathrm{pp},\mathbf{s},u_2,W)\in\mathcal R'_2:=\operatorname{BatchCE}_{K+k}(B_{\mathrm{amb}},\mathcal L),$$

and define

$$p_x:=\Pr[A\mid x].$$

For two independent executions on the same fixed  $x$ , let  $A_i$  be the ambient-success event and  $W_i$  the output witness of execution  $i\in\{0,1\}$ , and define

$$\delta_x:=\Pr[A_0\land A_1\land W_0\ne W_1\mid x].$$

The global quantities satisfy

$$\epsilon'=\mathbb E_x[p_x], \qquad \delta:=\mathbb E_x[\delta_x]\le\operatorname{negl}(\lambda),$$

where the second inequality follows because this event is contained in the witness-disagreement event from Definition 10(ii).

Consider the following success-gated rewinding extractor:

$\mathcal{E}(\mathrm{pp},\mathbf{s},u_1,\mathrm{st})\to w_1$ :

1. Compute  $(\mathrm{pk},\mathrm{vk})\leftarrow\mathcal K(\mathrm{pp},\mathbf{s})$ .
2. Simulate one fresh execution  $\mathrm{result}_0\leftarrow\langle\mathcal P^*,\mathcal V\rangle((\mathrm{pk},\mathrm{vk}),u_1,\mathrm{st})$ . If it does not output  $(u_2^{(0)},W_0)$  with  $(\mathrm{pp},\mathbf{s},u_2^{(0)},W_0)\in\mathcal R'_2$ , output  $\perp$ .
3. Repeatedly simulate fresh, independent executions until one outputs  $(u_2^{(*)},W_*)$  with  $(\mathrm{pp},\mathbf{s},u_2^{(*)},W_*)\in\mathcal R'_2$ .
4. If  $W_0\ne W_*$ , output  $\perp$ .
5. Parse  $(z_1,\dots,z_{K+k})\leftarrow W_*$ . For  $i\in[K]$ , set  $w_i^{\mathrm{CCS}}\leftarrow z_i[n_{\mathbb{F},\mathrm{in}}:]$ .
6. Output  $w_1:=(w_1^{\mathrm{CCS}},\dots,w_K^{\mathrm{CCS}},z_{K+1},\dots,z_{K+k})$ .

Let  $\operatorname{Rec}(W)$  denote the input witness reconstructed in steps 5--6, and let  $\operatorname{Invalid}(\operatorname{Rec}(W))$  denote the event that  $(\mathrm{pp},\mathbf{s},u_1,\operatorname{Rec}(W))\notin\mathcal R_1$ .

*Extractor runtime.* Let  $T_x$  be the expected running time of one fresh execution on fixed input  $x$ . If  $p_x=0$ , step 2 always outputs  $\perp$  and the loop is never entered. If  $p_x>0$ , the loop is entered with probability  $p_x$  and then uses  $1/p_x$  executions in expectation. Hence the expected number of executions is at most two on every fixed  $x$ ; for  $p_x>0$  it is

$$1+p_x\cdot\frac1{p_x}=2,$$

while for  $p_x=0$  it is one. More precisely, for  $p_x>0$ , condition on the initial success event  $A_0$  and let  $N_x$  be the number of loop executions. Independence of the fresh trials gives

$$\mathbb E\!\left[\sum_{j=1}^{N_x}\operatorname{time}(\operatorname{Run}_{x,j})\,\middle|\,x,A_0\right]
=T_x\sum_{j\ge1}(1-p_x)^{j-1}=\frac{T_x}{p_x}.$$

Therefore

$$\mathbb E[\operatorname{time}(\mathcal E)\mid x]
\le T_x+p_x\frac{T_x}{p_x}=2T_x.$$

Averaging over  $x$  preserves expected polynomial time because  $(\mathcal A,\mathcal P^*)$  is EPT; all remaining extractor work is polynomial.

*Witness disagreement.* For  $p_x>0$ , the retained successful loop execution has distribution  $\operatorname{Run}_x\mid A$  and uses randomness independent of the initial execution. Equivalently, for the probability analysis we may sample this conditioned execution in advance and reveal it only on branches where the initial execution succeeds. Therefore

$$\Pr[A_0\land W_0\ne W_*\mid x]
=\frac{\Pr[A_0\land A_1\land W_0\ne W_1\mid x]}{p_x}
=\frac{\delta_x}{p_x}. \tag{10}$$

All ratios involving  $p_x$  in this proof are defined as zero when  $p_x=0$ ; in that case  $\delta_x=0$  as well.

*Direct invalid-witness event.* Condition on an arbitrary successful loop execution and its retained witness  $W_*=(z_1,\dots,z_{K+k})$ , and let  $\mathbf z_i$  be the coefficient embedding of  $z_i$ . Membership in  $\mathcal R'_2$  explicitly gives, for every  $i\in[K+k]$ ,

$$c_i=\mathcal L(\mathbf z_i), \qquad \mathbf x_i=\mathcal L_{\mathrm{in}}(\mathbf z_i).$$

Because  $\mathcal L_{\mathrm{in}}$  projects the input coordinates, these equations also show that the first  $n_{\mathbb{F},\mathrm{in}}$  entries of each  $z_i$  equal the corresponding input  $x_i$ . Thus, if  $\operatorname{Rec}(W_*)$  is invalid, all commitment and input-projection requirements already hold and at least one of the three conditions in Lemma 7 must fail. Consequently,

$$T_{\mathrm{abs}}(C)\ne\sum_{\vec{x}\in\{0,1\}^{\log m}}Q(\vec{x},\vec A,C).$$

The initial execution is unconditioned and independent of  $W_*$ . Hence the joint event

$$A_0\land W_0=W_*\land\operatorname{Invalid}(\operatorname{Rec}(W_*))$$

can occur only if either the sum-check verifier accepts a false sum claim or the nonzero polynomial

$$P(C,\vec A):=T_{\mathrm{abs}}(C)-\sum_{\vec{x}\in\{0,1\}^{\log m}}Q(\vec{x},\vec A,C)$$

vanishes at the fresh point  $(\gamma,\alpha)$  used by the initial execution. This is a direct joint-event bound; it does not condition the initial execution on success or witness equality.

For the fixed input  $x$ , the individual degree of  $Q$  is at most

$$D_Q(x):=\max(D_f+1,2b,2),$$

so sum-check soundness contributes

$$\epsilon_{\mathrm{SC}}(x):=\frac{D_Q(x)\log m}{|\mathbb K|}.$$

The total degree of  $P(C,\vec A)$  is at most

$$D_{\mathrm{SZ}}:=2K+k-1+\max(\log m,ktd),$$

so Schwartz--Zippel contributes

$$\epsilon_{\mathrm{SZ}}:=\frac{D_{\mathrm{SZ}}}{|\mathbb K|}.$$

The joint-event bound is uniform over every fixed successful loop transcript. Averaging it over the retained loop transcript therefore gives, for every fixed  $x$  with  $p_x>0$ ,

$$\Pr[A_0\land W_0=W_*\land\operatorname{Invalid}(\operatorname{Rec}(W_*))\mid x]
\le \epsilon_{\mathrm{SC}}(x)+\epsilon_{\mathrm{SZ}}. \tag{11}$$

Combining Equations (10) and (11), and observing that the same lower bound is trivial when  $p_x=0$ ,

$$\Pr[\text{valid extraction}\mid x]
\ge p_x-\frac{\delta_x}{p_x}-\epsilon_{\mathrm{SC}}(x)-\epsilon_{\mathrm{SZ}}. \tag{12}$$

It remains to average the conditioning loss without assuming a pointwise lower bound on  $p_x$ . Since the two executions defining  $\delta_x$  are independent,

$$0\le\delta_x\le\Pr[A_0\land A_1\mid x]=p_x^2.$$

Therefore, by Cauchy--Schwarz,

$$\begin{aligned}
\mathbb E_x\!\left[\frac{\delta_x}{p_x}\right]
&=\mathbb E_x\!\left[\sqrt{\delta_x}\cdot\frac{\sqrt{\delta_x}}{p_x}\right] \\
&\le\sqrt{\mathbb E_x[\delta_x]}\sqrt{\mathbb E_x\!\left[\frac{\delta_x}{p_x^2}\right]} \\
&\le\sqrt{\delta}.
\end{aligned}$$

Since  $D_f\le u$  for every structure, define the uniform bounds

$$\overline D_Q:=\max(u+1,2b,2), \qquad
\overline\epsilon_{\mathrm{SC}}:=\frac{\overline D_Q\log m}{|\mathbb K|}.$$

Then  $\epsilon_{\mathrm{SC}}(x)\le\overline\epsilon_{\mathrm{SC}}$  for every  $x$ .

Averaging Equation (12) now yields

$$\Pr[\text{valid extraction}]
\ge\epsilon'-\sqrt{\delta}-\overline\epsilon_{\mathrm{SC}}-\epsilon_{\mathrm{SZ}}
=\epsilon'-\operatorname{negl}(\lambda).$$

The last equality follows because the square root of a negligible function is negligible,  $1/|\mathbb K|$  is negligible, and all degree and dimension parameters above are polynomially bounded. This proves condition (ii), and hence  $\Pi_{\mathrm{CCS}}$  is strong.  $\square$

### D.5 Proofs for $\Pi_{\mathrm{RLC}}$

**Lemma 9.** *The interactive reduction  $\Pi_{RLC} : \operatorname{BatchCE}_{K+k}(b, \mathcal{L}) \to \text{CE}(B, \mathcal{L})$  is complete and public coin.*

*Proof. Completeness.* For every  $i\in[K+k]$ , let  $\mathbf z_i$  and  $\mathbf x_i$  be the coefficient embeddings of  $z_i$  and  $x_i$ . By definition of  $\operatorname{BatchCE}_{K+k}(b, \mathcal{L})$  (Definition 13a), the input components share one point  $r$  and satisfy the conditions in Theorem 5. The verifier and prover compute

$$\mathbf x=\sum_{i=1}^{K+k}\rho_i\mathbf x_i,
\qquad
\mathbf z=\sum_{i=1}^{K+k}\rho_i\mathbf z_i,$$

and output their inverse coefficient embeddings  $x$  and  $z$ . Thus, Theorem 5 shows that the output tuple satisfies all requirements of  $\text{CE}(B, \mathcal{L})$ , except possibly  $\|z\|_\infty < B=b^k$ .

However, we show that this bound follows from the expansion factor  $T$  of  $\mathcal{C}$  chosen in Definition 14:

$$\begin{aligned} \|z\|_\infty=\|\mathbf z\|_\infty &= \left\| \sum_{i=1}^{k+K} \rho_i\mathbf z_i \right\|_\infty \le \sum_{i=1}^{k+K} \|\rho_i\mathbf z_i\|_\infty \\ &\le \sum_{i=1}^{k+K} T \|\mathbf z_i\|_\infty \le (k + K)T(b - 1) < B \end{aligned}$$

where the second inequality is from the expansion factor of  $\mathcal{C}$  being  $T$ , the third inequality is from the definition of  $\text{CE}(b, \mathcal{L})$ , which enforces a norm bound of  $b$ , and the last inequality is by assumption (Definition 14). Hence, the output tuple must belong to  $\text{CE}(B, \mathcal{L})$ .

**Public coin.** The verifier's randomness consists of challenges  $\rho_1, \dots, \rho_{k+K}$ , which are sampled uniformly at random from  $\mathcal{C}$  and sent to the prover.  $\square$

We prove the conditions of weak interactive reductions (Definition 9).

*Proof.* Consider an arbitrary expected-polynomial time adversary  $(\mathcal{A}, \mathcal{P}^*)$  for  $\Pi_{RLC}$  with success probability,  $\epsilon(\mathcal{A}, \mathcal{P}^*) \ge 1/\text{poly}(\lambda)$ . First, we can construct an adversary and verification function for Theorem 10,

$A_{(\text{pp}, \mathbf{s}, u_1, \text{st})}(\vec{c})$  :

1. Execute encoder  $(\text{pk}, \text{vk}) \leftarrow \mathcal{K}(\text{pp}, \mathbf{s})$ .
2. Simulate  $(u_2, w_2) \leftarrow \langle \mathcal{P}^*(\text{pk}, u_1, \text{st}), \mathcal{V}(\text{vk}, u_1) \rangle$  with verifier randomness  $\vec{c}$ .
3. Output  $w_2$

 $V_{(\text{pp}, \mathbf{s}, u_1, \text{st})}(\vec{c}, w_2) \to \{0, 1\}$  :

1. Parse  $(c_i,x_i,r,\{y_{i,j}\}_{j\in[t]})_{i\in[K+k]}\leftarrow u_1$  and  $(\rho_1,\dots,\rho_{K+k})\leftarrow\vec c$ , and let  $\mathbf x_i$  be the coefficient embedding of  $x_i$ .
2. Compute  $\mathbf x(\vec c):=\sum_i\rho_i\mathbf x_i$ , let  $x(\vec c)$  be its inverse coefficient embedding, and compute the deterministic  $\Pi_{\mathrm{RLC}}$  verifier output

$$u_2(\vec c):=\left(\sum_i\rho_i c_i,\ x(\vec c),\ r,\ \left\{\sum_i\rho_i y_{i,j}\right\}_{j\in[t]}\right).$$

3. Output accept if and only if  $(\mathrm{pp},\mathbf s,u_2(\vec c),w_2)\in\operatorname{CE}(B,\mathcal L)$ .

Let  $E_{(\text{pp}, \mathbf{s}, u_1, \text{st})}$  be the corresponding extractor from Theorem 10. We define  $E(\text{pp}, \mathbf{s}, u_1, \text{st})$  as the trivial algorithm that executes  $E_{(\text{pp}, \mathbf{s}, u_1, \text{st})}$  by simulating calls to  $A_{(\text{pp}, \mathbf{s}, u_1, \text{st})}$ . We construct an extractor for adversary  $(\mathcal{A}, \mathcal{P}^*)$  as follows:

$\mathcal{E}(\text{pp}, \mathbf{s}, u_1, \text{st})$  :

1.  $\text{result} \leftarrow E(\text{pp}, \mathbf{s}, u_1, \text{st})$ .
2. If  $u_1 = \perp$  or  $\text{result} = \perp$ , output  $\perp$ .
3. Parse  $(\vec{c}, w'), (\vec{c}_1, w'_1), \dots, (\vec{c}_{K+k}, w'_{K+k}) \leftarrow \text{result}$ .
4. Parse  $z \leftarrow w'$  and  $\rho_1, \dots, \rho_{K+k} \leftarrow \vec{c}$ , and let  $\mathbf z$  be the coefficient embedding of  $z$ .
5. For  $i \in [K+k]$ ,
  - (a) Parse  $z^{(i)} \leftarrow w'_i$  and  $\rho_1^{(i)}, \dots, \rho_{K+k}^{(i)} \leftarrow \vec{c}_i$ , and let  $\mathbf z^{(i)}$  be the coefficient embedding of  $z^{(i)}$ .
  - (b) Assign  $\mathbf{z}_i \leftarrow (\rho_i - \rho_i^{(i)})^{-1} \cdot (\mathbf{z} - \mathbf{z}^{(i)})$ , and let  $z_i$  be its inverse coefficient embedding.
6. Parse  $(c_i, x_i, r, \{y_{i,j}\}_{j \in [t]})_{i \in [K+k]} \leftarrow u_1$ .
7. Output  $w_1 := (z_i)_{i \in [K+k]}$  if and only if
$$(\mathbf{s}; c_i, x_i, r, \{y_{i,j}\}_{j \in [t]}; z_i)_{i \in [K+k]} \in \operatorname{BatchCE}_{K+k}(B_{\mathrm{amb}}, \mathcal{L})$$

*Extractor runtime.* By Theorem 10, we are guaranteed  $E_{(\text{pp}, \mathbf{s}, u_1, \text{st})}$  makes in expectation at most  $(K+k)+1$  calls to  $A_{(\text{pp}, \mathbf{s}, u_1, \text{st})}$ . Hence, our overall extractor  $\mathcal{E}$  runs in expected polynomial time.

*Extractor success probability.* Theorem 10 gives the exact coordinate-fork loss  $(K+k)/|\mathcal C|$ . We retain the weaker conservative corollary  $(K+k+1)/|\mathcal C|$  in this appendix. Thus  $E(\text{pp},\mathbf{s},u_1,\text{st})$  outputs  $(K+k)+1$  pairs  $(\vec c,w'),(\vec c_1,w'_1),\dots,(\vec c_{K+k},w'_{K+k})$  such that

- $V_{(\text{pp}, \mathbf{s}, u_1, \text{st})}(\vec{c}, w') = 1$ ,
- for all  $i \in [K+k]$ ,  $V(\vec{c}_i, w'_i) = 1$ , and
- $(\vec{c}, \vec{c}_1, \dots, \vec{c}_{K+k}) \in \text{SS}(\mathcal{C}, K+k)$

For each fixed  $x=(\mathrm{pp},\mathbf{s},u_1,\mathrm{st})$ , let  $\epsilon_x:=\epsilon^{V_x}(A_x)$ . Theorem 10 gives the event above probability at least  $\epsilon_x-(K+k+1)/|\mathcal C|$ . Averaging over  $x$  and using  $\mathbb E_x[\epsilon_x]=\epsilon(\mathcal A,\mathcal P^*)$  shows that these conditions hold

with probability at least

$$\epsilon(\mathcal A,\mathcal P^*)-\frac{K+k+1}{|\mathcal C|}. \tag{13}$$

Assume that this event occurs. Since  $V_{(\text{pp}, \text{s}, u_1, \text{st})}(\vec{c}, w') = 1$ , we must have, for  $\mathbf{x} := \sum_{i=1}^{K+k} \rho_i \mathbf{x}_i$  and its inverse coefficient embedding  $x$ , that

$$\left(\mathbf s;\left(c:=\sum_{i=1}^{K+k}\rho_i c_i,\ x,\ r,\ \left\{y_j:=\sum_{i=1}^{K+k}\rho_i y_{i,j}\right\}_{j\in[t]}\right);z\right)
\in\operatorname{CE}(B,\mathcal L). \quad (14)$$

where  $(c_i, x_i, r, \{y_{i,j}\}_j)_i$  are the instance elements in  $u_1$  (parsed in step 6) and  $z \leftarrow w'$  and  $(\rho_i)_i \leftarrow \vec{c}$  are the elements parsed in step 4.

For all  $i \in [K+k]$ , since  $V(\vec{c}_i, w'_i) = 1$ , we must have, for  $\mathbf{x}^{(i)} := \sum_{h=1}^{K+k} \rho_h^{(i)} \mathbf{x}_h$  and its inverse coefficient embedding  $x^{(i)}$ , that

$$\left(\mathbf s;\left(c^{(i)}:=\sum_{h=1}^{K+k}\rho_h^{(i)}c_h,\ x^{(i)},\ r,\ \left\{y_j^{(i)}:=\sum_{h=1}^{K+k}\rho_h^{(i)}y_{h,j}\right\}_{j\in[t]}\right);z^{(i)}\right)
\in\operatorname{CE}(B,\mathcal L). \quad (15)$$

where  $(c_i, x_i, r, \{y_{i,j}\}_j)_i$  are in  $u_1$  and  $z^{(i)} \leftarrow w'_i$  and  $(\rho_j^{(i)})_j \leftarrow \vec{c}_i$  are the elements parsed in step 5a. By definition of  $\text{CE}(B, \mathcal{L})$  (Definition 13), we must have

$$c = \mathcal{L}(\mathbf z), \quad c^{(i)} = \mathcal{L}(\mathbf z^{(i)}), \quad \mathbf{x} = \mathcal{L}_{\text{in}}(\mathbf z), \quad \mathbf{x}^{(i)} = \mathcal{L}_{\text{in}}(\mathbf z^{(i)}) \quad (16)$$

Since  $(\vec{c}, \vec{c}_1, \dots, \vec{c}_{K+k}) \in \text{SS}(\mathcal{C}, K+k)$ , we must have for all  $i \in [K+k]$  that

$$(\rho_1, \dots, \rho_{K+k}) \equiv_i (\rho_1^{(i)}, \dots, \rho_{K+k}^{(i)}) \quad (17)$$

which means the challenges differ only on index  $i$ . By definition of strong sampling set (Definition 17), we must have  $(\rho_i - \rho_i^{(i)})$  is invertible for all  $i \in [K+k]$ .

Thus, by Equation (16) and Equation (17), we have for all  $i \in [K+k]$ ,

$$\begin{aligned} c - c^{(i)} &= \mathcal{L}(\mathbf z) - \mathcal{L}(\mathbf z^{(i)}) \\ \mathbf{x} - \mathbf{x}^{(i)} &= \mathcal{L}_{\text{in}}(\mathbf z) - \mathcal{L}_{\text{in}}(\mathbf z^{(i)}) \\ \sum_{h=1}^{K+k} \rho_h c_h - \sum_{h=1}^{K+k} \rho_h^{(i)} c_h &= \mathcal{L}(\mathbf z) - \mathcal{L}(\mathbf z^{(i)}), \\ \sum_{h=1}^{K+k} \rho_h \mathbf{x}_h - \sum_{h=1}^{K+k} \rho_h^{(i)} \mathbf{x}_h &= \mathcal{L}_{\text{in}}(\mathbf z) - \mathcal{L}_{\text{in}}(\mathbf z^{(i)}) \end{aligned} \quad (18)$$

$$\begin{aligned} (\rho_i - \rho_i^{(i)}) \cdot c_i &= \mathcal{L}(\mathbf z) - \mathcal{L}(\mathbf z^{(i)}), \\ (\rho_i - \rho_i^{(i)}) \cdot \mathbf{x}_i &= \mathcal{L}_{\text{in}}(\mathbf z) - \mathcal{L}_{\text{in}}(\mathbf z^{(i)}) \end{aligned} \quad (19)$$

$$\begin{aligned} c_i &= \mathcal{L}\left((\rho_i - \rho_i^{(i)})^{-1} \cdot (\mathbf z - \mathbf z^{(i)})\right), \\ \mathbf{x}_i &= \mathcal{L}_{\text{in}}\left((\rho_i - \rho_i^{(i)})^{-1} \cdot (\mathbf z - \mathbf z^{(i)})\right) \end{aligned} \quad (20)$$

$$c_i = \mathcal{L}(\mathbf z_i), \quad \mathbf x_i = \mathcal{L}_{\text{in}}(\mathbf z_i) \quad (21)$$

where equation (18) follows from (14), (15), and Equation (16). Equation (19) follows from the equivalence in Equation (17). Equation (20) follows from  $\mathcal{L}, \mathcal{L}_{\text{in}}$  being  $\mathcal{R}$ -module homomorphisms and  $\mathcal{C}$  being a strong sampling set (Definition 17) which because  $\rho_i \neq \rho_i^{(i)}$  (guaranteed by (17)) means  $\rho_i - \rho_i^{(i)}$  is invertible. Equation (21) is by construction (step 5b).

We make a similar argument for the evaluations. In particular, by the definition of  $\text{CE}(B, \mathcal{L})$  (Definition 13), Equation (14), and Equation (15), we must have that

$$y_j := \widehat{\bar{M}_j\mathbf z}(r), \quad y_j^{(i)} := \widehat{\bar{M}_j\mathbf z^{(i)}}(r) \quad (22)$$

Thus, we must have for all  $i \in [K+k]$  and  $j \in [t]$ ,

$$y_j - y_j^{(i)} = \widehat{\bar{M}_j\mathbf z}(r) - \widehat{\bar{M}_j\mathbf z^{(i)}}(r) \quad (23)$$

$$\sum_{h=1}^{K+k} \rho_h y_{h,j} - \sum_{h=1}^{K+k} \rho_h^{(i)} y_{h,j} = \widehat{\bar{M}_j(\mathbf z - \mathbf z^{(i)})}(r) \quad (24)$$

$$(\rho_i - \rho_i^{(i)}) \cdot y_{i,j} = \widehat{\bar{M}_j(\mathbf z - \mathbf z^{(i)})}(r) \quad (25)$$

$$y_{i,j} = \widehat{\bar{M}_j\left((\rho_i - \rho_i^{(i)})^{-1} \cdot (\mathbf z - \mathbf z^{(i)})\right)}(r) \quad (26)$$

$$= \widehat{\bar{M}_j\mathbf z_i}(r) \quad (27)$$

where Equation (23) follows from Equation (22), Equation (24) follows from Equation (14) and Equation (15) and the linearity of multilinear evaluation, Equation (25) follows from the equivalence (17), and (26) follows from  $\mathcal{C}$  being a strong sampling set (Definition 17) which because  $\rho_i \neq \rho_i^{(i)}$  (guaranteed by (17)) means  $\rho_i - \rho_i^{(i)}$  is invertible.

Therefore, Equation (13), by Equation (20), and Equation (26), we must have with probability  $\epsilon(\mathcal{A}, \mathcal{P}^*) - ((K+k) + 1)/|\mathcal{C}|$ , the extractor outputs witness elements  $z_1, \dots, z_{K+k}$  such that

$$(\mathbf{s}; c_i, x_i, r, \{y_{i,j}\}_{j \in [t]}; z_i)_{i \in [K+k]} \in \operatorname{BatchCE}_{K+k}(B_{\mathrm{amb}}, \mathcal{L}),$$

for the universal strict ambient bound  $B_{\mathrm{amb}}=\lfloor q/2\rfloor+1$ , since every centered field representative has magnitude at most  $\lfloor q/2\rfloor<B_{\mathrm{amb}}$ .

Now, assume that  $\mathcal{A} := (\mathcal{B}, \mathcal{B}')$  such that

$$\Pr \left[ \begin{array}{c} u_1, u'_1 \neq \perp \\ \wedge \\ \phi(u_1) \neq \phi(u'_1) \end{array} \middle| \begin{array}{l} \text{pp} \leftarrow \mathcal{G}(1^\lambda, \text{sz}) \\ (\mathbf{s}, \text{st}^*) \leftarrow \mathcal{B}(\text{pp}) \\ (u_1, \text{st}) \leftarrow \mathcal{B}'(\text{st}^*) \\ (u'_1, \text{st}') \leftarrow \mathcal{B}'(\text{st}^*) \end{array} \right] = 0 \quad (28)$$

We will show that

$$\Pr \left[ \begin{array}{l} w_1, w'_1 \neq \perp \\ \wedge w_1 \neq w'_1 \end{array} \left| \begin{array}{l} \text{pp} \leftarrow \mathcal{G}(1^\lambda, \text{sz}) \\ (\mathbf{s}, \text{st}^*) \leftarrow \mathcal{B}(\text{pp}) \\ (u_1, \text{st}) \leftarrow \mathcal{B}'(\text{st}^*) \\ w_1 \leftarrow \mathcal{E}(\text{pp}, \mathbf{s}, u_1, \text{st}) \\ (u'_1, \text{st}') \leftarrow \mathcal{B}'(\text{st}^*) \\ w'_1 \leftarrow \mathcal{E}(\text{pp}, \mathbf{s}, u'_1, \text{st}') \end{array} \right. \right] \le \text{negl}(\lambda) \quad (29)$$

Assume that the event  $w_1, w'_1 \neq \perp \wedge w_1 \neq w'_1$  occurs. Since  $w_1, w'_1 \neq \perp$ , we have that  $u_1, u'_1 \neq \perp$  (otherwise, the extractor  $\mathcal{E}$  would have outputted  $\perp$ ).

1. By Equation (28), we must have that  $\phi(u_1) = \phi(u'_1)$ , which guarantees the instances share identical commitments  $(c_i)_{i \in [K+k]}$ .
2. Define  $(z_i)_{i \in [K+k]} = w_1$  and  $(z'_i)_{i \in [K+k]} = w'_1$ , and write  $\mathbf z_i,\mathbf z'_i$  for their coefficient embeddings. Then,  $w_1 \neq w'_1$  implies that there exists  $i \in [K+k]$  such that  $\mathbf z_i \neq \mathbf z'_i$ .

During the execution of  $\mathcal{E}(\text{pp}, \mathbf{s}, u_1, \text{st})$ , the call to algorithm  $E(\text{pp}, \mathbf{s}, u_1, \text{st})$  produces elements  $\rho_i, \rho_i^{(i)}, z, z^{(i)}$ . Similarly, during the execution of  $\mathcal{E}(\text{pp}, \mathbf{s}, u'_1, \text{st}')$ , the call to algorithm  $E(\text{pp}, \mathbf{s}, u'_1, \text{st}')$  produces elements  $\rho'_i, \rho'_i^{(i)'}, z', z^{(i)'}$ . Write  $\mathbf z,\mathbf z^{(i)},\mathbf z',\mathbf z^{(i)'}$  for their coefficient embeddings. These elements satisfy

$$\mathbf z_i \neq \mathbf z'_i \iff (\rho_i - \rho_i^{(i)})^{-1} \cdot (\mathbf z - \mathbf z^{(i)}) \neq (\rho'_i - \rho'_i^{(i)'})^{-1} \cdot (\mathbf z' - \mathbf z^{(i)'}) \quad (30)$$

$$\|z\|_\infty < B, \|z^{(i)}\|_\infty < B, \|z'\|_\infty < B, \|z^{(i)'}\|_\infty < B \quad (31)$$

Equation (30) follows from Item 2 and the construction of  $\mathcal{E}$ . Equation (31) follows from the construction of  $\mathcal{E}$ , which only outputs  $w_1, w'_1 \neq \perp$  when the internal extractor  $E$  (from Theorem 10) succeeds. In particular, the internal extractor  $E$  succeeding guarantees that the verification function  $V_{(\text{pp}, \mathbf{s}, u_1, \text{st})}$  accepts. This verification function checks that output tuples (corresponding to Equation (14) and Equation (15)) belong to  $\text{CE}(B, \mathcal{L})$ , which exactly checks the required norm bound on vectors  $z, z^{(i)}, z', z^{(i)'}$ . By Item 1 and Equation (20), we must have

$$c_i = \mathcal{L}\left((\rho_i - \rho_i^{(i)})^{-1} \cdot (\mathbf z - \mathbf z^{(i)})\right) = \mathcal{L}\left((\rho'_i - \rho'_i^{(i)'})^{-1} \cdot (\mathbf z' - \mathbf z^{(i)'})\right)$$

Thus, since  $\mathcal{L}$  is an  $\mathbb R_{\mathbb F}$ -module homomorphism, we have

$$(\rho_i - \rho_i^{(i)}) \cdot c_i = \mathcal{L}(\mathbf z - \mathbf z^{(i)}) \wedge (\rho'_i - \rho'_i^{(i)'}) \cdot c_i = \mathcal{L}(\mathbf z' - \mathbf z^{(i)'}) \quad (32)$$

All together, by Equations (30)--(32), we have that  $(c_i, \Delta_1 = \rho_i - \rho_i^{(i)}, \Delta_2 = \rho'_i - \rho'_i^{(i)'}, \mathbf z_1 = \mathbf z - \mathbf z^{(i)}, \mathbf z_2 = \mathbf z' - \mathbf z^{(i)'})$  is a  $(2B, \mathcal{C})$ -relaxed binding collision (Definition 4).

By assumption (Definition 14),  $\mathcal{L}$  is a ring commitment scheme that satisfies  $(2B, \mathcal{C})$ -relaxed binding. Thus, the probability of the original event (Equation (29)) must be less than or equal to  $\text{negl}(\lambda)$ . Otherwise, we could construct a corresponding relaxed-binding adversary which executes the extractor  $\mathcal{E}$  twice to retrieve the corresponding elements for the  $2B$ -relaxed binding collision with non-negligible probability, contradicting  $\epsilon_{\text{rk}}(2B, \mathcal{C})\le\operatorname{negl}(\lambda)$ .  $\square$

### D.6 $\Pi_{\mathrm{DEC}}$ is a Reduction of Knowledge (Theorem 7)

*Proof. Completeness:* First, we show that the verifier’s checks in step 2 pass. Then, we show that the output belongs to  $\operatorname{BatchCE}_k(b, \mathcal{L})$ .

By the definition of  $\text{CE}(B, \mathcal{L})$ , we must have that  $\|z\|_\infty < B = b^k$  (Definition 14). Thus  $\operatorname{split}_{b,k}(z)$  does not fail and returns digits with  $z = \sum_{i=1}^k b^{i-1} \cdot z_i$ . Write  $\mathbf z,\mathbf z_1,\dots,\mathbf z_k$  for their coefficient embeddings. Therefore,

$$\begin{aligned}
c
&= \mathcal{L}(\mathbf z) \\
&= \mathcal{L}\!\left(\sum_{i=1}^k b^{i-1}\mathbf z_i\right) \\
&= \sum_{i=1}^k b^{i-1}\mathcal{L}(\mathbf z_i) \\
&= \sum_{i=1}^k b^{i-1}c_i.
\end{aligned} \tag{33}$$

The third equality follows from  $\mathcal{L}$  being an  $\mathbb R_{\mathbb F}$ -module homomorphism, and the fourth follows from the construction  $c_i \leftarrow \mathcal{L}(\mathbf z_i)$  in step 1. Similarly, for every  $j \in [t]$ ,

$$\begin{aligned}
\widehat{\bar{M}_j\mathbf z}(r)
&= \widehat{\bar{M}_j\!\left(\sum_{i=1}^k b^{i-1}\mathbf z_i\right)}(r) \\
&= \sum_{i=1}^k b^{i-1}\widehat{\bar{M}_j\mathbf z_i}(r) \\
&= \sum_{i=1}^k b^{i-1}y_{i,j}.
\end{aligned} \tag{34}$$

$$y_j = \sum_{i=1}^k b^{i-1} \cdot y_{i,j} \tag{35}$$

Equation (34) follows from the linearity of the matrix transform, matrix-vector multiplication, and multilinear evaluation, together with the construction  $y_{i,j}\leftarrow\widehat{\bar{M}_j\mathbf z_i}(r)$  in step 1. By Definition 13,  $y_j=\widehat{\bar{M}_j\mathbf z}(r)$ , so Equation (35) follows. Thus, by Equations (33) and (35), the verifier’s checks pass.

Next, we show that the output tuple,  $(s; \{c_i, x_i, r, \{y_{i,j}\}_{j \in [t]}\}_{i \in [k]}; \{z_i\}_{i \in [k]})$ , belongs to  $\operatorname{BatchCE}_k(b, \mathcal{L})$ . By the definition of  $\operatorname{split}_{b,k}$ , we must have that  $\|z_i\|_\infty < b$  for all  $i \in [k]$ . Since  $\mathcal{L}_{\text{in}}$  projects the first  $n_{R,\text{in}}$  ring coordinates and  $\operatorname{split}_{b,k}$  commutes with coordinate projection, the verifier's digits satisfy  $\mathbf{x}_i = \mathcal{L}_{\text{in}}(\mathbf z_i)$  for all  $i \in [k]$ . Together with the construction of  $(c_i, \{y_{i,j}\}_{j \in [t]})_{i \in [k]}$  in step 1 and the shared unchanged point  $r$ , this proves that the output belongs to  $\operatorname{BatchCE}_k(b, \mathcal{L})$ .

**Public coin:** The verifier uses no randomness in this protocol. Thus, the protocol is trivially public coin.

**Knowledge soundness:** Consider an arbitrary expected-polynomial time adversary  $(\mathcal{A}, \mathcal{P}^*)$  for  $\Pi_{\text{DEC}}$  with success probability,  $\epsilon(\mathcal{A}, \mathcal{P}^*) \ge 1/\text{poly}(\lambda)$ . We construct an extractor  $\mathcal{E}$  for  $\Pi_{\text{DEC}}$  as follows,

$\mathcal{E}(\mathbf{pp}, \mathbf{s}, u_1, \mathbf{st}):$

1. Parse  $(c,x,r,(y_j)_{j\in[t]})\leftarrow u_1$ .
2. Execute encoder  $(\mathbf{pk}, \mathbf{vk}) \leftarrow \mathcal{K}(\mathbf{pp}, \mathbf{s})$ .
3. Simulate  $(u_2, w_2) \leftarrow \langle \mathcal{P}^*(\mathbf{pk}, u_1, \mathbf{st}), \mathcal{V}(\mathbf{vk}, u_1) \rangle$ .
4. If  $u_2 = \perp$ , output  $\perp$ .
5. Parse  $(z_1, \dots, z_k) \leftarrow w_2$ .
6. Output  $w_1 := \sum_{i=1}^k b^{i-1} z_i$ .

**Extractor runtime:** The extractor runs in expected polynomial time, since it simulates only one execution between the adversary  $\mathcal{P}^*$  and verifier  $\mathcal{V}$ , which both run in expected polynomial time.

**Extractor success probability:** Assume that the simulated adversary  $(\mathcal{A}, \mathcal{P}^*)$  succeeds in convincing the verifier  $\mathcal{V}$  and the parties jointly output  $(\mathbf{s}, u_2, w_2) \in \operatorname{BatchCE}_k(b, \mathcal{L})$ ; note that this occurs with probability  $\epsilon(\mathcal{A}, \mathcal{P}^*)$ . Define

$$(c_i, x_i, r, (y_{i,j})_{j \in [t]})_{i \in [k]} := u_2 \text{ and } z_1, \dots, z_k := w_2.$$

Write  $\mathbf z_i$  for the coefficient embedding of  $z_i$ .

By the definition of  $\text{CE}(b, \mathcal{L})$ , we have for all  $i \in [k]$  and  $j \in [t]$ ,

$$c_i := \mathcal{L}(\mathbf z_i), \quad \mathbf{x}_i := \mathcal{L}_{\text{in}}(\mathbf z_i), \quad \|z_i\|_{\infty} < b \quad \text{and} \quad y_{i,j} := \widehat{\bar{M}_j\mathbf z_i}(r) \quad (36)$$

Since the adversary convinces the verifier, we must have that for all  $j \in [t]$ ,

$$c = \sum_{i=1}^k b^{i-1} \cdot c_i \quad \text{and} \quad y_j = \sum_{i=1}^k b^{i-1} \cdot y_{i,j} \quad (37)$$

Because the verifier does not reject,  $\operatorname{split}_{b,k}(x)=(x_1,\dots,x_k)$ , so  $x = \sum_{i=1}^k b^{i-1} x_i$ . Define  $z := \sum_{i=1}^k b^{i-1} z_i$  and  $\mathbf z:=\sum_{i=1}^k b^{i-1}\mathbf z_i$ . Equations (36) and (37), together with the input-projection conclusion of Theorem 5, give  $c = \mathcal{L}(\mathbf z)$ ,  $\mathbf{x} = \mathcal{L}_{\text{in}}(\mathbf z)$ , and  $y_j = \widehat{\bar{M}_j\mathbf z}(r)$  for every  $j \in [t]$ . Since  $\|z_i\|_{\infty} < b$  for all  $i$ ,

$$\|z\|_\infty\le\sum_{i=1}^k b^{i-1}(b-1)=b^k-1<B.$$

These are exactly the conditions for  $(\mathbf{s}; u_1; z)$  to belong to  $\text{CE}(B, \mathcal{L})$ . Therefore, the extractor outputs a satisfying witness with probability  $\epsilon(\mathcal{A}, \mathcal{P}^*)$ .  $\square$

### D.7 Finding choices of cyclotomic and fields

```sage
# [LS18, eprint 2017/523], page 6.
def tau(index):
    return index if (index % 2) != 0 else index / 2

# Test the conditions in [LS18, Theorem 1.1].
def prime_support(value):
    return set(p for p, _ in factor(value))

def same_prime_support(left, right):
    return prime_support(left) == prime_support(right)

def thm1_1_cond(index, prime, z):
    return (
        same_prime_support(index, z)
        and (prime % z) == 1
        and Mod(prime, index).multiplicative_order() == index / z
    )

# The l-infinity invertibility bound in [LS18, Theorem 1.1].
def thm1_1_inv_bound(prime, z):
    return (prime^(1 / euler_phi(z)) / sqrt(tau(z))).n()

def thm1_1_num_factors(z):
    return euler_phi(z)

def divisors(index):
    return [z for z in range(1, index + 1) if index % z == 0]

def is_pow2(index):
    return sum(Integer(index).digits(2)) == 1

# [AL21, eprint 2021/202, Proposition 2].
def expansion_factor(index, norm):
    factor = 1 if is_pow2(index) else 2
    return factor * euler_phi(index) * norm

# Return prime-power cyclotomic indices and valid divisors z.
def candidates(prime, min_index=10, max_index=200):
    indices = [i for i in range(min_index, max_index) if len(factor(i)) == 1]
    return [
        (Integer(index), Integer(z))
        for index in indices
        for z in divisors(index)
        if thm1_1_cond(index, prime, z)
    ]

def pre_filter(prime, index, z, challenges):
    degree = cyclotomic_polynomial(index).degree()
    max_difference = challenges[-1] - challenges[0]
    challenge_bits = log(len(challenges)^degree, 2).n()
    return max_difference < thm1_1_inv_bound(prime, z) and challenge_bits >= 120

def info(prime, index, z, kappa, row_domain, challenges):
    phi = cyclotomic_polynomial(index)
    degree = phi.degree()
    ring_length = row_domain // degree
    field_length = degree * ring_length
    challenge_norm = max(abs(challenge) for challenge in challenges)
    max_difference = challenges[-1] - challenges[0]
    delta = 1.0045  # Root-Hermite factor from [ESSLL19, eprint 2018/773].
    expansion = expansion_factor(index, challenge_norm)

    # Analytic MSIS bounds from [MR09] and [CMNW24, eprint 2024/281].
    l2_bound = min(
        prime,
        2^(2 * sqrt(kappa * degree * log(prime, 2) * log(delta, 2))),
    )
    linf_bound = l2_bound / sqrt(field_length)
    protocol_bound = linf_bound / (8 * expansion)

    print("####")
    print("Cyclotomic index:", index)
    print("Cyclotomic polynomial:", phi)
    print("Degree:", degree)
    print("z:", z)
    print("Ring-vector length:", ring_length)
    print("Unused padded rows:", row_domain - field_length)
    print("Expansion factor T:", expansion)
    print("Invertible norm bound:", thm1_1_inv_bound(prime, z))
    print("log2(|C_small|):", log(len(challenges)^degree, 2).n())
    print("log2(analytic protocol B bound):", log(protocol_bound, 2).n())
    print("Number of factors:", thm1_1_num_factors(z))
    print("Challenge differences are invertible:", max_difference < thm1_1_inv_bound(prime, z))
    print()

def possible_settings(prime, kappa, row_domain, challenges):
    for index, z in candidates(prime):
        if pre_filter(prime, index, z, challenges):
            info(prime, index, z, kappa, row_domain, challenges)
        else:
            degree = cyclotomic_polynomial(index).degree()
            print(
                "[Does not satisfy pre-filter] index: {}, degree: {}, z: {}".format(
                    index, degree, z
                )
            )

GL = 2^64 - 2^32 + 1
AGL = GL - 32
M61 = 2^61 - 1

print("##### AGL #####")
possible_settings(AGL, 15, 2^33, [-1, 0, 1, 2])

print("##### M61 #####")
possible_settings(M61, 18, 2^28, [-2, -1, 0, 1, 2])

print("##### GL #####")
possible_settings(GL, 18, 2^30, [-2, -1, 0, 1, 2])
```

### D.8 Lattice Estimator Script

The source used for the reported estimates did not record an estimator revision. Pin the estimator revision and record its configuration before using regenerated output as a reproducible work-factor estimate.

```sage
from estimator import *
Logging.set_level(Logging.LEVEL0)

M61 = 2^61 - 1
GL = 2^64 - 2^32 + 1
AGL = GL - 32

def estimate(label, prime, degree, kappa, row_domain, k, K, expansion):
    b = 2
    B = b^k
    ring_length = row_domain // degree
    field_length = degree * ring_length

    assert field_length <= row_domain
    assert (K + k) * expansion * (b - 1) < B

    params = SIS.Parameters(
        n=kappa * degree,
        q=prime,
        m=field_length,
        length_bound=sqrt(field_length) * (8 * expansion * B),
        norm=2,
    )

    print("#####", label, "#####")
    print("ring-vector length:", ring_length)
    print("field-vector length:", field_length)
    print("unused padded rows:", row_domain - field_length)
    _ = SIS.estimate(params)

estimate("AGL", AGL, 64, 15, 2^33, 13, 50, 128)
estimate("GL", GL, 54, 18, 2^30, 14, 61, 216)
estimate("M61", M61, 54, 18, 2^28, 14, 61, 216)
```
