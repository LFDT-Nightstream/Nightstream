## 6 Strong and weak interactive reductions

**Definition 9 (Weak Interactive Reductions).** Consider relations  $\mathcal{R}_1$ ,  $\mathcal{R}'_1$ , and  $\mathcal{R}_2$  over public parameters, structure, instance, and witness tuples such that  $\mathcal{R}_1 \subseteq \mathcal{R}'_1$ . Let  $\mathcal{U}_1$  be the ambient instance space of  $\mathcal{R}_1$ .

An interactive reduction  $\Pi : \mathcal{R}_1 \to \mathcal{R}_2$ , defined by PPT algorithms  $(\mathcal{G}, \mathcal{K}, \mathcal{P}, \mathcal{V})$  (Definition 5), is **weak** if it is complete, public coin, and there exists a function  $\phi : \mathcal{U}_1 \to \mathbb{C}$  (for an arbitrary space  $\mathbb{C}$ ) such that for any EPT adversary  $(\mathcal{A}, \mathcal{P}^*)$ , there exists an EPT extractor  $\mathcal{E}$  such that if the success probability of the adversary  $\epsilon(\mathcal{A}, \mathcal{P}^*) \ge 1/\text{poly}(\lambda)$ , then

$$\Pr \left[ \left( \text{pp}, s, u_1, w_1 \right) \in \mathcal{R}'_1 \left| \begin{array}{l} \text{pp} \leftarrow \mathcal{G}(1^\lambda, \text{sz}) \\ (s, u_1, \text{st}) \leftarrow \mathcal{A}(\text{pp}) \\ (\text{pk}, \text{vk}) \leftarrow \mathcal{K}(\text{pp}, s) \\ w_1 \leftarrow \mathcal{E}(\text{pp}, s, u_1, \text{st}) \end{array} \right. \right] \ge \epsilon(\mathcal{A}, \mathcal{P}^*) - \text{negl}(\lambda).$$

and if  $\mathcal{A} := (\mathcal{B}, \mathcal{B}')$  such that

$$\Pr \left[ \begin{array}{c} u_1, u'_1 \neq \perp \\ \downarrow \\ \phi(u_1) = \phi(u'_1) \end{array} \left| \begin{array}{l} \text{pp} \leftarrow \mathcal{G}(1^\lambda, \text{sz}) \\ (s, \text{st}^*) \leftarrow \mathcal{B}(\text{pp}) \\ (u_1, \text{st}) \leftarrow \mathcal{B}'(\text{st}^*) \\ (u'_1, \text{st}') \leftarrow \mathcal{B}'(\text{st}^*) \end{array} \right. \right] = 1,$$

then

$$\Pr \left[ \begin{array}{c} w_1, w'_1 \neq \perp \\ \wedge w_1 \neq w'_1 \end{array} \left| \begin{array}{l} \text{pp} \leftarrow \mathcal{G}(1^\lambda, \text{sz}) \\ (s, \text{st}^*) \leftarrow \mathcal{B}(\text{pp}) \\ (u_1, \text{st}) \leftarrow \mathcal{B}'(\text{st}^*) \\ w_1 \leftarrow \mathcal{E}(\text{pp}, s, u_1, \text{st}) \\ (u'_1, \text{st}') \leftarrow \mathcal{B}'(\text{st}^*) \\ w'_1 \leftarrow \mathcal{E}(\text{pp}, s, u'_1, \text{st}') \end{array} \right. \right] \le \text{negl}(\lambda)$$

**Definition 10 (Strong Interactive Reductions).** Consider relations  $\mathcal{R}_1$ ,  $\mathcal{R}_2$ , and  $\mathcal{R}'_2$  over public parameters, structure, instance, and witness tuples such that  $\mathcal{R}_2 \subseteq \mathcal{R}'_2$ . Let  $\mathcal{U}_2$  be the ambient instance space of  $\mathcal{R}_2$ .

An interactive reduction  $\Pi : \mathcal{R}_1 \to \mathcal{R}_2$ , defined by PPT algorithms  $(\mathcal{G}, \mathcal{K}, \mathcal{P}, \mathcal{V})$  (Definition 5), is **strong** if it is complete, public coin, and there exists a function  $\phi : \mathcal{U}_2 \to \mathbb{C}$  (for an arbitrary space  $\mathbb{C}$ ) such that

(i) For any EPT adversary  $(\mathcal{A}, \mathcal{P}^*)$ ,

$$\Pr \left[ \begin{array}{c} u_2, u'_2 \neq \perp \\ \downarrow \\ \phi(u_2) = \phi(u'_2) \end{array} \left| \begin{array}{l} \text{pp} \leftarrow \text{Gen}(1^\lambda) \\ (s, u_1, \text{st}_1) \leftarrow \mathcal{A}(\text{pp}) \\ (\text{pk}, \text{vk}) \leftarrow \mathcal{K}(\text{pp}, s) \\ (u_2, w_2) \leftarrow \langle \mathcal{P}^*, \mathcal{V} \rangle ((\text{pk}, \text{vk}), u_1, \text{st}) \\ (u'_2, w'_2) \leftarrow \langle \mathcal{P}^*, \mathcal{V} \rangle ((\text{pk}, \text{vk}), u_1, \text{st}) \end{array} \right. \right] = 1$$

(ii) For any EPT adversary  $(\mathcal{A}, \mathcal{P}^*)$ , there exists an EPT extractor  $\mathcal{E}$  such that if

$$\epsilon'(\mathcal{A}, \mathcal{P}^*) := \Pr \left[ \left( \text{pp}, s, \langle \mathcal{P}^*, \mathcal{V} \rangle ((\text{pk}, \text{vk}), u_1, \text{st}) \right) \in \mathcal{R}'_2 \left| \begin{array}{l} \text{pp} \leftarrow \mathcal{G}(1^\lambda, \text{sz}) \\ (s, u_1, \text{st}) \leftarrow \mathcal{A}(\text{pp}) \\ (\text{pk}, \text{vk}) \leftarrow \mathcal{K}(\text{pp}, s) \end{array} \right. \right]$$

$\ge 1/\text{poly}(\lambda)$ , and

$$\Pr \left[ \begin{array}{l} w_2, w'_2 \neq \perp \\ \wedge \\ w_2 \neq w'_2 \end{array} \left| \begin{array}{l} \text{pp} \leftarrow \text{Gen}(1^\lambda) \\ (s, u_1, \text{st}) \leftarrow \mathcal{A}(\text{pp}) \\ (\text{pk}, \text{vk}) \leftarrow \mathcal{K}(\text{pp}, s) \\ (u_2, w_2) \leftarrow \langle \mathcal{P}^*, \mathcal{V} \rangle((\text{pk}, \text{vk}), u_1, \text{st}) \\ (u'_2, w'_2) \leftarrow \langle \mathcal{P}^*, \mathcal{V} \rangle((\text{pk}, \text{vk}), u_1, \text{st}) \end{array} \right. \right] \le \text{negl}(\lambda)$$

then we have that

$$\Pr \left[ (\text{pp}, s, u_1, w_1) \in \mathcal{R}_1 \left| \begin{array}{l} \text{pp} \leftarrow \mathcal{G}(1^\lambda, \text{sz}) \\ (s, u_1, \text{st}) \leftarrow \mathcal{A}(\text{pp}) \\ (\text{pk}, \text{vk}) \leftarrow \mathcal{K}(\text{pp}, s) \\ w_1 \leftarrow \mathcal{E}(\text{pp}, s, u_1, \text{st}) \end{array} \right. \right] \ge \epsilon'(\mathcal{A}, \mathcal{P}^*) - \text{negl}(\lambda).$$

**Theorem 6 (Strong-Weak Composition).** Consider relations  $\mathcal{R}_1, \mathcal{R}_2, \mathcal{R}'_2$  and  $\mathcal{R}_3$  over public parameters, structure, instance, and witness tuples such that  $\mathcal{R}_2 \subseteq \mathcal{R}'_2$ . Let  $\mathcal{U}_2$  be the ambient instance space of  $\mathcal{R}_2$ . Consider interactive reductions (Definition 5)  $\Pi_1 : \mathcal{R}_1 \to \mathcal{R}_2$  ( $\mathcal{R}'_2$ ),  $\Pi_2 : \mathcal{R}_2$  ( $\mathcal{R}'_2$ )  $\to \mathcal{R}_3$  such that

- (i)  $\Pi_1$  is **strong** (Definition 10) with respect to a function  $\phi : \mathcal{U}_2 \to \mathbb{C}$  and
- (ii)  $\Pi_2$  is **weak** (Definition 9) with respect to the **same** function  $\phi$ ,

then the sequential composition  $\Pi_2 \circ \Pi_1 : \mathcal{R}_1 \to \mathcal{R}_3$  is a **reduction of knowledge**.

*Proof.* For brevity, we defer the proof to Appendix D.3.  $\square$
