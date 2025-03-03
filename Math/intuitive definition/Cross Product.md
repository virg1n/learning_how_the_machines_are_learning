**One word:** perpendicular
**Short:** For two vectors in 3D, the cross product gives a new vector that is perpendicular to both, with a length equal to the area of the parallelogram they span.

**It points exactly perpendicular to the plane containing $\vec{a} \text{ and } \vec{b}$.**
Why? 
	- The cross product is specifically **constructed** to be perpendicular to both a and b. (In other words, to be $\vec(a)⋅\vec(c)=0 \text{ and } \vec(b)⋅\vec(c)=0$)

Length of cross product:
$$∥a×b∥=∥a∥∥b∥sinθ$$
It captures how strong the perpendicular (or "orthogonal") interaction is between the two vectors.

**This Length Equal the Area of the Parallelogram**
Why?
	- If you take b⃗ as the base, the corresponding height is the perpendicular distance from the tip of a⃗ (when a⃗ is “shifted” to start at the tail of b⃗) down to the line along b⃗. This height is exactly ∥a⃗∥sin⁡θ $$\text{Area} = \text{base} \times \text{height} = \|\mathbf{b}\| \times (\|\mathbf{a}\|\sin\theta) = \|\mathbf{a}\|\|\mathbf{b}\|\sin\theta.$$

Formula:
$$
\mathbf{a} \times \mathbf{b} =
\begin{vmatrix}
\mathbf{i} & \mathbf{j} & \mathbf{k} \\
a_1 & a_2 & a_3 \\
b_1 & b_2 & b_3
\end{vmatrix}
=
\left( a_2b_3 - a_3b_2 \right)\mathbf{i}
- \left( a_1b_3 - a_3b_1 \right)\mathbf{j}
+ \left( a_1b_2 - a_2b_1 \right)\mathbf{k}.
$$
