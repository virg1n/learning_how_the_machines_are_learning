**One word:** scaling
**Short:** It measures the change in “volume” (area in 2D, volume in 3D, etc.) induced by a linear transformation. + showing any flip in orientation.

Determinant tells how much the “size” of any shape (like a unit square in 2D or a unit cube in 3D) is scaled by [[Linear Transformation]].

**Example in R2:**
- We have a square with area = 1
- The transformation sends the basis vectors

$$
\begin{bmatrix} 1 \\ 0 \end{bmatrix} \to \begin{bmatrix} a \\ c \end{bmatrix}, \quad \begin{bmatrix} 0 \\ 1 \end{bmatrix} \to \begin{bmatrix} b \\ d \end{bmatrix}
$$

	These images form the sides of a parallelogram.

- The area of that parallelogram is **Determinant**
	- $∣det(A)∣=∣ad−bc∣$
	
	- If $∣ad−bc∣>1$, the transformation enlarges areas.
	- If $∣ad−bc∣<1$, it shrinks areas.
	- The **sign** of $ad−b$c indicates whether the transformation preserves the original orientation (positive) or flips it (negative).

## Why area of that parallelogram == Determinant?
### Proof with [[Cross Product]]
$$
\vec{v} = \begin{bmatrix} a \\ c \end{bmatrix} \quad \text{and} \quad \vec{w} = \begin{bmatrix} b \\ d \end{bmatrix}
$$
Area = $∥v∥⋅(∥w∥sinθ)$
Lets append zero to these vectors
$$\vec{v} = \begin{bmatrix} a \\ c \\ 0 \end{bmatrix} \quad \text{and} \quad \vec{w} = \begin{bmatrix} b \\ d \\ 0 \end{bmatrix}$$
[[Cross Product]] of v and w $= \begin{bmatrix} 0 \\ 0 \\ ad - bc \end{bmatrix}$

Also, [[Cross Product]] of v and w = $∥v∥∥w∥sinθ == Area of parallelogram$
so, 
$$ad - bc = ∥v∥∥w∥sinθ = \text{Area of parallelogram}$$

### Proof with [[Dot Product]] and Projections
$$
\vec{v} = \begin{bmatrix} a \\ c \end{bmatrix} \quad \text{and} \quad \vec{w} = \begin{bmatrix} b \\ d \end{bmatrix}
$$
1) The length of v⃗  (Base of parallelogram) is $∥v∥=\sqrt{a^2+c^2}$
2) The height is the component of w⃗ perpendicular to v⃗
	 $$\vec{u}_\perp = \frac{1}{\|\vec{v}\|} \begin{bmatrix} -c \\ a \end{bmatrix} = \frac{1}{\sqrt{a^2 + c^2}} \begin{bmatrix} -c \\ a \end{bmatrix}$$
	Height = absolute of projection of w⃗ onto this perpendicular direction. We can compute it with dot product
		$$\text{Height = }\vec{w} \cdot \vec{u}_\perp = \begin{bmatrix} b \\ d \end{bmatrix} \cdot \frac{1}{\sqrt{a^2 + c^2}}\begin{bmatrix} -c \\ a \end{bmatrix} = \frac{1}{\sqrt{a^2 + c^2}} (-cb + ad).$$
3) Final Area (Base * Height)
		$$A = \sqrt{a^2 + c^2} \times \frac{|ad - bc|}{\sqrt{a^2 + c^2}}.$$
		
#### Why dot Product works here?
$$w⋅u=∥w∥∥u∥cosθ$$
If ||u|| = 1:

$$w⋅u=∥w∥cosθ$$
cosθ == 1 when they are in the same direction (and projection of w == w), and then its decreasing