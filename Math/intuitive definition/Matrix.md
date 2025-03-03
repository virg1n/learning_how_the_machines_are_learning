**One word:** action
**Short:** Ordered collection of images of the standard basis after [[Linear Transformation]]
https://www.khanacademy.org/math/precalculus/x9e81a4f98389efdf:matrices/x9e81a4f98389efdf:matrices-as-transformations/a/matrices-as-transformations

In general, any vector [x,y,...,z] can be represented as:

$$
xe_1+ye_2+\dots + z \mathbf{e}_n​
$$

where e_i​ are the standard basis vectors.
After [[Linear Transformation]], it can be represented as
$$
xv_1+yv_2+\dots + z \mathbf{v}_n​
$$
where v_i​ are the new basis vectors after transformation.
Matrix is array of vectors v1, v2, ... vn, which are new basis vectors
$$
\mathbf{Matrix} = \begin{bmatrix} \vec(v_1), \vec(v_2), \vec(v_3), ..., \vec(v_n) \end{bmatrix}
$$

**Example In R2**
$$
\begin{bmatrix} x \\ y \end{bmatrix}
=
x \begin{bmatrix} 1 \\ 0 \end{bmatrix}
+ y \begin{bmatrix} 0 \\ 1 \end{bmatrix}

$$$$
\text{If the arrow } 
\begin{bmatrix} 1 \\ 0 \end{bmatrix}
\text{ lands on some vector } 
\begin{bmatrix} a \\ c \end{bmatrix},

\text{and the another arrow } 
\begin{bmatrix} 0 \\ 1 \end{bmatrix}
\text{ lands on some vector } 
\begin{bmatrix} b \\ d \end{bmatrix},
$$$$

\text{then the vector } 
\begin{bmatrix} x \\ y \end{bmatrix}
\text{ must land on }

 x \cdot \begin{bmatrix} a \\ c \end{bmatrix}
+ y \cdot \begin{bmatrix} b \\ d \end{bmatrix}
= \begin{bmatrix} ax + by \\ cx + dy \end{bmatrix}
$$
A really nice way to describe all this is to represent a given linear transform with the matrix:

$$
\mathbf{A} = \begin{bmatrix} a & b \\ c & d \end{bmatrix}
$$
