**One word:** function
**Short:** Some object moving, stretching, squishing, etc.

Transformation is a rule that moves every point in a space to a new location, like rotating, stretching, or shifting it, while preserving key features like straight lines or the origin (if Linear Transformation).

A linear transformation moves these basis vectors to new positions 
$$\begin{bmatrix} a \\ c \end{bmatrix} \rightarrow \begin{bmatrix} b \\ d \end{bmatrix}$$

When you know where these two vectors go, you know where any other vector goes because every vector can be expressed as a combination of the basis vectors with same coefficient.

*Why is the coeff needed for multiplying basis vectors for any vector the same before and after the transformation?*
Because in Linear Transformation We have 2 main rules:
- T(u+v)=T(u)+T(v)
- T(c * u)=c * T(u)
=>
$$
T\left(x\begin{bmatrix} 1 \\ 0 \end{bmatrix} + y\begin{bmatrix} 0 \\ 1 \end{bmatrix}\right)
= x\,T\left(\begin{bmatrix} 1 \\ 0 \end{bmatrix}\right) + y\,T\left(\begin{bmatrix} 0 \\ 1 \end{bmatrix}\right).

$$