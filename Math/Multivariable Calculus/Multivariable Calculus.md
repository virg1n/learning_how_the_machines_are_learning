Definitions:
[[Vector]], [[Matrix]], [[Linear Transformation]], [[Determinant]], [[Dot Product]], [[Cross Product]]

#### Partial derivatives
Imagine a function f(x,y). The **partial derivative** tells us function's slope **only in the direction of one variable**, treating the other variable as if it were fixed.
$$f(x,y)=x^2y+3y$$
$$\frac{\partial f}{\partial x} = 2xy \text{  |  } \frac{\partial f}{\partial y} = x^2 + 3$$
But, What is it? It is slope of slice of the graph.
Second partial derivative:


$$f(x,y) = \sin(x) y^2$$
$$\frac{\partial f}{\partial x} = \cos(x) y^2 \text{  |  } \frac{\partial f}{\partial y} = \sin(x) (2y)$$
$$\frac{\partial^2 f}{\partial x^2} = f_{xx} =-\sin(x) y^2 \text{  |  } \frac{\partial^2 f}{\partial y \partial x} = f_{xy} = \cos(x) (2y) \text{  |  }  \frac{\partial^2 f}{\partial x \partial y} = f_{yx} = \cos(x) (2y) $$
$$f_{yx} = f_{xy}$$
Why $f_{yx} = f_{xy}$?
	When you take a tiny step in xxx and then in yyy, you're effectively moving along the same small diagonal as you would if you took a step in yyy and then in xxx. For a smooth function, the way the function curves along that diagonal is the same regardless of the order of the steps.

#### [[Gradient]]
$$\text{Gradient of } f(x_1, x_2, ..., x_n) = \nabla f(x_1, x_2, ..., x_n) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$
**What its meaning?**
The gradient points in the direction of steepest ascent and its magnitude tells you how rapidly the function increases in that direction.

**Why?**
	Each component of the gradienttells us **how fast the function fff changes** in the direction of the corresponding variable $x_i$. 
	Suppose you take a small step in some direction represented by a unit vector v = $\begin{bmatrix} {v_1} \\ {v_2} \\ \vdots \\ {v_n} \end{bmatrix}$ 
	The rate at which $f(x_1, ... x_n)$ changes as you move in that direction is given by the **directional derivative**: $\nabla_{} f (\vec{a}) \cdot \vec{v}$ . Whis is maximized when two vectors are aligned (In other words when $\vec{v} = \frac{\nabla_{} f (\vec{a})} {\vert\vert{\nabla_{} f (\vec{a})\vert\vert}}$ ). So, the function increases fastest when you are moving along gradient 
	

The length of [[gradient]] tells us how fast change is. A large gradient means that f is changing rapidly in the steepest direction, while a small gradient means a more gradual slope.

##### Directional derivative
Derivative for vector v, which consists of basis vectors. (How the function changes when the inputs move in v direction)
$$
\frac{\partial f}{\partial \vec{v}} = \nabla_{\vec{v}} f (\vec{a}) = \lim_{h \to 0} \frac{f(\vec{a} + h\vec{v}) - f(\vec{a})}{h}
$$

#### Vector-valued Functions
$$\vec{r(t)} = x(t) \hat{i} + y(t)\hat{j}$$
$$\frac{\partial \vec{r}}{\partial t} = \frac{\partial x}{\partial t} \cdot \hat{i} + \frac{\partial y}{\partial t} \cdot \hat{j}$$
#### Multivariable chain rule

$$\frac{d}{dt} f(x(t), y(t)) = \frac{df}{dt} = \frac{\partial f}{\partial x} \cdot \frac{dx}{dt} + \frac{\partial f}{\partial y} \cdot \frac{dy}{dt}$$

Why? them same intuition as in default chain rule.
	
	$$dx = \frac{dx}{dt} dt$$
	
	$$dy = \frac{dy}{dt} dt$$ Change of x/y is propotinal to change in t. And that propotion is given dy dx/dy (respectively)
	
	$${d_{Due-to-dx}}: \quad df = \frac{\partial f}{\partial x} dx \space | \space
	{d_{Due-to-dy}}: \quad df = \frac{\partial f}{\partial y} dy$$
	
	$$df = \frac{\partial f}{\partial x} \cdot \frac{dx}{dt} \cdot dt + \frac{\partial f}{\partial y} \cdot \frac{dy}{dt} \cdot dt$$


##### Multivariable chain rule in vectors
$$\vec{V}(t) = \begin{bmatrix} x(t) \\ y(t) \end{bmatrix} => \frac{d\vec{V}}{dt} = \begin{bmatrix} \frac{dx}{dt} \\ \frac{dy}{dt} \end{bmatrix}$$
$$\begin{bmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{bmatrix} \cdot 
\begin{bmatrix} \frac{dx}{dt} \\ \frac{dy}{dt} \end{bmatrix} == \nabla f \cdot \vec{V'}(t)$$
$$\frac{d}{dt} f(x(t), y(t)) = \frac{df}{dt} = \nabla f (\vec{V}(t)) \cdot \vec{V'}(t)$$

##### Multivariable chain rule and directional derivatives
https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/multivariable-chain-rule/v/multivariable-chain-rule-and-directional-derivatives
![[Pasted image 20250305112456.png]]


#### Curvature
![[Curvature.excalidraw.svg]]