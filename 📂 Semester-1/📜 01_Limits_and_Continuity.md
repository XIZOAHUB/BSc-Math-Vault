# 📘 Chapter 1: Limits, Continuity, and Differentiability

## 🧠 Why this matters in Data Science & AI?
Before a machine can learn (using derivatives and algorithms like Gradient Descent), we must ensure the mathematical function is "smooth" and unbroken. Limits help us understand the extreme behavior of functions at specific points, which is the foundational step for Calculus and Optimization algorithms.

---

## 1. The Concept of Limits
A limit defines the value that a function approaches as the input approaches some value. It is not about the exact point, but the *neighborhood* around it.

**Mathematical Definition:**
$$\lim_{x \to a} f(x) = L$$
*(This means as $x$ gets closer to $a$, the function $f(x)$ gets closer to $L$.)*

**Left-Hand Limit (LHL):**
$$\lim_{x \to a^-} f(x)$$

**Right-Hand Limit (RHL):**
$$\lim_{x \to a^+} f(x)$$

> **Condition for Existence:** A limit only exists if $$LHL = RHL$$.

---

## 2. Standard Limit Formulas (Cheat Sheet)
Keep these handy for quick substitutions to avoid long calculations.

### Algebraic
* $$\lim_{x \to a} \frac{x^n - a^n}{x - a} = n a^{n-1}$$

### Trigonometric
* $$\lim_{x \to 0} \frac{\sin x}{x} = 1$$
* $$\lim_{x \to 0} \frac{\tan x}{x} = 1$$
* $$\lim_{x \to 0} \cos x = 1$$

### Exponential & Logarithmic
* $$\lim_{x \to 0} \frac{e^x - 1}{x} = 1$$
* $$\lim_{x \to 0} \frac{a^x - 1}{x} = \ln a$$
* $$\lim_{x \to 0} \frac{\ln(1+x)}{x} = 1$$
* $$\lim_{x \to 0} (1 + x)^{\frac{1}{x}} = e$$

---

## 3. L'Hôpital's Rule (The Lifesaver)
Used when direct substitution results in an indeterminate form like $$\frac{0}{0}$$ or $$\frac{\infty}{\infty}$$.

**The Rule:**
$$\lim_{x \to a} \frac{f(x)}{g(x)} = \lim_{x \to a} \frac{f'(x)}{g'(x)}$$
*(Differentiate the numerator and the denominator separately until the indeterminate form is resolved, then apply the limit).*

---

## 4. Continuity
A function $f(x)$ is continuous at a point $x = a$ if there are no breaks, jumps, or holes in its graph. 

**The 3 Golden Conditions for Continuity:**
1. $f(a)$ is strictly defined.
2. $$\lim_{x \to a} f(x)$$ exists (i.e., LHL = RHL).
3. $$\lim_{x \to a} f(x) = f(a)$$

*Note: All polynomial, exponential, logarithmic, and standard trigonometric functions are continuous in their respective domains.*

---

## 5. Differentiability (Gateway to Machine Learning)
A function is differentiable at $x = a$ if it has a defined derivative at that point (geometrically, it means it has a unique tangent line and no "sharp corners").

**Left Hand Derivative (LHD):**
$$LHD = \lim_{h \to 0} \frac{f(a - h) - f(a)}{-h}$$

**Right Hand Derivative (RHD):**
$$RHD = \lim_{h \to 0} \frac{f(a + h) - f(a)}{h}$$

**Condition:** A function is differentiable at $a$ if and only if:
$$LHD = RHD$$

> **Crucial Theorem:** Every differentiable function is continuous, but every continuous function is NOT necessarily differentiable. 
> *Example:* The modulus function $f(x) = |x|$ is continuous at $x=0$, but not differentiable there because it forms a sharp "V" corner.

---

## 💡 AI/ML Real-World Case Study: Activation Functions
In Neural Networks, we use an activation function called **ReLU (Rectified Linear Unit)**:
$$f(x) = \max(0, x)$$

* **Continuity:** It is continuous everywhere.
* **Differentiability:** It is **not differentiable** at exactly $x = 0$ (because it has a sharp corner where LHD $\neq$ RHD). 
* **The Hack:** Modern AI frameworks like TensorFlow and PyTorch handle this seamlessly by forcefully assigning a subgradient value (usually $0$) at $x=0$ so the Neural Network doesn't crash during backpropagation.
