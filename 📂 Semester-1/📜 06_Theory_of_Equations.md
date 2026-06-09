
# 📘 Chapter 6: Theory of Equations

## 🧠 Why this matters in Data Science & AI?
In Machine Learning, training a model is essentially an "Optimization Problem." We want to find the point where the error (Cost Function) is at its absolute minimum. Mathematically, the minimum of a curve occurs where its derivative is exactly zero. Therefore, finding the minimum error means finding the *roots* of the derivative equation! Understanding how polynomial equations behave, how many roots they have, and how to approximate them is the foundation of algorithmic optimization.

---

## 1. The Fundamental Theorem of Algebra
This theorem sets the baseline for understanding polynomial equations.

**Statement:** Every polynomial equation of degree $n$ with real or complex coefficients has exactly $n$ roots (which can be real or complex, and some may be repeated).

**General Equation:**
$$P(x) = a_0 x^n + a_1 x^{n-1} + a_2 x^{n-2} + \dots + a_n = 0$$
*(Where $a_0 \neq 0$ and $n$ is a positive integer).*

---

## 2. Relations Between Roots and Coefficients (Vieta's Formulas)
This is the most frequently tested concept in exams and helps solve equations without actually finding the individual roots.

Let the roots of the general equation of degree $n$ be $\alpha_1, \alpha_2, \dots, \alpha_n$.

* **Sum of the roots taken one at a time:**
  $$\sum \alpha_i = -\frac{a_1}{a_0}$$
* **Sum of the products of roots taken two at a time:**
  $$\sum \alpha_i \alpha_j = \frac{a_2}{a_0}$$
* **Sum of the products of roots taken three at a time:**
  $$\sum \alpha_i \alpha_j \alpha_k = -\frac{a_3}{a_0}$$
* **Product of all $n$ roots:**
  $$\alpha_1 \alpha_2 \dots \alpha_n = (-1)^n \frac{a_n}{a_0}$$

### Standard Examples:
**For a Cubic Equation** $ax^3 + bx^2 + cx + d = 0$ with roots $\alpha, \beta, \gamma$:
* $\alpha + \beta + \gamma = -\frac{b}{a}$
* $\alpha\beta + \beta\gamma + \gamma\alpha = \frac{c}{a}$
* $\alpha\beta\gamma = -\frac{d}{a}$

---

## 3. Symmetric Functions of Roots
A function of roots is called symmetric if its value does not change when any two roots are interchanged. We can evaluate these using Vieta's formulas without knowing the actual roots.

**Important Cheat Sheet (For roots $\alpha, \beta, \gamma$):**
* $$\sum \alpha^2 = (\sum \alpha)^2 - 2\sum \alpha\beta$$
* $$\sum \alpha^3 = (\sum \alpha)^3 - 3(\sum \alpha)(\sum \alpha\beta) + 3\alpha\beta\gamma$$
* $$\sum \frac{1}{\alpha} = \frac{\sum \alpha\beta}{\alpha\beta\gamma}$$

---

## 4. Descartes' Rule of Signs
This rule gives us a quick way to find the maximum possible number of positive and negative real roots of a polynomial just by looking at the equation!

Let $f(x) = 0$ be a polynomial equation with real coefficients.

1. **Positive Real Roots:** The number of positive real roots cannot exceed the number of sign changes in $f(x)$.
2. **Negative Real Roots:** The number of negative real roots cannot exceed the number of sign changes in $f(-x)$.
3. **Complex Roots:** If the maximum possible real roots (positive + negative) are less than the degree $n$, the remaining roots must be imaginary (complex roots always occur in conjugate pairs like $a + ib$ and $a - ib$).

---

## 5. Transformation of Equations
Sometimes, it's easier to solve a modified equation rather than the original one. If original roots are $\alpha, \beta, \gamma \dots$ 

* **Roots multiplied by a constant $m$:** Multiply the coefficients of $x^n, x^{n-1}, x^{n-2} \dots$ by $1, m, m^2 \dots$ respectively.
* **Reciprocal Roots ($\frac{1}{\alpha}, \frac{1}{\beta} \dots$):** Replace $x$ with $\frac{1}{x}$ in the original equation.
* **Roots with opposite signs ($-\alpha, -\beta \dots$):** Change the sign of the coefficients of all odd powers of $x$.

---

## 💡 AI/ML Real-World Case Study: Root Finding Algorithms
In the real world of computer science, exact analytical formulas (like the quadratic formula) don't exist for polynomials of degree 5 or higher (Abel-Ruffini Theorem). 

So, how do AI algorithms find roots when optimizing a neural network? They use **Iterative Numerical Methods**!

The most famous is the **Newton-Raphson Method**. Instead of algebra, it uses Calculus to guess the root. The AI starts with a random guess $x_0$ and applies this formula repeatedly:
$$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$$
*(Where $f(x_n)$ is the function value and $f'(x_n)$ is the derivative).*

This simple equation allows a computer to approximate the root of any complex polynomial in just a few milliseconds, making it a critical tool in Machine Learning optimization libraries!
