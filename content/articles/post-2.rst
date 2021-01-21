Conjugate Gradient Methods
##########################

:summary: We will learn about conjugate gradient methods.
:author: Manoj Baishya
:date: 2021-01-21 12:39
:tags: linear-algebra, optimisation
:category: linear-algebra
:slug: conjugate-gradient


Quadratic Forms and Linear Systems
----------------------------------
Many important quantities arising in engineering are mathematically modelled as Quadratic Forms. Output noise power in Signal Processing, Kinetic and Potential Energy functions in Mechanical Engineering, Covariance Models of Random Variable Linear Dependence and Principal Component Analysis in Statistics, etc. are some of the frequently occurring applications of Quadratic Forms, illustrating their ubiquity and importance.

A Quadratic Form :math:`f(x)` on :math:`\mathbb{R}^n` is a function

.. math::
    f : \mathbb{R}^n \rightarrow \mathbb{R} \text{ such that} \\
    f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^T H \mathbf{x}  + \mathbf{c}^T x \\
    \mathbf{c}, \mathbf{x} \in \mathbb{R}^n, H \in \mathbb{R}^{ n \times n} \text{ for simplicity} \\
    H = H^T, \text{ and } \mathbf{x}^T H \mathbf{x} > 0 \: \forall \: \mathbf{x} \neq 0, \mathbf{x} \in \mathbb{R}^{\, n}

The graph of a positive-definite convex quadratic form is a hyperparaboloid in :math:`\mathbb{R}^{n + 1}` dimensions.

.. image:: |static|/data/qform.svg

.. image:: |static|/data/qform_contour.svg

Frequently, in many applications, we are interested in minimizing the quadratic form :math:`f(x)`, for example, minimising the energy function of a mechanical system so that it is in a stable state.

Suppose that, we have at our hands, a quadratic form obtained by mathematically modelling some quantity of interest and we wish to minimize it. At the minimum point  :math:`\mathbf{x}^*` (henceforth referred to as the minimizer), the gradient :math:`\nabla : \mathbb{R}^n \rightarrow \mathbb{R}` of a function :math:`f(x)` is always zero by first order optimality condition. Hence,

.. math::
    \nabla f(\mathbf{x}) = \frac{1}{2} \left[ H + H^T \right] \mathbf{x} + \mathbf{c} \\
    \nabla f(\mathbf{x}) = H \mathbf{x} + \mathbf{c} \text{ since } H = H^T \\

At :math:`\mathbf{x}^*`, :math:`\nabla f(\mathbf{x}^*) = 0 \implies H \mathbf{x}^* + \mathbf{c} = 0 \text{ or } H \mathbf{x}^* = -\mathbf{c}`

Thus, finding the minimum of the quadratic form :math:`f(\mathbf{x})` is equivalent to solving the linear system :math:`H \mathbf{x}^* = -\mathbf{c}` for the unknown :math:`\mathbf{x}^*`.

Again, suppose that we have at our hands a one-dimensional two-point boundary value problem (BVP), for example, the governing equations of flow of heat in a thin conducting rod with a source (:math:`t` is the continuous 1D spatial domain, :math:`x` is the temperature, :math:`g(t)` is the heat source):

.. math::
    \frac{d^2 x}{dt^2} = g(t) \\

.. image:: |static|/data/heat.png
    :width: 500
    :align: center

If we discretise the domain and convert the above differential equation to a finite difference equation, we obtain a linear system :math:`H \mathbf{x} = - \mathbf{c}`, where the temperatures :math:`\mathbf{x}` can be a very high dimensional vector due to the underlying problem being continuous. The :math:`H` represents the second-order derivative operator :math:`\frac{d^2 x}{dt^2}` and :math:`- \mathbf{c}` the heat source in the discretised domain :math:`\mathbf{R}^n`.

Since the BVP is continuous, the discretisation :math:`x` is made high-dimensional to capture its fidelity; therefore the matrix :math:`H \in \mathbb{R}^{n \times n}` is **very large**. More importantly, due to the structure and regularity of the 2\ :sup:`nd` \ order finite difference operator, we make the following qualifiers: the matrix :math:`H` is **sparse**, **symmetric** and **positive definite**. Solving such a large linear system with a direct matrix factorisation method is prohibitively expensive (of the order of :math:`\mathcal{O}(\frac{2}{3} n^3)`). Hence, a better alternative is to solve the system iteratively and reach a solution in :math:`r << n` steps that is close enough to the actual solution, which will cost us much less than a direct method.

One idea to solve the linear system :math:`H \mathbf{x} = - \mathbf{c}` is to convert it to its dual quadratic form and use an iterative procedure, namely an optimisation process on the objective function.

.. math::
    H \mathbf{x} = - \mathbf{c} \\
    \Leftrightarrow \min_{x \in \mathbb{R}^n} \: f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^T H \mathbf{x}  + \mathbf{c}^T x

At the minimum of :math:`f(\mathbf{x})`, we would have found the solution to our original linear system.

We can summarise this section by making the following statement [Proof A1 in Appendix]:

.. image:: |static|/data/eqv1.svg
    :width: 800
    :align: center

Line Search Techniques
----------------------

Now that we have our objective function, namely the quadratic form :math:`f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^T H \mathbf{x}  + \mathbf{c}^T x`, our task is to minimise it.

Before we begin, let us introduce some definitions:

    1. Error at the i-th iterate: :math:`\mathbf{e}_i = \mathbf{x}_i - \mathbf{x}^*`
    2. Residual at the i-th iterate: :math:`\mathbf{r}_i = H \mathbf{x}_i + \mathbf{c}`
    3. Gradient at the i-th iterate: :math:`\nabla f(\mathbf{x}_i) = H \mathbf{x}_i + \mathbf{c}`

We notice that:
    1. :math:`\mathbf{r}_i = H \mathbf{x}_i + \mathbf{c} = H \mathbf{x}_i - H \mathbf{x}^* = H (\mathbf{x}_i - \mathbf{x}^*) = H \mathbf{e}_i`, that is, the residual is simply the error mapped from the domain to the range of :math:`H`.
    2. The Residual of the Linear System is equal to the Gradient of the Convex Quadratic Objective Function.
    3. Also, since :math:`\mathbf{x}^*` is unknown, :math:`\mathbf{e}_i` are unknown at every step, but the residuals are *always known*. So whenever we want to use the error, we can simply work with the residual in the Range of :math:`H`.

Now comes the core philosophy of Line Search: given our objective function :math:`f(\mathbf{x})`, we start with an initial guess :math:`\mathbf{x}_0`, and iterate our way downhill :math:`f(\mathbf{x})` to reach :math:`\mathbf{x}^*`. At any i-th iterate, we are at the point :math:`\mathbf{x}_i`, and to travel to our next point :math:`\mathbf{x}_{i + 1}`, we must choose a direction of descent :math:`\mathbf{p}_i`, and then move a step length :math:`\alpha_i` the right amount so that along this direction, :math:`f(\mathbf{x}_{i + 1}) = \phi(\alpha_i)` is minimum. Mathematically,

.. math::
    \mathbf{x}_{i + 1} = \mathbf{x}_i + \alpha_i \mathbf{p}_i \\
    \alpha_i = \underset{x_{i + 1}}{\mathrm{argmin}} \: f(\mathbf{x}_{i + 1}) = \underset{\alpha_{i}}{\mathrm{argmin}} \: f(\mathbf{x}_i + \alpha_i \mathbf{p}_i) = \underset{\alpha_{i}}{\mathrm{argmin}} \: \phi(\alpha_i)

