Least Squares Data Fitting
##########################

:summary: We will learn how to apply the least squares linear regression technique to problems in modelling relationships in data.
:author: Manoj Baishya
:date: 2021-01-02 11:15
:tags: linear-algebra, statistics, data
:category: linear-algebra
:slug: least-squares-data-fitting

The goal of Least Squares is to find a mathematical model, or an approximate model, of some relationship, given some observed data or measurements or sampled signals.

.. image:: |static|/data/wop50-09.svg
    :height: 450 px
    :width: 600 px
    :alt: Least Squares model of World Oil Production data from 1950 to 2009

Formulation
-----------

Suppose we have an n-vector :math:`x`, and a scalar :math:`y`, and we believe that they are related, perhaps approximately, by some function :math:`f : \mathbb{R}^n → \mathbb{R}`:

.. math::
    y \approx f(x)

The vector :math:`x` might represent a set of n feature values, and is called the *feature vector* or the vector of *independent variables*, depending on the context. The scalar :math:`y` represents some *outcome* (also called *response variable*) that we are interested in.