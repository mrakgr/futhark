.. _language-overview:

Language Overview
=================

This document has been prepared as an informal overview of the Futhark
language.  In time, we hope to develop it into a formal specification,
with accompanying proofs of correctness, but for now, words will have
to suffice where formulae would be ideal.

The Futhark programming language is a purely functional,
call-by-value, mostly first-order language that permits bulk
operations on arrays using *second-order array combinators* (SOACs).

The primary idea behind Futhark is to design a language that has
enough expressive power to conveniently express complex programs, yet
is also amenable to aggressive optimisation and parallelisation.
Unfortunately, as the expressive power of a language grows, the
difficulty of optimisation often rises likewise.  For example, we
support nested parallelism, despite the complexities of efficiently
mapping to the flat parallelism supported by hardware, as many
algorithms are awkward to write without this feature.  On the other
hand, we do not support non-regular arrays, as they complicate size
analysis a great deal.  The fact that Futhark is purely functional is
intended to give an optimising compiler more leeway in rearranging the
code and performing high-level optimisations.  It is also the plan to
eventually design a rigorous cost model for Futhark, although this
work has not yet been completed.

Lexical Syntax
--------------

The syntax of Futhark is derived from Haskell and Standard ML,
although somewhat less flexible.  Futhark is not whitespace-sensitive,
and indentation is only used for readability.  An identifier starts
with a letter, followed by any number of letters, digits and
underscores.  Numeric, string and character literals use the same
notation as Haskell (which is very similar to C), including all escape
characters.  Comments are indicated with ``--`` and span to end of
line.  Block comments are not presently supported.

First-order Futhark
-------------------

An Futhark program consists of a sequence of *function definitions*,
of the following form::

   fun return_type name(params...) = body

A function must declare both its return type and the types of all its
parameters.  All functions (except for inline anonymous functions; see
below) are defined globally.  Futhark does not use type inference.
Symbolic constants are not supported, although 0-ary functions can be
defined.  As a concrete example, here is the recursive definition of
the factorial function in Futhark::

  fun int fact(int n) =
    if n == 0 then 1
              else n * fact(n-1)

Indentation has no syntactical significance in Futhark, but recommended for
readability.

The syntax for tuple types is a comma-separated list of types or
values enclosed in braces, so ``{int, real}`` is a pair of an integer
and a floating-point number.  Both single-element and empty tuples are
permitted.  Array types are written as the element type surrounded by
brackets, meaning that ``[int]`` is a one-dimensional array of
integers, and ``[[[{int, real}]]]`` is a three-dimensional array of
tuples of integers and floats.  An immediate array is written as a
sequence of elements enclosed by brackets::

  [1, 2, 3]       -- Array of type [int].
  [[1], [2], [3]] -- Array of type [[int]].

All arrays must be *regular* (often termed *full*).  This means that,
for example, all rows of a two-dimensional array must have the same
number of elements::

  [[1, 2], [3]]      -- Compile-time error.
  [iota(1), iota(2)] -- A run-time error if reached.

The restriction to regular arrays simplifies size analysis and
optimisation.

Arrays are indexed using the common row-major notation, e.g., ``a[i1,
i2, i3...]``.  An indexing is said to be *full* if the number of given
indexes is equal to the dimensionality of the array.

A ``let``-expression can be used to refer to the result of a
subexpression::

  let z = x + y in ...

Recall that Futhark is eagerly evaluated, so the right-hand side of
the ``let`` is evaluated exactly once, at the time it is first
encountered.  The ``in`` keyword is optional when it precedes another
``let``.  This means that instead of writing::

  let a = 0 in
  let b = 1 in
  let c = 2 in
  a + b + c

we can write::

  let a = 0
  let b = 1
  let c = 2
  in a + b + c

The final ``in`` is still necessary.

Two-way ``if-then-else`` is the only branching construct in Futhark.
Pattern matching is supported in a limited way for taking apart
tuples, but this can only be done in ``let``-bindings, and not
directly in a function argument list.  Specifically, the following
function definition is not valid::

  fun int sumpair({int, int} {x, y}) = x + y -- WRONG!

Instead, we must use a ``let``-binding explicitly, as follows::

  fun int sumpair({int, int} t) =
    let {x,y} = t in x + y

Pattern-matching in a binding is the only way to access the components
of a tuple.

Function calls are written as the function name followed by the
arguments enclosed in parentheses.  All function calls must be fully
saturated - currying is only permitted in SOACs_.

Sequential Loops
~~~~~~~~~~~~~~~~

Futhark has a built-in syntax for expressing certain tail-recursive
functions.  Consider the following tail-recursive formulation of a
function for computing the Fibonacci numbers::

  fun int fib(int n) = fibhelper(1,1,n)

  fun int fibhelper(int x, int y, int n) =
    if n == 1 then x else fibhelper(y, x+y, n-1)

We can rewrite this using the ``loop`` construct::

  fun int fib(int n) =
    loop ({x, y} = {1,1}) = for i < n do
                              {y, x+y}
    in x

The semantics of this is precisely as in the tail-recursive function
formulation.  In general, a loop::

  loop (pat = initial) = for i < bound do loopbody
  in body

Has the following the semantics:

1. Bind *pat* to the initial values given in *initial*.

2. While *i < bound*, evaluate *loopbody*, rebinding *pat* to be the
   value returned by the body.  At the end of each iteration, increment
   *i* by one.

3. Evaluate *body* with *pat* bound to its final value.

Semantically, a ``loop`` expression is completely equivalent to a
call to its corresponding tail-recursive function.

For example, denoting by ``t`` the type of ``x``, this loop::

  loop (x = a) =
    for i < n do
      g(x)
    in body

has the semantics of a call to this tail-recursive function::

  fun t f(int i, int n, t x) =
    if i >= n then x
       else f(i+1, n, g(x))

  let x = f(i, n, a)
  in body

The purpose of ``loop`` is partly to render some sequential
computations slightly more convenient, but primarily to express
certain very specific forms of recursive functions, specifically those
with a fixed iteration count.  This property is used for analysis and
optimisation by the Futhark compiler.

Apart from the ``i < n`` form, which loops from zero, Futhark also
supports the ``v <= i < n`` form which starts at ``v``.  We can also
invert the order of iteration by writing ``n > i`` or ``n > i >= v``,
which loops down from the upper bound to the lower.

Apart from ``for``-loops, Futhark also supports ``while`` loops.
These do not provide as much information to the compiler, but can be
used for convergence loops, where the number of iterations cannot be
predicted in advance.  For example, the following program doubles a
given number until it exceeds a given threshold value::

  fun int main(int x, int bound) =
    loop (x) = while x < bound do x * 2
    in x

In all respects other than termination criteria, ``while``-loops
behave identically to ``for``-loops.

For brevity, the initial value expression can be elided, in which case
an expression equivalent to the pattern is implied.  This is easier to
understand with an example.  The loop::

  fun int fib(int n) =
    let x = 1
    let y = 1
    loop ({x, y} = {x, y}) = for i < n do {y, x+y}
    in x

can also be written::

  fun int fib(int n) =
    let x = 1
    let y = 1
    loop ({x, y}) = for i < n do {y, x+y}
    in x

This can sometimes make imperative code look more natural.

In-Place Updates
~~~~~~~~~~~~~~~~

In an array-oriented programming language, a common task is to modify
some elements of an array.  In a pure language, we cannot permit free
mutation, but we can permit the creation of a duplicate array, where
some elements have been changed.  General modification of array
elements is done using the ``let-with`` construct.  In its most
general form, it looks as follows::

  let dest = src with [indexes] <- value
  in body

This evaluates ``body`` with ``dest`` bound to the value of ``src``,
except that the element(s) at the position given by ``indexes`` take
on the new value ``value``.  The given indexes need not be complete,
but in that case, ``value`` must be an array of the proper size.  As
an example, here's how we could replace the third row of an ``n * 3``
array::

  let b = a with [2] <- [1,2,3] in b

Yes, this is the *third* binding construct in the language, ignoring
function abstraction!  As a convenience, whenever ``dest`` and ``src``
are the same, we can write::

    let dest[indexes] = value in body

as a shortcut.  Note that this has no special semantic meaning, but is
simply a case of normal name shadowing.

For example, this loop implements the "imperative" version of matrix
multiplication of two ``N * N`` matrices::

  fun *[[int]] matmultImp(int N, [[int]] a, [[int]] b) =
      let res = replicate(N, iota(N)) in
      loop (res) = for i < N do
          loop (res) = for j < N do
              let partsum =
                  let res = 0 in
                  loop (res) = for k < N do
                      let res = res + a[i,k] * b[k,j]
                      in  res
                  in res
              in let res[i,j] = partsum in res
          in res
      in res

With the naive implementation based on copying the source array,
executing the ``let-with`` expression would require memory
proportional to the entire source array, rather than proportional to
the slice we are changing.  This is not ideal.  Therefore, the
``let-with`` construct has some unusual restrictions to permit
in-place modification of the ``src`` array, as described in
:ref:`uniqueness-types`.  Simply put, we track that ``src`` is never used
again.  The consequence is that we can guarantee that the execution of
a ``let-with`` expression does not involve any copying of the source
array in order to create the newly bound array, and therefore the time
required for the update is proportional to the section of the array we
are updating, not the entire array.  We can think of this as similar
to array modification in an imperative language.

SOACs
-----

The language presented in the previous section is in some sense
"sufficient", in that it is Turing-complete, and can express
imperative-style loops in a natural way with ``do`` and
``while``-loops.  However, Futhark is not intended to be used in this
way - bulk operations on arrays should be expressed via one of the
*second-order array combinators* (SOACs) shown below, as this
maximises the amount of parallelism that the compiler is able to take
advantage of.

.. productionlist::
   e: "map" "(" `lambda` "," `e` ")"
    : "zipWith" "(" `lambda` "," `e` "," ... "," `e` ")"
    : "filter" "(" `lambda` "," `e` ")"
    : "partition" "(" `lambda` "," ... `lambda` "," `e` ")"
    : "reduce" "(" `lambda` "," `e` "," `e` ")"
    : "scan" "(" `lambda` "," `e` "," `e` ")"

A lambda can be an anonymous function, the name of a function (with
optional curried arguments), or an operator (possibly with one operand
curried):

.. productionlist::
   lambda: "fn" `rettype` (`param`...) "=>" `e`
         : `fname`
         : `fname` (`e`, ..., `e`)
         : `op` `e`
         : `e` `op`
         : `op`

The semantics of the SOACs is identical to the similarly-named
higher-order functions found in many functional languages.  For
specifics, see :ref:`language-reference`.

The ``scan`` SOAC performs an inclusive prefix scan, and returns an
array of the same outer size as the original array.  The functions
given to ``reduce`` and ``scan`` must be binary associative operators,
and the value given as the initial value of the accumulator must be
the neutral element for the function.  These properties are not
checked by the Futhark compiler, and are the responsibility of the
programmer.
