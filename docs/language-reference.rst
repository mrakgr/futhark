.. _language-reference:

Language Reference
==================

The primitive types in Futhark are the signed integer types ``i8``,
``i16``, ``i32``, ``i64``, the unsigned integer types ``u8``, ``u16``,
``u32``, ``u64``, the floating-point types ``f32``, ``f64``, as well
as ``bool``.  Furthermore, ``int`` is an alias for
``i32``.  An ``f32`` is always a single-precision float and a ``f64``
is a double-precision float.  All primitive types can be combined in
tuples and arrays.

Numeric literals can be suffixed with their intended type.  For
example ``42i8`` is of type ``i8``, and ``1337e2f64`` is of type
``f64``.  If no suffix is given, integer literals are of type ``i32``,
and decimal literals are of type ``f64``.

Numeric values can be converted between different types by using the
desired type name as a function.  E.g., ``i32(1.0f32)`` would convert
the floating-point number ``1.0`` to a 32-bit signed integer.

Character and string literals are supported, but only as an alias for
integers and arrays of integers, respectively.  There is no character
data type.

The following list describes every syntactical language construct in
the language.  For convenience, we will sometimes talk of expressions
"modifying" the input.  As Futhark is a pure language, what happens is
of course that a new array is returned instead of the original being
modified.

Function Declarations
---------------------

A function declaration must specify the return type, name, parameters,
and body of the function::

  fun rettype_type name(params...) = body

Type inference is not supported, and functions are fully
monomorphic.  Optionally, the programmer may put *shape declarations*
in the return type and parameter types.  These can be used to express
invariants about the shapes of arrays that are accepted or produced by
the function, e.g::

  fun [int,n] f([int,n] a) =
    map(+1, a)

The above declaration specifies a function that takes an array
containing ``n`` elements and returns an array likewise containing
``n`` elements.  In general, shape declarations in parameters are
fresh names, whilst shape declarations in return types must refer to a
name of type ``i32`` in scope.  A shape declaration can also be an
integer constant (with no suffix).

The same name can be used in several dimensions, or even in several
parameters.  This can be used to give a natural type to a function for
computing dot products::

  fun int dotProduct([int,n] a, [int,n] b) =
    reduce(+, 0, zipWith(*, a, b))

Or matrix multiplication::

  fun [[int,n],n] matMult([[int,m],n] x, [[int,n],m] y) =
    ...

The dimension names bound in a parameter shape declaration can be used
as ordinary variables inside the scope of the parameter.

Shape declarations serve two main purposes:

1. They document the shape assumptions of the function in an easily
   understandable form.

2. More importantly, they help the compiler understand the invariants
   of the program, which it may otherwise have trouble figuring out.

Note that adding shape declarations is never unsafe - the compiler
still inserts dynamic checks, so if an incorrect declaration is made,
the result will merely be an abrubt but controlled termination as it
collides with reality.  Shape declarations matter most when used for
the input parameters of the ``main`` function and for the return type
of functions used to ``map``.

Apart from declaring a function with the keyword ``fun``, it can also
be declared with ``entry``.  This means that if the Futhark program is
compiled as a library instead of an executable program, that function
will be exposed as an entry point.  Any function named ``main`` will
always be considered an entry point, whether it is declared with
``entry`` or not.

Type Aliases
------------

Futhark supports simple type aliases to improve code readability.
Examples::

  type person_id = int
  type int_pair  = (int, int)
  type vec3 = (f32, f32, f32)

  type pilot = person_id
  type passengers = [person_id]
  type position = vec3
  type velocity = vec3
  type mass     = f32

  type airplane = (pilot, passengers, position, velocity, mass)

The aliases are merely a syntactic convenience.  With respect to type
checking the ``position`` and ``velocity`` types are identical.  It is
currently not possible to put shape declarations in type aliases.
When using uniqueness attributes with type aliases, inner uniqueness
attributes are overrided by outer ones::

  type uniqueInts = *[int]
  type nonuniqueIntLists = [intlist]
  type uniqueIntLists = *nonuniqueIntLists

  -- Error: using non-unique value for a unique return value.
  fun uniqueIntLists (nonuniqueIntLists p) = p

*Dimension declarations*

To declare dimensions on an array data type using type aliases, the type alias must
define either a primitive type, or a tuple.

File Inclusions
---------------

You can include external Futhark code into a Futhark file like this::

  include module

The above will include all functions from whatever ``module`` is and make them
available in the current Futhark program.

All include headers must be at the top of the Futhark file, before any function
declarations.

Currently, Futhark can only include files.  You can include a file into your
main Futhark program like this::

  include other_file

The ``.fut`` extension is implied, so the above will include the file
``other_file.fut``.

You can also include files from subdirectories::

  include path.to.a.file

The above will include the file ``path/to/a/file.fut``.

Simple Expressions
------------------

*constant*
~~~~~~~~~~

Evaluates to itself.

*variable*
~~~~~~~~~~

Evaluates to its value in the environment.

``x`` *arithop* ``y``
~~~~~~~~~~~~~~~~~~~~~

Evaluate the binary arithmetic operator on its operands, which must
both be of the same numeric type.  The following operators are
supported: ``+``, ``*``, ``-``, ``/``, ``%``, ``//``, ``%%``, ``==``,
``!=`` ``<``, ``<=``, ``**``.

``x`` *bitop* ``y``
~~~~~~~~~~~~~~~~~~~

Evaluate the binary bitwise operator on its operands, which must both
be of integer type.  The following operators are supported: ``^``,
``&``, ``|``, ``>>``, ``<<``, ``>>>``, i.e., bitwise xor, and, or,
arithmetic shift right and left, and logical shift right.  Shift
amounts must be non-negative.

``x && y``
~~~~~~~~~~

Logical conjunction; both operands must be of type ``bool``.  Not
short-circuiting, as this complicates program transformation.  If
short-circuiting behaviour is desired, the programmer can use ``if``
explicitly.

``x || y``
~~~~~~~~~~

Logical disjunction; both operands must be of type ``bool``.  As with
``&&``, not short-circuiting.

``! x``
~~~~~~~~~

Logical negation of ``x``, which must be of type ``bool``.

``- x``
~~~~~~~

Numerical negation of ``x``, which must be of numeric type.

``~ x``
~~~~~~~

Bitwise negation of ``x``, which must be of integral type.

``abs x``
~~~~~~~~~

Absolute value of ``x``, which must be of integral type.

``signum x``
~~~~~~~~~~~~

Sign of ``x``, which must be of an integral type.  Returns 1, 0, or
-1.

``a[i]``
~~~~~~~~

Return the element at the given position in the array.  The index may
be a comma-separated list of indexes instead of just a single index.
If the number of indices given is less than the rank of the array, an
array is returned.

``zip(x, y, z)``
~~~~~~~~~~~~~~~~~~

Zips together the elements of the outer dimensions of arrays ``x``,
``y``, and ``z``.  Static or runtime check is performed to check that
the sizes of the outermost dimension of the arrays are the same.  If
this property is not true, program execution stops with an error.  Any
number of arrays may be passed to ``unzip``.  If *n* arrays are given,
the result will be a single-dimensional array of *n*-tuples (where the
the tuple components may themselves be arrays).

``unzip(a)``
~~~~~~~~~~~~

If the type of ``a`` is ``[(t_1, ..., t_n)]``, the result is a tuple
of *n* arrays, i.e., ``([t_1], ..., [t_n])``, and otherwise a type
error.

``unsafe e``
~~~~~~~~~~~~

Elide safety checks (such as bounds checking) for operations lexically
with ``e``.  This is useful if the compiler is otherwise unable to
avoid bounds checks (e.g. when using indirect indexes), but you really
do not want them here.

``iota(n)``
~~~~~~~~~~~

An array of the integers from ``0`` to ``n-1``.

``replicate(n, x)``
~~~~~~~~~~~~~~~~~~~

An array consisting of ``n`` copies of ``a``.

``size(i, a)``
~~~~~~~~~~~~~~

The size of dimension ``i`` of array ``a``, where ``i`` is a static
integer constant.

``split((i_1, ..., i_n), a)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Partitions the given array ``a`` into ``n+1`` disjoint arrays
``(a[0...i_1-1], a[i_1...i_2-1], ..., a[i_n...])``, returned as a tuple.
The split indices must be weakly ascending, ie ``i_1 <= i_2 <= ... <= i_n``.

Example: ``split((1,1,3), [5,6,7,8]) == ([5],[],[6,7],[8])``

``concat(a_1, ..., a_n)``
~~~~~~~~~~~~~~~~~~~~~~~~~

Concatenate the rows/elements of several arrays.  The shape of the
arrays must be identical in all but the first dimension.

``copy(a)``
~~~~~~~~~~~
Return a deep copy of the argument.  Semantically, this is just
the identity function, but it has special semantics related to
uniqueness types as described in :ref:`uniqueness-types`.

``reshape((d_1, ..., d_n), a)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reshape the elements of ``a`` into an ``n``-dimensional array of the
specified shape.  The number of elements in ``a`` must be equal to the
product of the new dimensions.

``rearrange((d_1, ..., d_n), a)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Permute the dimensions in the array, returning a new array.

For example, if ``b==rearrange((2,0,1),a)``, then ``b[x,y,z] =
a[y,z,x]``.

``transpose(a)``
~~~~~~~~~~~~~~~~

Return the transpose of ``a``, which must be a two-dimensional array.


``let pat = e in body``
~~~~~~~~~~~~~~~~~~~~~~~

Evaluate ``e`` and bind the result to the pattern ``pat`` while
evaluating ``body``.  The ``in`` keyword is optional if ``body`` is a
``let`` or ``loop`` expression.

``let dest = src with [i] <- v in body``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluate ``body`` with ``dest`` bound to the value of
``src``, except that the element(s) at the position given by the
index take on the value of ``v``.  The given index need not be
complete, but in that case, the value of ``v`` must be an array
of the proper size.

``if c then a else b``
~~~~~~~~~~~~~~~~~~~~~~

If ``c`` evaluates to ``True``, evaluate ``a``, else evaluate ``b``.

``loop (pat = initial) = for i < bound do loopbody in body``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The name ``i`` is bound here and initialised to zero.

1. Bind ``pat`` to the initial values given in ``initial``.

2. While ``i < bound``, evaluate ``loopbody``, rebinding ``pat`` to be
   the value returned by the body, increasing ``i`` by one after each
   iteration.

3. Evaluate ``body`` with ``pat`` bound to its final
   value.

The ``= initial`` can be left out, in which case initial values for
the pattern are taken from equivalently named variables in the
environment.  I.e., ``loop (x) = ...`` is equivalent to ``loop (x = x)
= ...``.

``loop (pat = initial) = while cond do loopbody in body``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Bind ``pat`` to the initial values given in ``initial``.

2. While ``cond`` evaluates to true, evaluate ``loopbody``, rebinding
   ``pat`` to be the value returned by the body.

3. Evaluate ``body`` with ``pat`` bound to its final value.

Parallel Expressions
--------------------

It is not guaranteed that the the parallel constructs in Futhark are
evaluated in parallel, especially if they are nested in complicated
ways.  Their purpose is to give the compiler as much freedom and
information is possible, in order to enable it to maximise the
parallelism of the generated code.

``map(f, a)``
~~~~~~~~~~~~~

Apply ``f`` to every element of ``a`` and return the resulting array.

``zipWith(f, a_1, ..., a_n)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Syntactic sugar for ``map(f, zip(a_1, ..., a_n))``.

``reduce(f, x, a)``
~~~~~~~~~~~~~~~~~~~

Left-reduction with ``f`` across the elements of ``a``, with ``x`` as
the neutral element for ``f``.  The function ``f`` must be
associative.  If it is not, the evaluation result is not defined.

``scan(f, x, a)``
~~~~~~~~~~~~~~~~~~~

Inclusive prefix scan.  Has the same caveats with respect to
associativity as ``reduce``.

``filter(f, a)``
~~~~~~~~~~~~~~~~

Remove all those elements of ``a`` that do not satisfy the predicate
``f``.

``partition(f_1, ..., f_n, a)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Divide the array ``a`` into disjoint partitions based on the given
predicates.  Each element of ``a`` is called with the predicates
``f_1`` to ``f_n`` in sequence, and as soon as one as one of them
returns ``True``, the element is added to the corresponding partition.
If none of the functions return ``True``, the element is added to a
catch-all partition that is returned last.  Always returns a tuple
with *n+1* components.  The partitioning is stable, meaning that
elements of the partitions retain their original relative positions.

``write(indexes, values, a)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set each index of the ``indexes`` array in the ``a`` array to each value of
the ``values`` array.  If an index is -1, ignore it and its associated value.
Return the modified array.  It is an error if there are duplicate indexes.
``write`` does its work in-place and consumes ``a``.

Tuple Shimming
--------------

In a SOAC, if the given function expects *n* arguments of types
``t_1=, ..., t_n``, but the SOAC will call the function with a
single argument of type ``(t_1, ..., t_n)`` (that is,
a tuple), the Futhark compiler will automatically generate an anonymous
unwrapping function.  This allows the following expression to
type-check (and run)::

  map(+, zip(as, bs))

Without the tuple shimming, the above would cause an error, as ``+``
is a function that takes two arguments, but is passed a two-element
tuple by ``map``.

Arrays of Tuples
----------------

For reasons related to code generation and efficient representation,
arrays of tuples are in a sense merely syntactic sugar for tuples of
arrays.  The type ``[(int, f32)]`` is transformed to ``([int],
[f32])`` during the compilation process, and all code interacting
with arrays of tuples is likewise transformed.  In most cases, this is
fully transparent to the programmer, but there are edge cases where
the transformation is not trivially an isomorphism.

Consider the type ``[([int], [f32])]``, which is transformed
into ``([[int]], [[f32]])``.  These two types are not
isomorphic, as the latter has more stringent demands as to the
fullness of arrays.  For example::

  [
    ([1],   [1.0]),
    ([2,3], [2.0])
  ]

is a value of the former, but the first element of the
corresponding transformed tuple::

  (
    [[1],   [2, 3]],
    [[1.0], [2.0]]
  )

is not a full array.  Hence, when determining whether a program
generates full arrays, we must hence look at the *transformed*
values - in a sense, the fullness requirement "transcends" the tuples.

Literal Defaults
----------------

By default, Futhark interprets integer literals as ``i32`` values, and decimal
literals (integer literals containing a decimal point) as ``f64`` values. These
defaults can be changed using the `Haskell-inspired
<https://wiki.haskell.org/Keywords#default>`_ ``default`` keyword.

To change the ``i32`` default to e.g. ``i64``, type the following at the top of
your file::

  default(i64)

To change the ``f64`` default to ``f32``, type the following at the top of your
file::

  default(f32)

To change both, type::

  default(i64,f32)
