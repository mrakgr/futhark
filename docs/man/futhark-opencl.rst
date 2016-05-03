.. role:: ref(emphasis)

.. _futhark-opencl(1):

==============
futhark-opencl
==============

SYNOPSIS
========

futhark-opencl [-V] [-o outfile] infile

DESCRIPTION
===========

``futhark-opencl`` translates a Futhark program to C code invoking
OpenCL kernels, and then compiles that C code with gcc(1) to an
executable binary program.  The standard Futhark optimisation pipeline
is used, and GCC is invoked with ``-O3``.  The first device of the
first OpenCL platform is used.

The resulting program will otherwise behave exactly as one compiled
with ``futhark-c``.

OPTIONS
=======

-o outfile
  Where to write the resulting binary.  By default, if the source
  program is named 'foo.fut', the binary will be named 'foo'.

-V verbose
  Enable debugging output.  If compilation fails due to a compiler
  error, the result of the last successful compiler step will be
  printed to standard error.

-h
  Print help text to standard output and exit.

-v
  Print version information on standard output and exit.

SEE ALSO
========

futharki(1), futhark-test(1), futhark-c(1)
