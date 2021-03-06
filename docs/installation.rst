.. _installation:

Installation
============

There are two ways to install the Futhark compiler: using a
precompiled tarball or compiling from source.  Both methods are
discussed below.  If you are using Windows, make sure to read
:ref:`windows-installation`.

Compiling from source
---------------------

We use the the `Haskell Tool Stack`_ to handle dependencies and
compilation of the Futhark compiler, so you will need to install the
``stack`` tool.  Fortunately, the ``stack`` developers provide ample
documentation about `installing Stack`_ on a multitude of operating
systems.  If you're lucky, it may even be in your local package
repository.

We do not presently issue source releases of Futhark, so the only way
to compile from source is to perform a checkout of our Git
repository::

  $ git clone https://github.com/HIPERFIT/futhark.git

This will create a directory ``futhark``, which you must enter::

  $ cd futhark

To get all the prerequisites for building the Futhark compiler
(including, if necessary, the appropriate version of the Haskell
compiler), run::

  $ stack setup

Note that this will not install anything system-wide and will have no
effect outside the Futhark build directory. Now you can run the
following command to build the Futhark compiler, including all
dependencies::

  $ stack build

The Futhark compiler and its tools will now be built. You can copy
them to your ``$HOME/.local/bin`` directory by running::

  $ stack install

Note that this does not install the Futhark manual pages.

Installing from a precompiled snapshot
--------------------------------------

We do not yet have any proper releases as such, but every day a
program automatically clones the Git repository, builds the compiler,
and packages a simple tarball containing the resulting binaries, built
manpages, and a simple ``Makefile`` for installing.  The implication
is that these tarballs are not vetted in any way, nor more stable than
Git HEAD at any particular moment in time.  They are provided merely
for users who are unable or unwilling to compile Futhark themselves.

At the moment, we build such snapshots only for a single operating
system:

Linux (x86_64)
  `futhark-nightly-linux-x86_64.tar.xz <https://futhark-lang.org/releases/futhark-nightly-linux-x86_64.tar.xz>`_

In time, we hope to make snapshots available for more platforms, but
we are limited by system availability.  We also intend to make proper
releases once the language matures.

.. _`Haskell tool stack`: http://docs.haskellstack.org/
.. _`installing Stack`: http://docs.haskellstack.org/#how-to-install

.. _windows-installation:

Installing Futhark on Windows
-----------------------------

While the Futhark compiler itself is easily installed on Windows via
``stack`` (see above), it takes a little more work to make the OpenCL
and PyOpenCL backends functional.  This guide was last updated on the
5th of May 2016, and is for computers using 64-bit Windows along with
`CUDA 7.5`_ and Python 2.7 (`Anaconda`_ preferred).

Also `Git for Windows`_ is required for its Linux command line tools.
If you have not marked the option to add them to path, there are
instructions below how to do so. The GUI alternative to ``git``,
`Github Desktop`_ is optional and does not come with the required
tools.

.. _`CUDA 7.5`: https://developer.nvidia.com/cuda-downloads
.. _`Anaconda`: https://www.continuum.io/downloads#_windows
.. _`Git for Windows`: https://git-scm.com/download/win
.. _`Github Desktop`: https://desktop.github.com/

Setting up Futhark and OpenCL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1) Clone the Futhark repository to your hard drive.

2) Install `Stack`_ using the 64-bit installer.  Compile the Futhark
   compiler as described in :ref:`installation`.

3) For editing environment variables it is strongly recommended that
   you install the `Rapid Environment Editor`_

4) For a Futhark compatible C/C++ compiler, that you will also need to
   install pyOpenCL later, install MingWpy. Do this using the ``pip
   install -i https://pypi.anaconda.org/carlkl/simple mingwpy``
   command.

5) Assuming you have the latest Anaconda distribution as your primary
   one, it will get installed to a place such as
   ``C:\Users\UserName\Anaconda2\share\mingwpy``. The pip installation
   will not add its bin or include directories to path.

   To do so, open the Rapid Environment Editor and add
   ``C:\Users\UserName\Anaconda2\share\mingwpy\bin`` to the system-wide
   ``PATH`` variable.

   If you have other MingW or GCC distributions, make sure MingWpy takes
   priority by moving its entry above the other distributions. You can
   also change which Python distribution is the default one using the
   same trick should you need so.

   If have done so correctly, typing ``where gcc`` in the command prompt
   should list the aforementioned MingWpy installation at the top or show
   only it.

   To finish the installation, add the
   ``C:\UserName\Marko\Anaconda2\share\mingwpy\include`` to the ``CPATH``
   environment variable (note: *not* ``PATH``). Create the variable if
   necessary.

6) The header files and the .dll for OpenCL that comes with the CUDA
   7.5 distribution also need to be installed into MingWpy.  Go to
   ``C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include``
   and copy the ``CL`` directory into the MingWpy ``include`` directory.

   Next, go to ``C:\Program Files\NVIDIA Corporation\OpenCL`` and copy
   the ``OpenCL64.dll`` file into the MingWpy ``lib`` directory (it is
   next to ``include``).

   The CUDA distribution also comes with the static ``OpenCL.lib``, but
   trying to use that one instead of the ``OpenCL64.dll`` will cause
   programs compiled with ``futhark-opencl`` to crash, so ignore it
   completely.

Now you should be able to compile ``futhark-opencl`` and run Futhark
programs on the GPU.

Congratulations!

.. _`Stack`: http://docs.haskellstack.org/en/stable/install_and_upgrade/#windows
.. _`Rapid Environment Editor`: http://www.rapidee.com/en/about

Setting up PyOpenCL
~~~~~~~~~~~~~~~~~~~

The following instructions are for how to setup the
``futhark-pyopencl`` backend.

First install Mako using ``pip install mako``.

Also install PyPNG using ``pip install pypng`` (not stricly necessary,
but some examples make use of it).

7) Clone the `PyOpenCL repository`_ to your hard drive. Do
   this instead of downloading the zip, as the zip will not contain
   some of the other repositories it links to and you will end up with
   missing header files.

8) If you have ignored the instructions and gotten Python 3.x instead
   2.7, you will have to do some extra work.

   Edit ``.\pyopencl\compyte\ndarray\gen_elemwise.py`` and
   ``.\pyopencl\compyte\ndarray\test_gpu_ndarray.py`` and convert most
   Python 2.x style print statements to Python 3 syntax. Basically wrap
   print arguments in brackets "(..)" and ignore any lines containing
   StringIO ``>>`` operator.

   Otherwise just go to the next point.

9) Go into the repo directory and from the command line execute
   ``python configure.py``.

   Edit ``siteconf.py`` to following::

     CL_TRACE = False
     CL_ENABLE_GL = False
     CL_INC_DIR = ['c:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v7.5\\include']
     CL_LIB_DIR = ['C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v7.5\\lib\\x64']
     CL_LIBNAME = ['OpenCL']
     CXXFLAGS = ['-std=c++0x']
     LDFLAGS = []

   Run the following commands::

     > python setup.py build_ext --compiler = mingw32
     > python setup.py install

If everything went in order, pyOpenCL should be installed on your machine now.

10) Lastly, Pygame needs to be installed.  Again, not stricly
    necessary, but some examples make use of it.  To do so on Windows,
    download ``pygame-1.9.2a0-cp27-none-win_amd64.whl`` from `here
    <http://www.lfd.uci.edu/~gohlke/pythonlibs/#pygame>`_. ``cp27``
    means Python 2.7 and ``win_amd64`` means 64-bit Windows.

    Go to the directory you have downloaded the file and execute ``pip
    install pygame-1.9.2a0-cp27-none-win_amd64.whl`` from the command
    line.

Now you should be able to run the `Mandelbrot Explorer`_ and and `Game of Life`_ examples.

11) To run the makefiles, first setup ``make`` by going to the ``bin``
    directory of MingWpy and making a copy of
    ``mingw32-make.exe``. Then simply rename ``mingw32-make –
    Copy.exe`` or similar to ``make.exe``. Now you will be able to run
    the makefiles.

    Also, if you have not selected to add the optional Linux command
    line tools to ``PATH`` during the ``Git for Windows``
    installation, add the ``C:\Program Files\Git\usr\bin`` directory
    to ``PATH`` manually now.

12) This guide has been written off memory, so if you are having
    difficulties - ask on the `issues page`_. There might be errors in
    it.

.. _`PyOpenCL repository`: https://github.com/pyopencl/pyopencl
.. _`Mandelbrot Explorer`: https://github.com/HIPERFIT/futhark-benchmarks/tree/master/misc/mandelbrot-explorer
.. _`Game of Life`: (https://github.com/HIPERFIT/futhark-benchmarks/tree/master/misc/life)
.. _`issues page`: https://github.com/HIPERFIT/futhark/issues
