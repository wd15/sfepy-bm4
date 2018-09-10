# Solving Benchmark 4 with Sfepy

<p align="center">
<a href="https://travis-ci.org/wd15/sfepy-bm4" target="_blank">
<img src="https://api.travis-ci.org/wd15/sfepy-bm4.svg"
alt="Travis CI">
</a>
</p>

This repository is an attempt to solve [PFHub's benchmark
4](https://pages.nist.gov/pfhub/benchmarks/benchmark4.ipynb/) with
[Sfepy]. The benchmark is a combined linear elasticity and phase
field example. [Sfepy] hasn't been used much for phase field
problems. The first attempt will solve the linear elasticity with
[Sfepy] and the phase field equation using
[FiPy](https://www.ctcms.nist.gov/fipy/).

## Installation

The installation is set up using [Nix](). Read these instructions and
the Nix manual to get started with Nix. Once nix is installed use

    $ nix-shell

to drop into a shell with everything needed to run the examples
installed. The installation has been tested on Linux and should work
on Mac, but not Windows.

[Sfepy]: http://sfepy.org/doc-devel/index.html