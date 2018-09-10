# Solving Benchmark 4 with Sfepy

<p align="center">
<a href="https://travis-ci.org/wd15/sfepy-bm4" target="_blank">
<img src="https://api.travis-ci.org/wd15/sfepy-bm4.svg"
alt="Travis CI">
</a>
</p>

This repository is an attempt to solve [PFHub's benchmark
4](https://pages.nist.gov/pfhub/benchmarks/benchmark4.ipynb/) with
[Sfepy]. The benchmark demonstrates combining linear elasticity with a
phase field equation. Since [Sfepy] hasn't been used much for phase
field problems, the first attempt will solve the linear elasticity
with [Sfepy] and the phase field equation using
[FiPy](https://www.ctcms.nist.gov/fipy/). Later version will try and
implement the phase field equation using [Sfepy].

## Installation

The installation is set up using [Nix]. Read [these
instructions](https://github.com/wd15/nixes/blob/master/NIX-NOTES.md)
and the [Nix manual][Nix] to get started with Nix. Once nix is
installed use

    $ nix-shell

to drop into a shell with everything needed to run the examples
installed. The installation has been tested on Linux and should work
on Mac, but not Windows.

[Sfepy]: http://sfepy.org/doc-devel/index.html
[Nix]: https://nixos.org/nix/manual/