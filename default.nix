let
  nixpkgs = import ./nix/nixpkgs_version.nix;
  pypkgs = nixpkgs.python36Packages;
  pytest-cov = import ./nix/pytest-cov.nix { inherit nixpkgs pypkgs; };
  sfepy = import ./nix/sfepy.nix { inherit nixpkgs pypkgs; };
  toml = import ./nix/toml.nix { inherit pypkgs; };
  black = import ./nix/black.nix { inherit pypkgs toml; };
  nbval = import ./nix/nbval.nix { inherit nixpkgs pypkgs; };
in
  pypkgs.buildPythonPackage rec {
    name = "sfepy-env";
    env = nixpkgs.buildEnv { name=name; paths=buildInputs; };
    buildInputs =  [
      pypkgs.pip
      pypkgs.numpy
      pypkgs.scipy
      pypkgs.pytest
      pypkgs.matplotlib
      pypkgs.sympy
      pypkgs.cython
      pypkgs.jupyter
      pytest-cov
      nbval
      nixpkgs.pkgs.git
      pypkgs.tkinter
      pypkgs.setuptools
      sfepy
      pypkgs.toolz
      pypkgs.pylint
      pypkgs.flake8
      black
      nixpkgs.python36Packages.tkinter
      pypkgs.ipywidgets
      pypkgs.appdirs
      pypkgs.click
      toml
    ];
    src=./.;
    catchConflicts=false;
    doCheck=false;
    preShellHook = ''
      jupyter nbextension install --py widgetsnbextension --user
      jupyter nbextension enable widgetsnbextension --user --py
    '';
    postShellHook = ''
       SOURCE_DATE_EPOCH=$(date +%s)
       export PYTHONUSERBASE=$PWD/.local
       export USER_SITE=`python -c "import site; print(site.USER_SITE)"`
       export PYTHONPATH=$PYTHONPATH:$USER_SITE
       export PATH=$PATH:$PYTHONUSERBASE/bin
       # pip install --user toolz
     '';
  }
