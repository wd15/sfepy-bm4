{ nixpkgs }:
let
  fipy_download = nixpkgs.fetchFromGitHub {
    owner = "wd15";
    repo = "fipy";
    rev = "c82d781d4e4005040cb51ce1a30f3bc5bcc7917b";
    sha256 = "0an0raxnb6nv30yqq4zj5ybszdkqb99cii6yw72brdrhh0l9dnv5";
  };
in
  import (fipy_download.outPath + "/nix/py3.nix") { inherit nixpkgs; }
