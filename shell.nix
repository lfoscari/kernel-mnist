with import <nixpkgs> {};

# let keops =
# 	python39.pkgs.buildPythonPackage rec {
# 		pname = "pykeops";
# 		version = "1.5";
# 		src = python39.pkgs.fetchPypi {
# 			inherit pname version;
# 			sha256 = "e7c846bb1fe48f89c4a9660b27a1216dd3478dc1399144e827d9223aa031acd9";
# 		};
# 		propagatedBuildInputs = [ pkgs.python3Packages.numpy pkgs.python3Packages.pytorch ];
# 		doCheck = false;
# 		meta = { };
# 	}; in

let tikzplotlib =
	python39.pkgs.buildPythonPackage rec {
        pname = "tikzplotlib";
        version = "0.10.1";
        format = "pyproject";

        src = fetchFromGitHub {
            owner = "nschloe";
            repo = pname;
            rev = "v${version}";
            sha256 = "sha256-PLExHhEnxkEiXsE0rqvpNWwVZ+YoaDa2BTx8LktdHl0=";
        };

        propagatedBuildInputs = with python39.pkgs; [
            matplotlib
            numpy
            pillow
            webcolors
            flit-core
        ];

        doCheck = false;
	}; in

let kmeans =
	python39.pkgs.buildPythonPackage rec {
        pname = "kmeans_pytorch";
        version = "0.3";
        format = "pyproject";

        src = fetchFromGitHub {
            owner = "subhadarship";
            repo = pname;
            rev = "v${version}";
            sha256 = "sha256-FM2NSpi94hRMyoXlRYU+OfhE2gIT56cRGob0h/YGUO0=";
        };

        propagatedBuildInputs = with python39.pkgs; [
            matplotlib
            numpy
            pillow
            webcolors
            flit-core
        ];

        doCheck = false;
	}; in

let myPythonPackages =
	python39.withPackages (p: with p; [
	    # Perceptron
		pytorch
		torchvision
		kmeans

		# Visualisation
		jupyter
		matplotlib
		tikzplotlib
		pandas

		# Utils
		tqdm
	]); in

pkgs.mkShell {
	buildInputs = with pkgs; [
		myPythonPackages
		jupyter
	];
}

# scikit-learn
# keops
# cmake