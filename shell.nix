with import <nixpkgs> {};

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

    shellHook = ''
        export PS1='\[\e[0;38;5;129m\]\W \[\e[0m\]& \[\e[0m\]' 

        echo "To repeat the experiments execute experiments.py."
        echo "The results can be found in /results divided by sketching size."
    '';
}