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
        version = "f7f36bd1cb4e3a761d73d584866d0a9c6b4d2805";
        format = "pyproject";

        src = fetchFromGitHub {
            owner = "subhadarship";
            repo = pname;
            rev = "f7f36bd1cb4e3a761d73d584866d0a9c6b4d2805";
            sha256 = "sha256-WQ1t52qkcTODIhDUJKu0A0hcfDh0YPJ8AVwrzxbFwfI=";
        };

        propagatedBuildInputs = with python39.pkgs; [
            matplotlib
            numpy
            pillow
            webcolors
            flit-core
            numba
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
        ipykernel

		# Utils
		tqdm
	]); in

pkgs.mkShell {
	buildInputs = with pkgs; [
		myPythonPackages
	];

    shellHook = ''
        export PS1='\[\e[0;38;5;129m\]\W \[\e[0m\]& \[\e[0m\]'

        echo ""
        echo "To repeat the experiments execute experiments.py."
        echo "The results can be found in ./results."
    '';
}
