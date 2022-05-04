with import <nixpkgs> {};

let keopscore =
	python39.pkgs.buildPythonPackage rec {
		pname = "keopscore";
		version = "2.0";
		src = python39.pkgs.fetchPypi {
			inherit pname version;
			sha256 = "sha256-lgTkYfhQuH/HWXAYyMfGDOLjOEXURPoxYVnvlmOw3Ms=";
		};
		doCheck = false;
	}; in

let keops =
	python39.pkgs.buildPythonPackage rec {
		pname = "pykeops";
		version = "2.0";
		src = python39.pkgs.fetchPypi {
			inherit pname version;
			sha256 = "sha256-J2amB8LfRf3rtxcVP9jvpCliUdCBq91Po/q+ZwjVkls=";
		};
		propagatedBuildInputs = [ keopscore python39.pkgs.numpy python39.pkgs.pytorch python39.pkgs.pybind11 ];
		doCheck = false;
	}; in

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

let myPythonPackages =
	python39.withPackages (p: with p; [
	    # Perceptron
		pytorch
		torchvision
        keops

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
        jupyter
	];

    shellHook = ''
        export PS1='\[\e[0;38;5;129m\]\W \[\e[0m\]& \[\e[0m\]'

        echo ""
        echo "To repeat the experiments execute experiments.py."
        echo "The results can be found in ./results."
    '';
}
