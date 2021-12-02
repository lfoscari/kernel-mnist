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

let kmeans =
	python39.pkgs.buildPythonPackage rec {
		pname = "kmeans_pytorch";
		version = "0.3";
		src = python39.pkgs.fetchPypi {
			inherit pname version;
			sha256 = "c0e7279078f5592c0a80a836897efd1567c3275544e7d0ad844bff24053d8e78";
		};
		doCheck = false;
		meta = { };
	}; in

let myPythonPackages =
	python39.withPackages (p: with p; [
		pytorch
		torchvision
		kmeans
		tqdm
		scikit-learn
	]); in

pkgs.mkShell {
	buildInputs = with pkgs; [
		myPythonPackages
		jupyter
		# keops
		# cmake
	];
}
