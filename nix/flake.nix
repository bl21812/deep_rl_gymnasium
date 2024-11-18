{
  description = "An awesome machine-learning project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    utils.url = "github:numtide/flake-utils";

    ml-pkgs.url = "github:howird/ml-pkgs/howird/maniskill3";
    ml-pkgs.inputs.nixpkgs.follows = "nixpkgs";
    ml-pkgs.inputs.utils.follows = "utils";
  };

  outputs = {
    self,
    nixpkgs,
    ...
  } @ inputs:
    {
      overlays.default = nixpkgs.lib.composeManyExtensions [
        inputs.ml-pkgs.overlays.default
      ];
    } // inputs.utils.lib.eachSystem [ "x86_64-linux" ] (system:
    let
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
          cudaCapabilities = [ "7.5" "8.6" ];
          cudaForwardCompat = true;
        };
        overlays = [
          self.overlays.default
        ];
      };
    in {
      devShells = {
        default = pkgs.callPackage ./dev-shells {};
      };
    });
}
