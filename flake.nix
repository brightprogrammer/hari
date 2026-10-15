{
  description = "Hari - Custom Harness, Written for 'Making Agents Work For You' Series";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        nativeDeps = with pkgs; [
          pkg-config
          cmake
          ninja
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs =
            nativeDeps
            ++ (with pkgs; [
              llvmPackages_21.clang
              llvmPackages_21.libclang
              llvmPackages_21.llvm
              libxml2
              libffi
              (python3.withPackages (ps: [
                ps.transformers
                ps.openai
              ]))
            ]);
        };
      }
    );
}
