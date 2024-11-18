{ lib
, mkShell
# , vulkan-loader
, python3Packages
# , wayland
}:

mkShell rec {
  name = "impurePythonEnv";

  venvDir = "./env";
  packages = with python3Packages; [
    venvShellHook
    python

    gymnasium
    matplotlib
    scipy
    pillow
    pandas
    torch
    stable-baselines3
    optuna
    tensorboard
    pygame
    debugpy
  ];

  postVenvCreation = ''
    unset SOURCE_DATE_EPOCH
    pip install -r ${./requirements.txt}
  '';
  postShellHook = "unset SOURCE_DATE_EPOCH";
}
