#!/bin/bash
clearml-session --docker "quay.io/jax79sg/dockerindocker:v1 --privileged" --packages "clearml" "nvidia-pyindex" "nvidia-tlt" --queue 1gpu
