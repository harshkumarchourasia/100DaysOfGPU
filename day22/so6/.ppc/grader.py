#!/usr/bin/env python3

from ppcgrader.cli import cli
import ppcso

if __name__ == "__main__":
    cli(ppcso.Config(code='so6', openmp=False, gpu=True))
