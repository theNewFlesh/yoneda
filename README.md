<!-- <img id="logo" src="resources/logo.png" style="max-width: 717px"> -->

[![](https://img.shields.io/badge/License-MIT-F77E70?style=for-the-badge)](https://github.com/theNewFlesh/yoneda/blob/master/LICENSE)
[![](https://img.shields.io/pypi/pyversions/yoneda?style=for-the-badge&label=Python&color=A0D17B&logo=python&logoColor=A0D17B)](https://github.com/theNewFlesh/yoneda/blob/master/docker/config/pyproject.toml)
[![](https://img.shields.io/pypi/v/yoneda?style=for-the-badge&label=PyPI&color=5F95DE&logo=pypi&logoColor=5F95DE)](https://pypi.org/project/yoneda/)
[![](https://img.shields.io/pypi/dm/yoneda?style=for-the-badge&label=Downloads&color=5F95DE)](https://pepy.tech/project/yoneda)

# Introduction

Categeory theory in python.

See [documentation](https://theNewFlesh.github.io/yoneda/) for details.

# Installation
### Python
`pip install yoneda`

### Docker
1. Install [docker-desktop](https://docs.docker.com/desktop/)
2. `docker pull theNewFlesh/yoneda:[version]`

### Docker For Developers
1. Install [docker-desktop](https://docs.docker.com/desktop/)
2. Ensure docker-desktop has at least 4 GB of memory allocated to it.
3. `git clone git@github.com:theNewFlesh/yoneda.git`
4. `cd yoneda`
5. `chmod +x bin/yoneda`
6. `bin/yoneda docker-start`

The service should take a few minutes to start up.

Run `bin/yoneda --help` for more help on the command line tool.

### ZSH Setup

1. `bin/yoneda` must be run from this repository's top level directory.
2. Therefore, if using zsh, it is recommended that you paste the following line
    in your ~/.zshrc file:
    - `alias yoneda="cd [parent dir]/yoneda; bin/yoneda"`
    - Replace `[parent dir]` with the parent directory of this repository
3. Running the `zsh-complete` command will enable tab completions of the cli
   commands, in the next shell session.

   For example:
   - `yoneda [tab]` will show you all the cli options, which you can press
     tab to cycle through
   - `yoneda docker-[tab]` will show you only the cli options that begin with
     "docker-"

---

# Quickstart Guide
This repository contains a suite commands for the whole development process.
This includes everything from testing, to documentation generation and
publishing pip packages.

These commands can be accessed through:

  - The VSCode task runner
  - The VSCode task runner side bar
  - A terminal running on the host OS
  - A terminal within this repositories docker container

Running the `zsh-complete` command will enable tab completions of the CLI.
See the zsh setup section for more information.

### Command Groups

Development commands are grouped by one of 10 prefixes:

| Command    | Description                                                                        |
| ---------- | ---------------------------------------------------------------------------------- |
| build      | Commands for building packages for testing and pip publishing                      |
| docker     | Common docker commands such as build, start and stop                               |
| docs       | Commands for generating documentation and code metrics                             |
| library    | Commands for managing python package dependencies                                  |
| session    | Commands for starting interactive sessions such as jupyter lab and python          |
| state      | Command to display the current state of the repo and container                     |
| test       | Commands for running tests, linter and type annotations                            |
| version    | Commands for bumping project versions                                              |
| quickstart | Display this quickstart guide                                                      |
| zsh        | Commands for running a zsh session in the container and generating zsh completions |

### Common Commands

Here are some frequently used commands to get you started:

| Command           | Description                                               |
| ----------------- | --------------------------------------------------------- |
| docker-restart    | Restart container                                         |
| docker-start      | Start container                                           |
| docker-stop       | Stop container                                            |
| docs-full         | Generate documentation, coverage report, diagram and code |
| library-add       | Add a given package to a given dependency group           |
| library-graph-dev | Graph dependencies in dev environment                     |
| library-remove    | Remove a given package from a given dependency group      |
| library-search    | Search for pip packages                                   |
| library-update    | Update dev dependencies                                   |
| session-lab       | Run jupyter lab server                                    |
| state             | State of                                                  |
| test-dev          | Run all tests                                             |
| test-lint         | Run linting and type checking                             |
| zsh               | Run ZSH session inside container                          |
| zsh-complete      | Generate ZSH completion script                            |

---

# Development CLI
bin/yoneda is a command line interface (defined in cli.py) that
works with any version of python 2.7 and above, as it has no dependencies.
Commands generally do not expect any arguments or flags.

Its usage pattern is: `bin/yoneda COMMAND [-a --args]=ARGS [-h --help] [--dryrun]`

### Commands
The following is a complete list of all available development commands:

| Command                 | Description                                                         |
| ----------------------- | ------------------------------------------------------------------- |
| build-package           | Build production version of repo for publishing                     |
| build-prod              | Publish pip package of repo to PyPi                                 |
| build-publish           | Run production tests first then publish pip package of repo to PyPi |
| build-test              | Build test version of repo for prod testing                         |
| docker-build            | Build Docker image                                                  |
| docker-build-from-cache | Build Docker image from cached image                                |
| docker-build-prod       | Build production image                                              |
| docker-container        | Display the Docker container id                                     |
| docker-destroy          | Shutdown container and destroy its image                            |
| docker-destroy-prod     | Shutdown production container and destroy its image                 |
| docker-image            | Display the Docker image id                                         |
| docker-prod             | Start production container                                          |
| docker-pull-dev         | Pull development image from Docker registry                         |
| docker-pull-prod        | Pull production image from Docker registry                          |
| docker-push-dev         | Push development image to Docker registry                           |
| docker-push-dev-latest  | Push development image to Docker registry with dev-latest tag       |
| docker-push-prod        | Push production image to Docker registry                            |
| docker-push-prod-latest | Push production image to Docker registry with prod-latest tag       |
| docker-remove           | Remove Docker image                                                 |
| docker-restart          | Restart Docker container                                            |
| docker-start            | Start Docker container                                              |
| docker-stop             | Stop Docker container                                               |
| docs                    | Generate sphinx documentation                                       |
| docs-architecture       | Generate architecture.svg diagram from all import statements        |
| docs-full               | Generate documentation, coverage report, diagram and code           |
| docs-metrics            | Generate code metrics report, plots and tables                      |
| library-add             | Add a given package to a given dependency group                     |
| library-graph-dev       | Graph dependencies in dev environment                               |
| library-graph-prod      | Graph dependencies in prod environment                              |
| library-install-dev     | Install all dependencies into dev environment                       |
| library-install-prod    | Install all dependencies into prod environment                      |
| library-list-dev        | List packages in dev environment                                    |
| library-list-prod       | List packages in prod environment                                   |
| library-lock-dev        | Resolve dev.lock file                                               |
| library-lock-prod       | Resolve prod.lock file                                              |
| library-remove          | Remove a given package from a given dependency group                |
| library-search          | Search for pip packages                                             |
| library-sync-dev        | Sync dev environment with packages listed in dev.lock               |
| library-sync-prod       | Sync prod environment with packages listed in prod.lock             |
| library-update          | Update dev dependencies                                             |
| library-update-pdm      | Update PDM                                                          |
| quickstart              | Display quickstart guide                                            |
| session-lab             | Run jupyter lab server                                              |
| session-python          | Run python session with dev dependencies                            |
| session-server          | Runn application server inside Docker container                     |
| state                   | State of repository and Docker container                            |
| test-coverage           | Generate test coverage report                                       |
| test-dev                | Run all tests                                                       |
| test-fast               | Test all code excepts tests marked with SKIP_SLOWS_TESTS decorator  |
| test-lint               | Run linting and type checking                                       |
| test-prod               | Run tests across all support python versions                        |
| version                 | Full resolution of repo: dependencies, linting, tests, docs, etc    |
| version-bump-major      | Bump pyproject major version                                        |
| version-bump-minor      | Bump pyproject minor version                                        |
| version-bump-patch      | Bump pyproject patch version                                        |
| version-commit          | Tag with version and commit changes to master                       |
| zsh                     | Run ZSH session inside Docker container                             |
| zsh-complete            | Generate oh-my-zsh completions                                      |
| zsh-root                | Run ZSH session as root inside Docker container                     |

### Flags

| Short | Long      | Description                                          |
| ----- | --------- | ---------------------------------------------------- |
| -a    | --args    | Additional arguments, this can generally be ignored  |
| -h    | --help    | Prints command help message to stdout                |
|       | --dryrun  | Prints command that would otherwise be run to stdout |

