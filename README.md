# sk4slam
A Swiss knife for SLAM.


## Introduction to the packages

*Detailed docs are coming... maybe. If I can get around to it...* :P


This repository primarily contains the following packages:

**sk4slam_basic**: (Does not depend on any other packages or third-party libraries) 

- Provides some basic functionalities, including multi-level logging, common string manipulation, time handling, as well as some general macros and basic template classes.

**sk4slam_cpp**: (Depends on sk4slam_basic) 

- Provides some C++ extensions, including memory pools, lock-free data structures (circular queue, hash map, etc), thread pools, task queues, timers, binary search, etc.

**sk4slam_math**: (Depends on sk4slam_basic) 

- Provides commonly used mathematical tools (solving higher-order equations, matrix-related computations, RANSAC, etc.), as well as definitions of basic mathematical objects (linear spaces, manifolds, etc.).

**sk4slam_liegroups**: (Depends on sk4slam_basic, sk4slam_math (manifold)) 

- Provides **a general Lie group framework** and implements **common Lie groups** (SO2/3, SE2/3, Sim2/3, GLn, S1, S3, etc.), supporting the definition of **direct products of Lie groups**, etc. 

- Supported **Lie group operations** include: `Exp()`, `Log()`, `operator*()`, `inverse()`, `Ad()`, `ad()`, `bracket()`, `Jl()` and its inverse `invJl()`, `Jr()` and its inverse `invJr()`, `hat()`, `vee()`, `generator()`, `cast<>()`, etc.

- Provides **additional convenient interfaces** for *matrix groups* and *affine groups*: allowing multiplication with vectors or matrices. 

- Supports **left and right perturbations** for Lie groups, and *for affine groups, it also supports perturbations that separate their linear and translational components*.

**sk4slam_pose**: (Depends on sk4slam_basic, sk4slam_math (manifold), sk4slam_liegroups) 

- Provides common operations for handling poses (SE3), including implementations like PoseWithCov, PoseBuf, tf, etc.

**sk4slam_camera**: (Depends on sk4slam_basic) 

- Provides support for common camera distortion models.

**sk4slam_imu**: (Depends on sk4slam_basic, sk4slam_math (manifold), sk4slam_liegroups, sk4slam_pose) 

- Provides handling of IMU data, such as pre-integration.

**sk4slam_backends**: (Depends on ceres-solver and gtsam, sk4slam_basic, sk4slam_math (manifold), sk4slam_liegroups) 

- Provides **a general factor graph framework interface**, defining the templated (CRTP) base class for factors. The base class includes implementations for interfacing with different optimization libraries (gtsam, ceres-solver). *After this framework is used, each factor only needs to be implemented once, without the need for separate implementations for different underlying optimization libraries*. 

- **Optimized ceres handling of manifolds**: Under the official ceres interface, the cost function must first differentiate with respect to the higher-dimensional ambient space, and then map to the lower-dimensional tangent space. This is inefficient, especially in cases of heavy overparameterization (e.g., using SO3 to represent rotations), and the differentiation of the ambient space is also complex and prone to errors. sk4slam_backends provides optimization for manifold handling in ceres, allowing *differentiation directly in the tangent space*. 

- **Optimized gtsam handling of manifolds**: Although gtsam directly differentiates in the tangent space, each manifold is bound to a fixed Retraction implementation in its underlying design, which can cause issues. For example, when a factor involves a Lie group, and we want to use a different perturbation type (e.g., left or right perturbation), we would have to reimplement that factor. *Our design deliberately separates the manifold definition from the Retraction definition and supports conversion between different Retractions*. Therefore, when using sk4slam_backends, even if a factor is implemented with a specific perturbation type, another perturbation type can be chosen during optimization, and *the framework will automatically handle Jacobian conversion between different retractions*. (About retractions and perturbation types: In our framework, *perturbation types* are simply alternative names for *retractions*, especially when dealing with Lie groups.)

- Implements an **incremental smoother** based on ISAM2 and an adaptive marginalization strategy that balances accuracy and efficiency. 

- Additionally, it provides general factors and commonly used factors in visual SLAM (*additional dependencies*: sk4slam_pose, sk4slam_camera, sk4slam_imu).

**sk4slam_geometry**: (Depends on sk4slam_basic, sk4slam_math (manifold)) 

- Provides solutions for PnP, two-view geometry, and other related problems.

**sk4slam_serial**: (Depends on yaml-cpp, sk4slam_basic) 

- Primarily implements YAML-based serialization and configuration files.

**sk4slam_msgflow**: 

- Encapsulates the message-flow from maplab. Using message-flow to design programs helps decouple different modules effectively.


## Developing

### Prepare workspace and code

```
mkdir catkin_ws
cd catkin_ws
mkdir src
cd src

# catkin_simple
git clone https://github.com/catkin/catkin_simple.git

# sk4slam
git clone --recursive https://github.com/NewThinker-Jeffrey/sk4slam.git
# git clone --recursive git@github.com:NewThinker-Jeffrey/sk4slam.git
```

### Setup the linter

Install python dependencies
```
python3 -m pip install requests pyyaml pylint yapf
```


Install clang-format (See [README for linter](tools/linter/README.md))
For Mac: 
```
brew install clang-format
ln -s /usr/local/share/clang/clang-format-diff.py /usr/local/bin/clang-format-diff
```
For Ubuntu: (Compatible with ```clang-format-3.8 - 6.0```)
```sudo apt install clang-format-${VERSION}```


Initialize linter for the repo:
```
cd catkin_ws/src/sk4slam
python3 tools/linter/bin/init_linter_git_hooks
```


Add the following in your "~/.bashrc" (or other file matching for your shell). Or run this in your working terminal manually before ```git commit```.
```
. tools/linter/setup_linter.sh
```

### Install catkin_tools

https://catkin-tools.readthedocs.io/en/latest/installing.html

### Build

Initialize the worksapce (for the first time):

```
cd catkin_ws
catkin init
```

Build
```
catkin build <target_packge_name>
```

### Run UT

To build & run a catkin test for a specific catkin package:
```
catkin run_tests --no-dep <package_name> 
```

To run a catkin test for a specific catkin package, from a directory within that package:
```
catkin run_tests --no-dep --this
```

For more information: https://catkin-tools.readthedocs.io/en/latest/verbs/catkin_test.html