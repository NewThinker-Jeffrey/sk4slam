# sk4slam
A Swiss knife for SLAM.



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
