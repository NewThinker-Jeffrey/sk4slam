# sk4slam
A Swiss knife for SLAM.



## Developing

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
python3 tools/linter/bin/init_linter_git_hooks
```


Add the following in your "~/.bashrc" (or other file matching for your shell). Or run this in your working terminal manually before ```git commit```.
```
. tools/linter/setup_linter.sh
```
