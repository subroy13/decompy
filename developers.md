## Developer Useful Commands

Here are few useful commands to build and deploy the package.

```
python3 -m pip install twine build
```

Then to build the project use, 
* `python3 -m build --sdist` to create a minimal source distribution.
* `python3 -m build --wheel` to build wheel binary files that anyone can easily install.

Run a check using `twine` using `twine check dist/*`.

Finally, upload to `pypi` using `twine upload dist/*`

To generate documentation, use 
1. Use the `mintlify AI doc generator` to generate docstring from code using explainer AI. Use `numpy` type documentation.
2. `sphinx-apidoc -o docs/ -d 3 ./src` to generate the rst files.
3. Run the makefile `make.bat` with `html` argument, i.e., `make clean && make html` inside `./src/docs` folder.
4. The generated html files will be present inside `_build` folder.


## Reference 

1. https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/
