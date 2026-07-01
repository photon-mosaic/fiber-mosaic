# fiber-mosaic

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
![Python](https://img.shields.io/badge/python->=3.12-blue?logo=python)
![Interrogate](https://img.shields.io/badge/interrogate-100.0%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)

_Description pending._

## Installation

To use the software, in the root directory, run:
```bash
pip install -e .
```

To develop the code, run:
```bash
pip install -e .[dev]
```

## Examples

A runnable walkthrough lives in [`examples/quickstart.ipynb`](examples/quickstart.ipynb): building per-color recordings, reading fluorescence with the fiber-native API, attaching per-fiber timestamps with `set_times`, bundling colors into a `FiberPhotometryRecordingGroup`, and discovering streams with `get_streams`.

The notebook is committed **without cell outputs** to keep diffs small. To populate the outputs, install Jupyter and matplotlib, then run it:

```bash
pip install -e .[dev]
pip install jupyter matplotlib

# run interactively in the browser:
jupyter notebook examples/quickstart.ipynb

# or execute top-to-bottom and write the outputs back in place:
jupyter nbconvert --to notebook --execute --inplace examples/quickstart.ipynb
```

## Contributing

### Linters and testing

There are several libraries used to run linters, check documentation, and run tests.

- Please test your changes using **pytest**, which will run the tests and log a coverage report:

```bash
pytest
```

- Use **interrogate** to check that code is well documented:

```bash
interrogate .
```

- Use **ruff** to lint code and sort imports:

```bash
ruff check .
```

- Use **black** to automatically format the code into PEP standards:

```bash
black .
```

### Pull requests

For internal members, please create a branch. For external members, please fork the repository and open a pull request from the fork. We'll primarily use [Angular](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit) style for commit messages. Roughly, they should follow the pattern:

```text
<type>(<scope>): <short summary>
```

where scope (optional) describes the packages affected by the code changes and type (mandatory) is one of:

- **build**: Changes that affect build tools or external dependencies (example scopes: pyproject.toml, setup.py)
- **ci**: Changes to our CI configuration files and scripts (examples: .github/workflows/ci.yml)
- **docs**: Documentation only changes
- **feat**: A new feature
- **fix**: A bug fix
- **perf**: A code change that improves performance
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **test**: Adding missing tests or correcting existing tests

### Semantic Release

The table below, from [semantic release](https://github.com/semantic-release/semantic-release), shows which commit message gets you which release type when `semantic-release` runs (using the default configuration):

| Commit message                                                                                                                                                                                   | Release type                                                                                                    |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| `fix(pencil): stop graphite breaking when too much pressure applied`                                                                                                                             | ~~Patch~~ Fix Release, Default release                                                                          |
| `feat(pencil): add 'graphiteWidth' option`                                                                                                                                                       | ~~Minor~~ Feature Release                                                                                       |
| `perf(pencil): remove graphiteWidth option`<br><br>`BREAKING CHANGE: The graphiteWidth option has been removed.`<br>`The default graphite width of 10mm is always used for performance reasons.` | ~~Major~~ Breaking Release <br /> (Note that the `BREAKING CHANGE: ` token must be in the footer of the commit) |

### Documentation

To generate the rst files source files for documentation, run:

```bash
sphinx-apidoc -o doc_template/source/ src
```

Then to create the documentation HTML files, run:

```bash
sphinx-build -b html doc_template/source/ doc_template/build/html
```

More information on sphinx installation can be found [here](https://www.sphinx-doc.org/en/master/usage/installation.html).
