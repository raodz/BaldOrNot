# Contributing to [Project-Name]

## Table of Contents

1. [Python Version](#python-version)
2. [Code Style](#code-style)
3. [Installing pre-commit hooks](#installing-pre-commit-hooks)
4. [Documentation](#documentation)
5. [Directory Structure](#directory-structure)
6. [Testing](#testing)
7. [Version Control](#version-control)
8. [Code Reviews](#code-reviews)
9. [Additional Resources](#additional-resources)

## Python Version

This project requires Python version 3.12. Ensure you have the correct version installed by running:

```sh
python --version
```

Or if you have multiple versions of Python installed, you can specify the version explicitly:

```sh
python3.12 --version
```

Make sure to use a virtual environment to manage dependencies specific to this project.

## Code Style

Please follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines for writing Python code.

- **Indentation**: Use 4 spaces per indentation level.
- **Line Length**: Limit all lines to a maximum of 79 characters.
- **Naming Conventions**:
  - Functions and variables: `snake_case` (e.g., `train_model`, `learning_rate`)
  - Classes: `CamelCase` (e.g., `CNNClassifier`)
- **Imports**: Group imports into three categories in the following order:
  1. Standard library imports.
  2. Related third-party imports.
  3. Local application/library-specific imports.
- **String Quotes**: Use single quotes for strings ('example') except when the string contains a single quote that needs escaping.
- **Code Formatting**: Use `black` for automatic code formatting to ensure consistency and adherence to PEP 8 guidelines.

## Installing Pre-commit Hooks

To ensure code quality and consistency across our project, we use pre-commit hooks that automatically check and format code before commits. Follow the instructions below to set up the pre-commit hooks:

1. **Install Pre-commit**:
   Ensure you have `pre-commit` installed. You can install it via pip:
   ```bash
   pip install pre-commit
   ```

2. **Download the Pre-commit Configuration File**:
   Download the `.pre-commit-config.yaml` file from the root directory of the project repository.

3. **Place the Configuration File**:
   Ensure the downloaded `.pre-commit-config.yaml` file is placed in the root directory of your local project repository.

4. **Install the Pre-commit Hooks**:
   Run the following command in the root directory of the project to install the hooks:
   ```bash
   pre-commit install
   ```

5. **Run Pre-commit Hooks Manually (Optional)**:
   To check all existing files in the repository, you can run the pre-commit hooks manually:
   ```bash
   pre-commit run --all-files
   ```

### Description of Hooks

- **Black**: Automatically formats Python code to adhere to PEP 8 guidelines.
- **Flake8**: Checks for style errors and potential bugs in Python code.
- **Mypy**: Performs static type checking based on type annotations.

By following these steps, you will ensure that your code is automatically formatted and checked for common errors before being committed, helping maintain high code quality throughout the project.


## Documentation

- **Docstrings**: Document all public modules, functions, classes, and methods.
- **Format**: Use Google style docstrings.

### Example:

```python
def train_model(data, labels, epochs):
    """
    Trains the CNN model on the provided data and labels.

    Args:
        data (np.ndarray): Training data.
        labels (np.ndarray): Training labels.
        epochs (int): Number of epochs to train.

    Returns:
        model: Trained CNN model.
    """
    pass
```

## Directory Structure

```
Project-Name/
├── scripts/
│   ├── main.py
│   ├── preprocess_data.py
├── src/
│   ├── data.py
│   ├── model.py
├── tests/
│   ├── test_load_data.py
│   ├── test_preprocess_data.py
├── notebooks/
│   └── exploratory_data_analysis.ipynb
├── .github/
│   └── pull_request_template.md
├── README.md
├── CONTRIBUTING.md
├── requirements.txt
├── pre-commit-conflig.yaml
├── .gitignore
└── setup.py
```

- **scripts/**: Contains high-level scripts to run various parts of the project.
- **src/**: Contains the core modules, including data handling and model definition.
- **tests/**: Contains test modules for ensuring code correctness.
- **notebooks/**: Contains Jupyter notebooks for data exploration and analysis.
- **README.md**: Provides an overview of the project, installation instructions, and usage examples.
- **CONTRIBUTING.md**: Contains guidelines for contributing to the project.
- **requirements.txt**: Lists the project dependencies.
- **.gitignore**: Specifies files and directories to be ignored by Git.
- **setup.py**: Script for installing the package and its dependencies.

## Testing

- **Framework**: Use `pytest` for running tests.
- **Coverage**: Ensure your code coverage is at least 90%.
- **Writing Tests**: Write unit tests for your code. Place them in the `tests/` directory.

### Example:

```python
def test_train_model():
    data = np.array([...])
    labels = np.array([...])
    model = train_model(data, labels, epochs=10)
    assert model is not None
```

## Version Control

- **Commit Messages**: Use meaningful commit messages.
  - Follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages.
  - Example: `feat: add data augmentation`, `fix: correct data loader bug`.
- **Branching**: Use feature branches for new features and bug fixes.
  - Example: `feature/add-data-augmentation`, `fix/issue-123`.

## Code Reviews

- **Pull Requests**: All changes must be submitted via pull requests.
- **Review Process**: At least one team member must review and approve your pull request before it can be merged.
- **Testing**: Ensure that your code passes all tests before submitting for review.

## Additional Resources

- [PEP 8](https://www.python.org/dev/peps/pep-0008/): Python style guide.
- [Google Python Style Guide](http://google.github.io/styleguide/pyguide.html): Google's style guide for Python.
- [Conventional Commits](https://www.conventionalcommits.org/): A specification for adding human and machine-readable meaning to commit messages.
