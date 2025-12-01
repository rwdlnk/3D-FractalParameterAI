# Contributing to 3D-FractalParameterAI

Thank you for your interest in contributing to 3D-FractalParameterAI! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Create a new branch for your feature or fix
4. Make your changes
5. Submit a pull request

## Development Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/3D-FractalParameterAI.git
cd 3D-FractalParameterAI

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

## Code Style

- Follow PEP 8 guidelines for Python code
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and modular

## Testing

- Write tests for new functionality
- Ensure all existing tests pass before submitting a PR
- Run tests with: `pytest tests/`

## Pull Request Process

1. Update the README.md if your changes affect usage
2. Update the CHANGELOG.md with a description of your changes
3. Ensure your code passes all tests
4. Request review from maintainers

## Types of Contributions

### Bug Reports

When reporting bugs, please include:
- Python version and OS
- Steps to reproduce the issue
- Expected vs actual behavior
- Relevant error messages or logs

### Feature Requests

For feature requests, please describe:
- The problem you're trying to solve
- Your proposed solution
- Any alternatives you've considered

### Code Contributions

We welcome contributions in the following areas:
- 3D surface extraction algorithms
- Box-counting implementations
- AI/ML parameter optimization
- Documentation improvements
- Test coverage expansion
- Performance optimizations

## Questions?

Feel free to open an issue for any questions about contributing.
