# Contributing to Multi-Horizon Urban Growth Prediction

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository**
2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Multi-Horizon-Urban-Growth-Prediction.git
   cd Multi-Horizon-Urban-Growth-Prediction
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-number
```

Branch naming conventions:
- `feature/` - New features
- `bugfix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions/updates

### 2. Make Changes

Follow these guidelines:
- Write clear, concise commit messages
- Add docstrings to all functions and classes
- Follow PEP 8 style guide
- Add tests for new functionality

### 3. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_model.py -v

# Check coverage
pytest --cov=src tests/
```

### 4. Submit Pull Request

1. Push your branch:
   ```bash
   git push origin feature/your-feature-name
   ```
2. Open a pull request on GitHub
3. Describe your changes clearly
4. Link related issues

## Code Style

### Python Style Guide

Follow PEP 8 with these specifics:
- Line length: 100 characters
- Use double quotes for strings
- Type hints for function arguments and returns

Example:
```python
def process_tile(tile_data: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Process a single tile.
    
    Args:
        tile_data: Input tile array
        normalize: Whether to normalize values
        
    Returns:
        Processed tile array
    """
    if normalize:
        tile_data = (tile_data - tile_data.mean()) / tile_data.std()
    return tile_data
```

### Docstring Format

Use Google-style docstrings:
```python
def function_name(arg1, arg2):
    """
    Short description.
    
    Longer description if needed.
    
    Args:
        arg1 (type): Description
        arg2 (type): Description
        
    Returns:
        type: Description
        
    Raises:
        ValueError: When condition occurs
    """
```

## Testing Guidelines

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names

Example:
```python
def test_convlstm_handles_variable_sequence_length():
    """Test that ConvLSTM processes sequences of different lengths."""
    model = create_model()
    
    for seq_len in [2, 3, 5]:
        x = torch.randn(1, seq_len, 2, 128, 128)
        predictions, _ = model(x)
        assert predictions.shape[1] == seq_len
```

### Test Coverage

Aim for >80% code coverage:
```bash
pytest --cov=src --cov-report=html tests/
open htmlcov/index.html
```

## Documentation

### Adding Documentation

- Update relevant `.md` files in `docs/`
- Add docstrings to new code
- Update `README.md` if adding major features

### Building Documentation

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme

# Build docs
cd docs
make html
open _build/html/index.html
```

## Issues and Bugs

### Reporting Bugs

Use the issue template and include:
1. **Description**: Clear description of the bug
2. **Steps to Reproduce**: Minimal code example
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment**: OS, Python version, PyTorch version

Example:
```markdown
**Bug**: Model fails on tiles with all-zero values

**Reproduce**:
```python
model = create_model()
x = torch.zeros(1, 3, 2, 128, 128)
predictions, _ = model(x)  # Raises RuntimeError
```

**Expected**: Should handle zero-valued inputs gracefully
**Actual**: RuntimeError: division by zero
**Environment**: macOS 13.0, Python 3.10, PyTorch 2.0
```

### Feature Requests

Open an issue with:
- Clear description of the feature
- Use case and motivation
- Proposed implementation (if applicable)

## Pull Request Process

1. **Update Documentation**: Add/update docstrings and READMEs
2. **Add Tests**: Ensure >80% coverage for new code
3. **Run Tests**: All tests must pass
4. **Update CHANGELOG**: Add entry for your changes
5. **Review Process**: Maintainers will review within 7 days

### PR Checklist

Before submitting:
- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] No merge conflicts
- [ ] Descriptive commit messages

## Communication

- **GitHub Issues**: Bug reports and feature requests
- **Pull Requests**: Code contributions
- **Discussions**: General questions and ideas
- **Email**: rohan.bali@example.com for private inquiries

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be acknowledged in:
- `CONTRIBUTORS.md` file
- Release notes
- Paper acknowledgments (for significant contributions)

## Questions?

Don't hesitate to ask! Open an issue with the `question` label or reach out directly.

Thank you for contributing.
