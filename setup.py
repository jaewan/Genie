#!/usr/bin/env python3
"""Setup script for Genie - GPU Disaggregation Framework for PyTorch."""

from setuptools import setup, find_packages
import os

# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read README for long description
def read_readme():
    """Read README.md for long description."""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, encoding='utf-8') as f:
            return f.read()
    return ""

setup(
    name="genie-gpu",
    version="0.1.0",
    description="GPU Disaggregation Framework for PyTorch with Zero-Copy Transport",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Genie Team",
    author_email="",
    url="https://github.com/yourusername/genie",
    license="MIT",
    
    # Package discovery
    packages=find_packages(exclude=['tests', 'tests.*', 'benchmarks', 'example', 'docs', 'scripts']),
    
    # Include pre-built C++ extensions
    package_data={
        'genie': [
            '*.so',
            '_C*.so',
            '_runtime*.so',
        ],
    },
    
    # Dependencies
    install_requires=read_requirements('requirements.txt'),
    
    extras_require={
        'dev': read_requirements('requirements-dev.txt') if os.path.exists('requirements-dev.txt') else [],
    },
    
    # Python version requirement
    python_requires='>=3.8',
    
    # Classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: C++',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: System :: Distributed Computing',
    ],
    
    # Entry points (optional - for command-line tools)
    entry_points={
        'console_scripts': [
            # Add command-line tools here if needed
            # 'genie-server=genie.server:main',
        ],
    },
)

