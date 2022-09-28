from setuptools import setup, find_packages

test_deps = [
    'pytest>=4',
    'pytest-cov>=2.6.0',
    'pytest-flake8',
    'flake8<5.0.0'
]

setup(
    name='EduKTM',
    version='0.0.10',
    extras_require={
        'test': test_deps,
    },
    packages=find_packages(),
    install_requires=[
        "torch",
        "tqdm",
        "numpy>=1.16.5",
        "scikit-learn",
        "pandas",
        "networkx"
    ],  # And any other dependencies for needs
    entry_points={
    },
)
