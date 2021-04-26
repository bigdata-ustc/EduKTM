from setuptools import setup

test_deps = [
    'pytest>=4',
    'pytest-cov>=2.6.0',
    'pytest-flake8',
]

setup(
    name='EduKTM',
    version='0.0.2',
    extras_require={
        'test': test_deps,
    },
    install_requires=[
        "torch",
        "tqdm",
        "numpy>=1.16.5",
        "scikit-learn",
        "pandas",
    ],  # And any other dependencies for needs
    entry_points={
    },
)
