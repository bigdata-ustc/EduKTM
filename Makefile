VERSION=`ls dist/*.tar.gz | sed "s/dist\/EduKTM-\(.*\)\.tar\.gz/\1/g"`

ifdef ENVPIP
    PIP = $(ENVPIP)
else
    PIP = pip3
endif

ifdef ENVPYTHON
    PYTHON = $(ENVPYTHON)
else
    PYTHON = python3
endif

ifdef ENVPYTEST
    PYTEST = $(ENVPYTEST)
else
    PYTEST = pytest
endif

help:

	@echo "install              install EduKTM"
	@echo "test                 run test"
	@echo "release              publish to PyPI and release in github"
	@echo "release_test         publish to TestPyPI"
	@echo "clean                remove all build, test, coverage and Python artifacts"
	@echo "clean-build          remove build artifacts"
	@echo "clean-pyc            remove Python file artifacts"
	@echo "clean-test           remove test and coverage artifacts"

.PHONY: install, test, build, release, release_test, version, .test, .build, clean

install:
	@echo "install EduKTM"
	$(PIP) install -e . --user

test:
	@echo "run test"
	$(PYTEST)

build: test, clean
	$(PYTHON) setup.py bdist_wheel sdist

.test:
	$(PYTEST) > /dev/null

.build: clean
	$(PYTHON) setup.py bdist_wheel sdist > /dev/null

version: .build
	@echo $(VERSION)

release: test, build
	@echo "publish to pypi and release in github"
	@echo "version $(VERSION)"

	-@twine upload dist/* && git tag "v$(VERSION)"
	git push && git push --tags

release_test: test, build
	@echo "publish to test pypi"
	@echo "version $(VERSION)"

	-@twine upload --repository test dist/*

clean: clean-build clean-pyc clean-test

clean-build:
	rm -rf build/*
	rm -rf dist/*
	rm -rf .eggs/*
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +

clean-test:
	rm -f .coverage