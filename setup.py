from setuptools import find_packages, setup

def read_requirements(filename):
	with open(filename) as file:
		return [r.strip() for r in file.readlines()]

setup(
	name="qmlmodels",
	packages=find_packages(),
	version="0.0.1",
	description="A package containing templates and parts for Quantum Machine Learning models.",
	license="MIT",
	install_requires=read_requirements("requirements.txt")
)
