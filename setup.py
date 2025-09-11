from setuptools import find_packages, setup

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="poseadapt",
    version="0.1.0-alpha01",
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
)
