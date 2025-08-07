from setuptools import setup, find_packages

setup(
    name="hj_avoid",
    version="0.1.0",
    author="Nishanth Rao",
    description="A safety filter for gym environments",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.26.2",
        "pygame>=2.1.0",
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",],
    python_requires=">=3.9",
)