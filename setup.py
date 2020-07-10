from setuptools import setup, find_namespace_packages
import re

# read the contents of your README file
from os import path

with open("README.md") as f:
    long_description = f.read()

verstr = "unknown"
try:
    verstrline = open("covid19_npis/_version.py", "rt").read()
except EnvironmentError:
    pass
else:
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    mo = re.search(VSRE, verstrline, re.M)
    if mo:
        verstr = mo.group(1)
    else:
        raise RuntimeError("unable to find version in covid19_inference/_version.py")

setup(
    name="covid19_npis",
    author="Priesemann group",
    author_email="jonas.dehning@ds.mpg.de",
    packages=find_namespace_packages(),
    url="https://github.com/Priesemann-Group/covid19_npis_europe",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.6.0",
    version=verstr,
    install_requires=["pymc4"],
)
