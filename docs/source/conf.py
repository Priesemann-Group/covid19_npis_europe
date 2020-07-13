# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import re

sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------

project = "COVID-19 NPIS"
copyright = "2020, Priesemann group"
author = "Priesemann group"

verstr = "unknown"
try:
    verstrline = open("../../covid19_npis/_version.py", "rt").read()
except EnvironmentError:
    pass
else:
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    mo = re.search(VSRE, verstrline, re.M)
    if mo:
        verstr = mo.group(1)
    else:
        raise RuntimeError("unable to find version in covid19_npis/_version.py")
print("sphinx found version: {}".format(verstr))
# The short X.Y version
version = verstr
# The full version, including alpha/beta/rc tags
release = verstr

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
import sphinx_rtd_theme

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    #'sphinx.ext.mathjax',
    "sphinx.ext.napoleon",
    # 'numpydoc',
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.imgmath",
    "recommonmark"
    # 'sphinx_autorun'
]

# Mock imports
autodoc_mock_imports = [
    "numpy",
    "matplotlib",
    "matplotlib.pyplot",
    "mpl_toolkits",
    "scipy",
    "pymc3",
    "theano",
    "pandas",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "collapse_navigation": False,
    "display_version": False,
    "navigation_depth": 4,
    "sticky_navigation": True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# -- Extension configuration -------------------------------------------------

intersphinx_mapping = {
    "python": ("http://docs.python.org/3", None),
    "numpy": ("http://docs.scipy.org/doc/numpy", None),
    "scipy": ("http://docs.scipy.org/doc/scipy/reference", None),
    "matplotlib": ("http://matplotlib.sourceforge.net", None),
    "pymc3": ("https://docs.pymc.io", None),
    "theano": ("http://deeplearning.net/software/theano/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

imgmath_image_format = "svg"
imgmath_font_size = 14

autodoc_member_order = "bysource"

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True