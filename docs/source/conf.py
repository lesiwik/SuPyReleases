# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
from datetime import date

import numpydoc.docscrape as np_docscrape

sys.path.append(os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "supy"
copyright = f"{date.today().year}, Leszek Siwik"
author = """
Leszek Siwik <leszek.siwik@gmail.com>
Marcin Łoś <marcin.los.91@gmail.com>
"""


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


nitpicky = True

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    # links to other project's documentation
    "sphinx.ext.intersphinx",
    # check documentation coverage
    "sphinx.ext.coverage",
    # numpy style docstring parser
    "numpydoc",
    # links to project code
    "sphinx.ext.viewcode",
    # copy button in the code cells
    "sphinx_copybutton",
    # Better automatic documentation for more exotic constructions.
    # These extensions do not work well with autosummary.
    # "sphinx_toolbox.more_autodoc.autonamedtuple",
    # "sphinx_toolbox.more_autodoc.autoprotocol",
    # "enum_tools.autoenum",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/devdocs", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

add_function_parentheses = False

# Do not show type hints in function signature
autodoc_typehints = "none"

autosummary_generate = True

numpydoc_class_members_toctree = False

np_docscrape.ClassDoc.extra_public_methods = [  # should match class.rst
    "__call__",
    "__getitem__",
]

# The reST default role (used for this markup: `text`) to use for all documents.
default_role = "autolink"

# Do not copy prompts and outputs from code fragments
copybutton_exclude = ".linenos, .gp, .go"


templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "furo"
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = f"{project} documentation"

html_theme_options = {
    "navigation_with_keys": False,
}
