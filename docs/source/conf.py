# Configuration file for the Sphinx documentation builder.
# 
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os,sys
sys.path.insert(0,os.path.abspath('..'))
sys.path.insert(1,'./src')
sys.path.insert(2,'../src')
sys.path.insert(3,'../../src')

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SMRL'
copyright = '2026, John E. Bond'
author = 'John E. Bond'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = extensions = [
    'sphinx.ext.autodoc',
    'sphinx_rtd_theme',
    'sphinx_book_theme',
    'sphinxcontrib.video',
]

templates_path = ['_templates']
exclude_patterns = []

autodoc_member_order = 'bysource'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_css_files = [
    'twocol.css',
]

html_theme_options = {
    "repository_url": "https://github.com/johnemmanuelbond/gym-coll",
    "use_repository_button": True,
    "show_toc_level": 2,
    # Other theme options can be set here
}
