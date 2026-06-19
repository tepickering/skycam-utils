# Licensed under a 3-clause BSD style license - see LICENSE.rst

import datetime
import sys
from importlib.metadata import metadata

try:
    from sphinx_astropy.conf.v1 import *  # noqa: F401,F403
except ImportError:
    print("ERROR: the documentation requires the sphinx-astropy package to be installed")
    sys.exit(1)

_meta = metadata("skycam_utils")

project = _meta["Name"]
author = _meta["Author-email"].split(" <")[0] if _meta["Author-email"] else _meta["Author"]
copyright = f"{datetime.datetime.now().year}, {author}"
release = _meta["Version"]
version = release.split("-", 1)[0]

GITHUB_PROJECT = "tepickering/skycam-utils"

highlight_language = "python3"
exclude_patterns.append("_templates")  # noqa: F405
rst_epilog += "\n"  # noqa: F405

html_theme_options = {
    "logotext1": "skycam_utils",
    "logotext2": "",
    "logotext3": ":docs",
}
html_title = f"{project} v{release}"
htmlhelp_basename = project + "doc"

latex_documents = [
    ("index", project + ".tex", project + " Documentation", author, "manual"),
]
man_pages = [
    ("index", project.lower(), project + " Documentation", [author], 1),
]

extensions += ["sphinx_astropy.ext.edit_on_github"]  # noqa: F405
edit_on_github_project = GITHUB_PROJECT
edit_on_github_branch = "main"
edit_on_github_source_root = ""
edit_on_github_doc_root = "docs"

github_issues_url = f"https://github.com/{GITHUB_PROJECT}/issues/"
