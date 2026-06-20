# Licensed under a 3-clause BSD style license - see LICENSE.rst

import datetime
import sys
from importlib.metadata import metadata

try:
    from sphinx_astropy.conf.v3 import *  # noqa: F401,F403
except ImportError:
    print("ERROR: the documentation requires the sphinx-astropy[confv3] package to be installed")
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

html_title = f"{project} v{release}"

html_static_path = ["_static"]

# The trimmed grey MMT logo reads on both the light and dark theme, so use the
# one image for both modes.
html_logo = "_static/mmt_logo.png"

# MMT favicon (from mmto.org) in place of the theme's astropy default.
html_favicon = "_static/mmt_favicon.ico"

# astropy-unified is a pydata-based theme; wire up the GitHub link and the
# "Edit on GitHub" button. v3 deletes the inherited theme options.
html_theme_options = {
    "github_url": f"https://github.com/{GITHUB_PROJECT}",
    "use_edit_page_button": True,
}
html_context = {
    "github_user": GITHUB_PROJECT.split("/")[0],
    "github_repo": GITHUB_PROJECT.split("/")[1],
    "github_version": "main",
    "doc_path": "docs",
}
