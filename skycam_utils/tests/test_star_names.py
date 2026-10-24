# Licensed under a 3-clause BSD style license - see LICENSE.rst
import csv
from importlib.resources import files


def _rows():
    path = files("skycam_utils") / "data" / "star_names.csv"
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_one_row_per_star():
    """Alternate names live in Aliases, not in duplicate rows (issue #16)."""
    rows = _rows()
    hd = [r["HD"] for r in rows]
    assert len(set(hd)) == len(hd)
    names = [r["Star Name"] for r in rows]
    assert len(set(names)) == len(names)


def test_aliases_do_not_collide_with_names():
    rows = _rows()
    names = {r["Star Name"] for r in rows}
    aliases = [a for r in rows for a in r["Aliases"].split(";") if a]
    assert len(set(aliases)) == len(aliases)
    assert not names & set(aliases)
    rigel = next(r for r in rows if r["Star Name"] == "Rigel")
    assert rigel["Aliases"] == "Algebar"


def test_misattributed_names_are_gone():
    """Names that belong to a neighbouring star, not the one in the row."""
    rows = _rows()
    every = {r["Star Name"] for r in rows} | {
        a for r in rows for a in r["Aliases"].split(";") if a}
    wrong = {"Alula Australis", "Lesath", "Nunki", "Zaniah", "Kuma", "Keid",
             "Sualocin"}
    assert not every & wrong
    assert not any("Bodtis" in r["Identity"] for r in rows)
