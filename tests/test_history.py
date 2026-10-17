"""U3: the Character page's edit history (core.history, no Qt)."""
from eu4unitgen.core.history import History, Snapshot


def test_undo_redo_and_saved_mark():
    a, b, c = Snapshot({"n": 1}), Snapshot({"n": 2}), Snapshot({"n": 3})
    h = History(a)
    assert not h.dirty and not h.can_undo
    assert h.push(b) and not h.push(Snapshot({"n": 2}))  # equal to the current step: not a step
    h.push(c)
    assert h.dirty and h.undo() == b and h.undo() == a and h.undo() is None
    assert not h.dirty and h.redo() == b
    h.push(Snapshot({"n": 9}))  # a new edit drops the redo steps
    assert not h.can_redo
    h.mark_saved()
    assert not h.dirty and h.undo() == b and h.dirty


def test_images_compare_by_identity_and_the_limit():
    img = object()
    assert Snapshot({}, {"m": img}) == Snapshot({}, {"m": img})
    assert Snapshot({}, {"m": img}) != Snapshot({}, {"m": object()})
    assert Snapshot({}, {"m": img}) != Snapshot({})
    h = History(Snapshot({"n": 0}), limit=3)
    for n in range(1, 6):
        h.push(Snapshot({"n": n}))
    assert h.undo().doc == {"n": 4} and h.undo().doc == {"n": 3} and h.undo() is None
