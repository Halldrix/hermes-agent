"""Regression: a catastrophic command must still be hardline-blocked when a shell
redirection precedes it (e.g. ``> /dev/null rm -rf /``).
"""
from tools.approval_detection import detect_hardline_command


def test_leading_redirect_does_not_bypass_hardline_rm():
    for cmd in (
        "> /dev/null rm -rf /",
        "2>&1 rm -rf /",
        "< /dev/null rm -rf /",
        ">file rm -rf /",
        "rm -rf / > /dev/null",
    ):
        is_hardline, desc = detect_hardline_command(cmd)
        assert is_hardline, (cmd, desc)


def test_leading_redirect_benign_not_hardline():
    is_hardline, _ = detect_hardline_command("> /dev/null echo hi")
    assert not is_hardline