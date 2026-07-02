"""Tests for the html_to_text helper in processor.extractor."""

from idi_corporate_structure.extractor import html_to_text


class TestHtmlToText:
    """Tests for html_to_text()."""

    def test_strips_tags_preserves_cell_text(self):
        raw = "<table><tr><td>senseFly Inc., a Delaware Corporation</td></tr></table>"
        text = html_to_text(raw)

        assert "senseFly Inc., a Delaware Corporation" in text

    def test_decodes_html_entities(self):
        raw = "<p>&#8220;American Eagle&#8221; Holdings &amp; Co.</p>"
        text = html_to_text(raw)

        assert "&#8220;" not in text
        assert "&amp;" not in text
        assert "American Eagle" in text
        assert "Holdings & Co." in text

    def test_removes_script_and_style(self):
        raw = (
            "<html><head><style>p { color: red; }</style></head>"
            "<body><script>alert(1)</script><p>Foo Inc.</p></body></html>"
        )
        text = html_to_text(raw)

        assert "Foo Inc." in text
        assert "color" not in text
        assert "alert" not in text

    def test_table_rows_separated_by_newlines(self):
        raw = (
            "<table>"
            "<tr><td>Foo, Inc.</td><td>Delaware</td></tr>"
            "<tr><td>Bar, LLC</td><td>Nevada</td></tr>"
            "</table>"
        )
        text = html_to_text(raw)
        lines = [line for line in text.split("\n") if line]

        assert any("Foo, Inc." in line for line in lines)
        assert any("Bar, LLC" in line for line in lines)
        foo_idx = next(i for i, line in enumerate(lines) if "Foo, Inc." in line)
        bar_idx = next(i for i, line in enumerate(lines) if "Bar, LLC" in line)
        assert foo_idx != bar_idx

    def test_nested_inline_tags_do_not_break_names(self):
        raw = "<td><b>Foo</b>, Inc.</td>"
        text = html_to_text(raw)

        assert "Foo, Inc." in text

    def test_plain_text_input_passes_through(self):
        text = html_to_text("Apple Operations LLC (Delaware)")

        assert "Apple Operations LLC (Delaware)" in text

    def test_cells_on_same_row_stay_on_one_line(self):
        raw = "<tr><td>senseFly Inc., a Delaware Corporation</td><td>USA</td></tr>"
        text = html_to_text(raw)

        matching = [line for line in text.split("\n") if "senseFly" in line]
        assert len(matching) == 1
        assert "USA" in matching[0]

    def test_html_comments_removed(self):
        raw = "<!-- hidden --><p>Visible Corp</p>"
        text = html_to_text(raw)

        assert "Visible Corp" in text
        assert "hidden" not in text

    def test_cell_wrapped_in_div_does_not_fragment_row(self):
        """A cell wrapped in its own <div> must not fragment the row it's in.

        SEC filing software often wraps each cell's text in a <div> purely
        for styling. Treating that as a paragraph break used to turn one
        logical row into one paragraph per cell (name, %, counts), which
        corrupted the chunker's row-count heuristics.
        """
        raw = (
            "<table><tr>"
            "<td><div>Freedom Finance JSC, Kazakhstan</div></td>"
            "<td><div>100%</div></td>"
            "<td><div>-</div></td>"
            "<td><div>3</div></td>"
            "</tr></table>"
        )
        text = html_to_text(raw)

        assert text.split("\n\n") == ["Freedom Finance JSC, Kazakhstan 100% - 3"]

    def test_nested_table_inside_cell_still_breaks_rows(self):
        """A genuinely nested table inside a cell should still get row breaks."""
        raw = (
            "<table><tr><td>"
            "<table><tr><td>Inner A</td></tr><tr><td>Inner B</td></tr></table>"
            "</td></tr></table>"
        )
        text = html_to_text(raw)
        lines = [line for line in text.split("\n") if line.strip()]

        assert "Inner A" in lines
        assert "Inner B" in lines
        assert lines.index("Inner A") != lines.index("Inner B")
