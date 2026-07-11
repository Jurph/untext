# Editable Text Sources

These flat files drive the procedural text generator.

- `url_prefixes.txt`, `url_nouns.txt`, and `url_tlds.txt` define the URL-like
  mode.
- `copyright_first_names.txt` and `copyright_last_names.txt` define the
  `© Firstname Lastname` mode.

The generator picks the copyright-name mode for roughly 25% of samples and the
URL-like mode for the remainder.

Rules:

- one token per line
- blank lines and `#` comments are ignored
- keep the files UTF-8 encoded
- keep the lists alphabetical if you want them easy to edit by hand

The current lists are intentionally biased toward common English letters while
still covering A-Z, including `Xenia` and `Xoroboros`.
