Remove (un)wanted text by detecting it with DocTR and in-painting the masked region with LaMa (default) or Deep-Image-Prior.

untext = find words ➜ generate pixel mask ➜ in-paint

* Text detection: [DocTR](https://github.com/mindee/doctr) DB-Net (lightweight, CPU friendly)
* In-painting back-end (pick one at run-time)
  * **LaMa** via the `simple-lama-inpainting` wheel  ← default, fast, good quality
  * Deep-Image-Prior (`--method dip`) for pure-PyTorch fallback
  * Edge-fill toy (`--method edge_fill`) for instant mock-ups

---

## Installation

```bash
git clone https://github.com/jurph/untext.git
cd untext
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# (optional) install as editable module
pip install -e .
```

The requirements pull:

* PyTorch ≥ 2.0
* DocTR 0.6.1 (text detection)
* simple-lama-inpainting 0.1.2 (LaMa backend)

All wheels are available on PyPI—no heavy git check-outs needed.

---

## Command-line usage

```bash
untext -i watermarked.jpg -o clean.jpg            # uses LaMa
untext -i img.jpg -o out.jpg --method dip         # use Deep-Image-Prior
untext -h                                         # full options
```

Internally the CLI will:

1. detect words with DocTR and convert each bounding box into a binary mask
2. dilate & feather the mask
3. in-paint with the selected backend

---

## Python API (simplest path)

```python
from untext.image_patcher import ImagePatcher

patcher = ImagePatcher()
result  = patcher.patch_image('photo.jpg', 'mask.png', method='lama')
```

`method` can be `"lama"`, `"dip"`, or `"edge_fill"`.

---

## Tests

```bash
pytest -v
```

Unit tests cover detection, mask generation, and in-painting with LaMa.

---

© 2025  Jurph – MIT license



