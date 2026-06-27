# Detector Model Bibliography

This project uses three text detectors in consensus. This note records where
each model comes from, how `untextre` loads it, and where to fetch model files
manually if an automatic download breaks.

## EAST

- **Model used here:** EAST TensorFlow graph loaded through OpenCV DNN from
  `frozen_east_text_detection.pb`.
- **Invented by:** Xinyu Zhou, Cong Yao, He Wen, Yuzhi Wang, Shuchang Zhou,
  Weiran He, and Jiajun Liang at Megvii Technology Inc.
- **Publication:** "EAST: An Efficient and Accurate Scene Text Detector,"
  CVPR 2017. The paper describes EAST as a detector that directly predicts
  words or text lines in full images and avoids intermediate candidate
  aggregation and word partitioning steps.
- **How we ingest it:** `untextre.detector._load_east_model()` checks
  `~/.untextre/models/frozen_east_text_detection.pb`. If the file is missing,
  `_download_east_model()` downloads it, rejects tiny/truncated responses, and
  only then moves it into the cache. OpenCV loads the cached file with
  `cv2.dnn.readNet()`.
- **Automatic download URL used by this repo:**
  `https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb`
- **Manual fallback:** Download `frozen_east_text_detection.pb` from the URL
  above and place it at `~/.untextre/models/frozen_east_text_detection.pb`.
  OpenCV's sample references the original EAST project at
  `https://github.com/argman/EAST` and a `frozen_east_text_detection.tar.gz`
  archive at
  `https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1`.
- **Primary references:**
  - Paper: https://arxiv.org/abs/1704.03155
  - CVF PDF: https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhou_EAST_An_Efficient_CVPR_2017_paper.pdf
  - OpenCV sample note: https://github.com/opencv/opencv/blob/4.x/samples/dnn/text_detection.py

## DocTR / DBNet

- **Model used here:** `doctr.models.detection.db_resnet50(pretrained=True)`.
  The detector architecture is DBNet with a ResNet-50 backbone, loaded through
  the python-doctr package.
- **Invented by:** The DB module was proposed by Minghui Liao, Zhaoyi Wan,
  Cong Yao, Kai Chen, and Xiang Bai. In our dependency stack, the packaged
  DocTR implementation and pretrained checkpoints are maintained by Mindee.
- **Publication:** "Real-time Scene Text Detection with Differentiable
  Binarization," AAAI 2020. The paper describes Differentiable Binarization as
  a module that integrates binarization into the segmentation network so
  thresholds can be learned during training.
- **How we ingest it:** `TextDetector.__init__()` calls
  `detection.db_resnet50(pretrained=True)`. DocTR resolves and downloads the
  pretrained checkpoint into its own cache, then `DetectionPredictor` wraps it
  with this project's fixed preprocessor settings.
- **Canonical model location:** In the installed DocTR package, the PyTorch
  `db_resnet50` config points at
  `https://doctr-static.mindee.com/models?id=v0.7.0/db_resnet50-79bd7d70.pt&src=0`.
  Downloaded DocTR checkpoints are stored under `.cache/doctr/models` according
  to Mindee maintainer guidance.
- **Manual fallback:** Prefer the URL embedded in the installed
  `doctr.models.detection.differentiable_binarization.pytorch` module for the
  exact package version in use, because DocTR can change checkpoint hashes by
  release.
- **Primary references:**
  - Paper: https://arxiv.org/abs/1911.08947
  - AAAI record: https://ojs.aaai.org/index.php/AAAI/article/view/6812
  - DocTR project: https://github.com/mindee/doctr
  - DocTR maintainer note on local checkpoints:
    https://github.com/mindee/doctr/discussions/1350

## EasyOCR / CRAFT

- **Model used here:** EasyOCR's default detection network, CRAFT, plus the
  English recognition model required by `easyocr.Reader(["en"], verbose=False)`.
- **Invented by:** CRAFT was proposed by Youngmin Baek, Bado Lee, Dongyoon Han,
  Sangdoo Yun, and Hwalsuk Lee at Clova AI Research, NAVER Corp. EasyOCR is
  maintained by JaidedAI/jDai.
- **Publication:** "Character Region Awareness for Text Detection," CVPR 2019.
  The paper describes CRAFT as detecting text areas by modeling character
  regions and affinities between characters.
- **How we ingest it:** `get_easyocr_reader()` imports `easyocr` lazily and
  constructs `easyocr.Reader(["en"], verbose=False)`. EasyOCR downloads missing
  model files automatically when `download_enabled=True`, its default.
- **Canonical model location:** EasyOCR's public model hub lists manual
  downloads for recognition models and the CRAFT detection model. The installed
  package config currently points CRAFT at
  `https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip`
  and English recognition at
  `https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip`.
- **Manual fallback:** Download the needed files from the EasyOCR model hub and
  place them in `~/.EasyOCR/model/`, or use EasyOCR's documented
  `model_storage_directory` option to point at another cache.
- **Primary references:**
  - Paper: https://arxiv.org/abs/1904.01941
  - Official CRAFT implementation: https://github.com/clovaai/CRAFT-pytorch
  - EasyOCR API docs: https://www.jaided.ai/easyocr/documentation/
  - EasyOCR model hub: https://www.jaided.ai/easyocr/modelhub/
