from typing import Any, Optional

from doctr.models.builder import KIEDocumentBuilder

from ..classification.predictor import CropOrientationPredictor
from ..predictor.base import _OCRPredictor

__all__ = ["_KIEPredictor"]


class _KIEPredictor(_OCRPredictor):
    """Implements an object able to localize and identify text elements in a set of documents

    Args:
    ----
        assume_straight_pages: if True, speeds up the inference by assuming you only pass straight pages
            without rotated textual elements.
        straighten_pages: if True, estimates the page general orientation based on the median line orientation.
            Then, rotates page before passing it to the deep learning modules. The final predictions will be remapped
            accordingly. Doing so will improve performances for documents with page-uniform rotations.
        preserve_aspect_ratio: if True, resize preserving the aspect ratio (with padding)
        symmetric_pad: if True and preserve_aspect_ratio is True, pas the image symmetrically.
        kwargs: keyword args of `DocumentBuilder`
    """

    crop_orientation_predictor: Optional[CropOrientationPredictor]

    def __init__(
        self,
        assume_straight_pages: bool = True,
        straighten_pages: bool = False,
        preserve_aspect_ratio: bool = True,
        symmetric_pad: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(assume_straight_pages, straighten_pages, preserve_aspect_ratio, symmetric_pad, **kwargs)

        self.doc_builder: KIEDocumentBuilder = KIEDocumentBuilder(**kwargs)
