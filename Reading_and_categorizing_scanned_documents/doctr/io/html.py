from typing import Any

from weasyprint import HTML

__all__ = ["read_html"]


def read_html(url: str, **kwargs: Any) -> bytes:
    """Read a PDF file and convert it into an image in numpy format

    >>> from doctr.io import read_html
    >>> doc = read_html("https://www.yoursite.com")

    Args:
    ----
        url: URL of the target web page
        **kwargs: keyword arguments from `weasyprint.HTML`

    Returns:
    -------
        decoded PDF file as a bytes stream
    """
    return HTML(url, **kwargs).write_pdf()
