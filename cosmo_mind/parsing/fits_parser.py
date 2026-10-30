"""
cosmo_mind.parsing.fits_parser
==============================

Parses FITS (Flexible Image Transport System) files — the lingua franca of
astronomical data — into CosmoMind ``Document`` objects.

Extracts:
* every HDU's header cards (as a plain dict),
* image data shape / dtype / bitpix,
* table column names/types/descriptions for binary/ASCII tables,
* a best-effort WCS sky-coordinate projection per image extension (when
  valid), which lets us correlate an observation with a catalog position.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
import warnings

from cosmo_mind.utils.logging import get_logger
from cosmo_mind.parsing.document import DocType, Document

log = get_logger("cosmo_mind.parsing.fits")


def _hdu_to_record(hdu: fits.PrimaryHDU | fits.ImageHDU | fits.BinTableHDU) -> dict:
    """Convert one HDU into a JSON-friendly metadata dict."""
    hdr = hdu.header
    rec: dict = {
        "extname": hdr.get("EXTNAME", ""),
        "extver": hdr.get("EXTVER", 1),
        "xtension": hdr.get("XTENSION", ""),
        "naxis": hdr.get("NAXIS", 0),
        "naxisn": [hdr.get(f"NAXIS{i}", 0) for i in range(1, hdr.get("NAXIS", 0) + 1)],
        "bitpix": hdr.get("BITPIX", 8),
        "datatype": hdr.get("DATATYPE", "image"),
        "object": hdr.get("OBJECT", ""),
        "telescop": hdr.get("TELESCOP", ""),
        "instrume": hdr.get("INSTRUME", ""),
        "filter": hdr.get("FILTER", ""),
        "exptime": hdr.get("EXPTIME", None),
        " RA": hdr.get("RA", None),
        " DEC": hdr.get("DEC", None),
        "ra_obj": hdr.get("RA_OBJ", hdr.get("RA", None)),
        "dec_obj": hdr.get("DEC_OBJ", hdr.get("DEC", None)),
    }
    # collect a sanitized full header
    cards: dict[str, str] = {}
    for k in hdr:
        val = hdr[k]
        if val is None:
            continue
        cards[k] = str(val)
    rec["header"] = cards
    return rec


def _wcs_summary(wcs: WCS) -> dict | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            if not wcs.wcs.naxis or 0 in wcs.wcs.naxis:
                return None
            crpix = wcs.wcs.crpix.tolist()
            crval = wcs.wcs.crval.tolist()
            cdelt = wcs.wcs.cdelt.tolist()
        return {
            "crpix": crpix,
            "crval": crval,
            "cdelt": cdelt,
            "ctype": list(wcs.wcs.ctype),
            "lon_axis": wcs.world_axis_excursion if hasattr(wcs, "world_axis_excursion") else None,
        }
    except Exception as e:
        log.debug(f"wcs extraction skipped: {e}")
        return None


def parse_fits(path: str | Path, title: str | None = None) -> Document:
    """Parse a FITS file at *path* into a :class:`Document`."""
    path = Path(path)
    with fits.open(path, memmap=False) as hdul:
        exts: list[dict] = []
        for idx, hdu in enumerate(hdul):
            rec = _hdu_to_record(hdu)
            rec["index"] = idx
            # image-specific
            if hasattr(hdu, "data") and hdu.data is not None and hdu.is_image:
                arr = np.asarray(hdu.data)
                rec["data_shape"] = list(arr.shape)
                rec["data_dtype"] = str(arr.dtype)
                rec["data_min"] = float(np.nanmin(arr)) if arr.size else None
                rec["data_max"] = float(np.nanmax(arr)) if arr.size else None
                try:
                    wcs = WCS(hdu.header)
                    w = _wcs_summary(wcs)
                    if w:
                        rec["wcs"] = w
                except Exception:
                    pass
            # table-specific
            elif isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)) and hdu.columns:
                cols = []
                for col in hdu.columns:
                    cols.append(
                        {
                            "name": col.name,
                            "format": col.format,
                            "unit": col.unit or "",
                            "dim": col.dim,
                            "description": col.desc or "",
                        }
                    )
                rec["table_columns"] = cols
                rec["n_rows"] = len(hdu.data) if hdu.data is not None else 0
            exts.append(rec)

    fits_headers = {str(e["index"]): e for e in exts}
    primary = exts[0] if exts else {}
    doc_id = f"fits:{hashlib.md5(str(path).encode()).hexdigest()[:12]}"
    full_text = "\n".join(
        f"{k}={v}" for e in exts for k, v in e.get("header", {}).items()
    )[:4000]

    return Document(
        doc_id=doc_id,
        source=str(path),
        title=title or primary.get("object", path.name),
        doc_type=DocType.FITS_FILE,
        full_text=full_text,
        fits_headers=fits_headers,
        metadata={
            "n_hdus": len(exts),
            "telescope": primary.get("telescop", ""),
            "instrument": primary.get("instrume", ""),
            "filter": primary.get("filter", ""),
        },
    )


def parse_fits_bytes(data: bytes, name: str = "upload.fits") -> Document:
    """Parse a FITS file held in memory (used by the agent tool path)."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as fh:
        fh.write(data)
        tmp = fh.name
    try:
        return parse_fits(tmp, title=name)
    finally:
        Path(tmp).unlink(missing_ok=True)


__all__ = ["parse_fits", "parse_fits_bytes"]
