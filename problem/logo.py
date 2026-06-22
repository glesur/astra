from astrapy import *
import numpy as np
#import matplotlib.pyplot as plt
try:
    from PIL import Image, ImageDraw, ImageFont
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "problem/logo.py requires Pillow (PIL). Install it with `python -m pip install pillow`."
    ) from e

def init_flow(grid, field):
    field["vx1"][:,:,:] = 0.0
    field["vx2"][:,:,:] = 0.0
    field["vx3"][:,:,:] = 0.0

    arr = text_to_array("ASTRA", size=(grid.npr_glob[IDIR], grid.npr_glob[JDIR]), font_size=512, padding=10, align="center", fg=1.0, bg=0.0)
    for k in range(grid.npr_glob[KDIR]):
        field["vx3"][:,:,k] = arr.T


def _load_font(font_size: int) -> ImageFont.FreeTypeFont:
    """Essaie de charger une police TrueType, sinon utilise la police par défaut."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",               # macOS
        "C:/Windows/Fonts/arialbd.ttf",                # Windows
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, font_size)
        except (IOError, OSError):
            continue
    # Repli : police bitmap intégrée à Pillow (taille fixe ~11px)
    return ImageFont.load_default()


def _measure_text(text: str, font: ImageFont.FreeTypeFont):
    """Retourne (width, height) du texte rendu avec la police donnée."""
    dummy = Image.new("L", (1, 1))
    bbox  = ImageDraw.Draw(dummy).textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1], bbox


def _fit_font_to_box(text: str, box_w: int, box_h: int, padding: int) -> ImageFont.FreeTypeFont:
    """
    Cherche par dichotomie la plus grande taille de police dont le texte
    tient dans (box_w - 2*padding) × (box_h - 2*padding).
    """
    avail_w = max(1, box_w - 2 * padding)
    avail_h = max(1, box_h - 2 * padding)
    lo, hi = 4, max(box_w, box_h) * 2   # bornes de recherche

    best_font = _load_font(lo)
    while lo <= hi:
        mid  = (lo + hi) // 2
        font = _load_font(mid)
        tw, th, _ = _measure_text(text, font)
        if tw <= avail_w and th <= avail_h:
            best_font = font
            lo = mid + 1
        else:
            hi = mid - 1
    return best_font


def text_to_array(
    text: str,
    size: "tuple[int, int] | None" = None,
    font_size: int = 64,
    padding: int = 10,
    align: str = "center",
    fg: float = 1.0,
    bg: float = 0.0,
) -> np.ndarray:
    """
    Convertit une chaîne de texte en tableau numpy 2D.

    Paramètres
    ----------
    text      : texte à rasteriser
    size      : (largeur, hauteur) du tableau de sortie en pixels.
                Si None, la taille est calculée automatiquement à partir
                du texte et de font_size.
    font_size : taille de la police (ignoré si size est fourni — la police
                est alors ajustée automatiquement pour remplir le cadre)
    padding   : marge intérieure en pixels (utilisé dans les deux modes)
    align     : alignement du texte dans le cadre : 'center', 'left', 'right'
                (ignoré si size est None)
    fg        : valeur des pixels "texte"  (défaut 1.0)
    bg        : valeur des pixels "fond"   (défaut 0.0)

    Retourne
    --------
    np.ndarray de forme (hauteur, largeur), dtype float32
    """
    if size is None:
        # ── Mode auto : taille déduite du texte ──────────────────────────
        font = _load_font(font_size)
        tw, th, bbox = _measure_text(text, font)
        w = tw + 2 * padding
        h = th + 2 * padding
        x0 = padding - bbox[0]
        y0 = padding - bbox[1]
    else:
        # ── Mode taille fixe : police ajustée pour remplir (w, h) ────────
        w, h = int(size[0]), int(size[1])
        font = _fit_font_to_box(text, w, h, padding)
        tw, th, bbox = _measure_text(text, font)

        # Alignement horizontal
        avail_w = w - 2 * padding
        if align == "center":
            x0 = padding + (avail_w - tw) // 2 - bbox[0]
        elif align == "right":
            x0 = w - padding - tw - bbox[0]
        else:  # "left"
            x0 = padding - bbox[0]

        # Centrage vertical systématique
        avail_h = h - 2 * padding
        y0 = padding + (avail_h - th) // 2 - bbox[1]

    # ── Rendu ────────────────────────────────────────────────────────────
    img  = Image.new("L", (w, h), color=0)
    draw = ImageDraw.Draw(img)
    draw.text((x0, y0), text, fill=255, font=font)

    # ── Normalisation vers [bg, fg] ───────────────────────────────────────
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = bg + arr * (fg - bg)
    return arr
