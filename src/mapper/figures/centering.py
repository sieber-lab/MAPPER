"""Center complete visible vector compositions, including their labels."""

import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

SVG = "http://www.w3.org/2000/svg"


def visible_bounds(path):
    pixels = np.asarray(Image.open(path).convert("RGB"))
    yy, xx = np.where(pixels.min(axis=2) < 245)
    if not len(xx):
        raise ValueError("Empty figure preview")
    return dict(
        left=int(xx.min()),
        right=int(xx.max()) + 1,
        top=int(yy.min()),
        bottom=int(yy.max()) + 1,
        page_width=pixels.shape[1],
        page_height=pixels.shape[0],
    )


def center_svg(path, preview, alignment="center"):
    bounds = visible_bounds(preview)
    tree = ET.parse(path)
    root = tree.getroot()
    width = float(root.get("viewBox").split()[2])
    if alignment == "center":
        ink_center = (bounds["left"] + bounds["right"]) / 2
        shift = (bounds["page_width"] / 2 - ink_center) * width / bounds["page_width"]
    elif alignment == "left":
        shift = (0.03 * bounds["page_width"] - bounds["left"]) * width / bounds["page_width"]
    else:
        raise ValueError(alignment)
    if abs(shift) > 0.05:
        group = ET.Element(
            f"{{{SVG}}}g", {"transform": f"translate({shift:.8f} 0)", "id": "centered_composition"}
        )
        for child in list(root):
            if child.tag.rsplit("}", 1)[-1] in {"metadata", "defs", "style"}:
                continue
            root.remove(child)
            group.append(child)
        root.append(group)
        tree.write(path, encoding="utf-8", xml_declaration=True)
    return dict(horizontal_shift_pt=shift, original_visible_bounds=bounds, alignment=alignment)
