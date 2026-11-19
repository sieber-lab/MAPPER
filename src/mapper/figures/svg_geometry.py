"""Display-scale chemical artwork without editing supplied source SVGs."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import numpy as np

NUMBER = r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?"
LINE = re.compile(rf"\s*M\s*({NUMBER})[ ,]+({NUMBER})\s*L\s*({NUMBER})[ ,]+({NUMBER})\s*")


def matrix(value):
    result = np.eye(3)
    for name, arguments in re.findall(r"(\w+)\(([^)]*)\)", value or ""):
        p = [float(x) for x in re.findall(NUMBER, arguments)]
        item = np.eye(3)
        if name == "matrix":
            a, b, c, d, e, f = p
            item = np.array([[a, c, e], [b, d, f], [0, 0, 1.0]])
        elif name == "translate":
            item[:2, 2] = [p[0], p[1] if len(p) > 1 else 0]
        elif name == "scale":
            item[0, 0] = p[0]
            item[1, 1] = p[1] if len(p) > 1 else p[0]
        elif name == "rotate":
            theta = np.deg2rad(p[0])
            c, s = np.cos(theta), np.sin(theta)
            item[:2, :2] = [[c, -s], [s, c]]
            if len(p) > 1:
                center = np.array(p[1:3])
                item[:2, 2] = center - item[:2, :2] @ center
        else:
            raise ValueError(f"Unsupported SVG transform: {name}")
        result = result @ item
    return result


def paths_with_transforms(root):
    def walk(node, parent):
        current = parent @ matrix(node.get("transform"))
        if node.tag.rsplit("}", 1)[-1] == "path":
            yield node, current
        for child in node:
            yield from walk(child, current)

    return walk(root, np.eye(3))


def bond_length(root):
    lengths = []
    for node, transform in paths_with_transforms(root):
        match = LINE.fullmatch(node.get("d", ""))
        if not match or "stroke-width" not in node.get("style", ""):
            continue
        x1, y1, x2, y2 = map(float, match.groups())
        length = np.linalg.norm(transform[:2, :2] @ np.array([x2 - x1, y2 - y1]))
        if length > 0:
            lengths.append(length)
    if len(lengths) < 3:
        raise ValueError("Insufficient chemical bond geometry")
    # Shortened bonds terminate at atom glyphs; the upper decile estimates
    # the nominal full-bond length rather than those shortened segments.
    return float(np.quantile(lengths, 0.9))


def normalize_strokes(root, scale, width_pt=0.25, schematic=False):
    for node, transform in paths_with_transforms(root):
        style = node.get("style", "")
        width = re.search(rf"stroke-width:({NUMBER})(?:px)?", style)
        if not width:
            continue
        if schematic and (float(width[1]) < 8 or not LINE.fullmatch(node.get("d", ""))):
            continue
        factor = float(np.sqrt(abs(np.linalg.det(transform[:2, :2])))) * scale
        if factor <= 0:
            continue
        style = re.sub(
            rf"stroke-width:{NUMBER}(?:px)?", f"stroke-width:{width_pt/factor:.8g}px", style
        )
        node.set("style", style + ";stroke-linecap:round;stroke-linejoin:round")
    if schematic:
        for node in root.iter():
            if node.tag.rsplit("}", 1)[-1] != "text":
                continue
            text = "".join(node.itertext()).strip()
            if text not in {"N", "O", "H", "F", "HN", "NH", "OH"}:
                continue
            style = node.get("style", "").replace("Arial-BoldMT", "ArialMT")
            style = re.sub(r"font-weight:\s*700", "font-weight:400", style)
            node.set("style", style)


def fit_chemical_artwork(items, page_width, page_height, inputs, target_bond_pt=5.5):
    chemical = [
        item for item in items if item["path"].split("/")[0] in {"figure4", "extended_data2"}
    ]
    information = []
    for item in chemical:
        root = ET.parse(inputs / "artwork" / item["path"]).getroot()
        _, _, width, height = map(float, root.get("viewBox").split())
        bond = bond_length(root)
        x, y, w, h = item["rect"]
        fit = min(w * page_width / width, h * page_height / height)
        information.append((item, width, height, bond, fit))
    if not information:
        return
    roles = {item.get("chemical_role") or "reference" for item, *_ in information}
    lengths = {
        role: min(
            6.0 if role == "example" else 4.5,
            *[
                bond * fit
                for item, _, _, bond, fit in information
                if (item.get("chemical_role") or "reference") == role
            ],
        )
        for role in roles
    }
    for item, width, height, bond, fit in information:
        common = lengths[item.get("chemical_role") or "reference"]
        x, y, w, h = item["rect"]
        scale = common / bond
        new_w = width * scale / page_width
        new_h = height * scale / page_height
        item["rect"] = [x + (w - new_w) / 2, y + (h - new_h) / 2, new_w, new_h]
        item["chemical_scale"] = scale
        item["bond_length_pt"] = common
        item["chemical_stroke"] = max(0.16, min(0.25, common * 0.045))


def fit_figure5_artwork(items, page_width, page_height, inputs):
    """Use one molecular bond scale, rather than stretching each to its box.

    Larger molecules therefore remain larger than small molecules. Sources are
    unchanged; display transforms and a thin display stroke are export options.
    """
    information = []
    for item in items:
        if not item["path"].startswith("figure5/"):
            continue
        root = ET.parse(inputs / "artwork" / item["path"]).getroot()
        _, _, width, height = map(float, root.get("viewBox").split())
        bond = bond_length(root)
        x, y, w, h = item["rect"]
        fit = min(w * page_width / width, h * page_height / height)
        information.append((item, width, height, bond, fit))
    if not information:
        return []
    common = min(5.5, *[bond * fit for _, _, _, bond, fit in information])
    report = []
    for item, width, height, bond, _ in information:
        x, y, w, h = item["rect"]
        scale = common / bond
        new_w = width * scale / page_width
        new_h = height * scale / page_height
        item["rect"] = [x + (w - new_w) / 2, y + (h - new_h) / 2, new_w, new_h]
        item["chemical_scale"] = scale
        item["chemical_stroke"] = 0.22
        labels = item.get("label_texts", [])
        for index, label in enumerate(labels):
            label.set_position(
                (item["rect"][0] + new_w / 2, item["rect"][1] - (4 + index * 10) / page_height)
            )
        report.append(dict(path=item["path"], bond_length_pt=common, bond_stroke_pt=0.22))
    return report
