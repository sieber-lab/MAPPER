"""Place supplied SVG components and export complete vector figures.

Artwork remains vector geometry; numerical plots are always drawn from tables.
The final SVG is assembled in code and converted with librsvg/Cairo. No manual
post-processing or frozen MAPPER plot PDFs is used. Declared third-party
STRING-web exports are the sole pre-rendered enrichment inputs.
"""

from __future__ import annotations

import copy
import ctypes as C
import ctypes.util
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import pandas as pd

from .style import INPUTS, OUTPUTS, USED_INPUTS, provenance, validate_style
from .svg_geometry import normalize_strokes, fit_figure5_artwork
from .growth_labels import arrange as arrange_growth_labels
from .centering import center_svg, visible_bounds

SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)


def reserve(ax, path, *, inset=0, chemical_role=None, view_box=None, text_patches=None):
    """Use an axis rectangle to locate a vector artwork asset in the export."""
    ax.set_axis_off()
    USED_INPUTS.add("artwork/" + path)
    bbox = ax.get_position()
    ax.figure._artwork = getattr(ax.figure, "_artwork", []) + [
        dict(
            path=str(path),
            rect=[
                bbox.x0 + inset,
                bbox.y0 + inset,
                bbox.width - 2 * inset,
                bbox.height - 2 * inset,
            ],
            chemical_role=chemical_role,
            view_box=view_box,
            text_patches=text_patches,
        )
    ]


def insert_svg(
    tree,
    source,
    rect,
    page_width,
    page_height,
    number,
    chemical_scale=None,
    chemical_stroke=None,
    view_box=None,
    text_patches=None,
):
    child = ET.parse(INPUTS / "artwork" / source).getroot()
    # Relabel declared annotation regions on a copy of the fixed artwork.
    # Frozen SVG files, enrichment values, circles and all other labels stay intact.
    for index, patch in enumerate(text_patches or []):
        x0, y0, x1, y1 = patch["bbox"]
        ET.SubElement(
            child,
            f"{{{SVG}}}rect",
            dict(
                x=str(x0 - 0.35),
                y=str(y0 - 0.35),
                width=str(x1 - x0 + 0.7),
                height=str(y1 - y0 + 0.7),
                fill="white",
                id=f"text_patch_mask_{index}",
            ),
        )
        text = ET.SubElement(
            child,
            f"{{{SVG}}}text",
            dict(
                x=str(patch["x"]), y=str(patch["y"]), fill="black", id=f"text_patch_label_{index}"
            ),
        )
        text.set("font-family", "Arial, Arimo, sans-serif")
        text.set("font-size", str(patch["font_size"]))
        text.text = patch["replacement"]
    if chemical_scale is not None:
        normalize_strokes(child, chemical_scale, width_pt=chemical_stroke)
    if view_box is not None:
        child.set("viewBox", " ".join(str(v) for v in view_box))
    identifiers = {
        e.attrib["id"]: f"asset{number}_{e.attrib['id']}" for e in child.iter() if "id" in e.attrib
    }
    classes = {
        name: f"asset{number}_{name}" for e in child.iter() for name in e.get("class", "").split()
    }
    for e in child.iter():
        if "id" in e.attrib:
            e.set("id", identifiers[e.attrib["id"]])
        if "class" in e.attrib:
            e.set("class", " ".join(classes[name] for name in e.get("class").split()))
        for key, value in list(e.attrib.items()):
            for old, new in identifiers.items():
                value = value.replace("url(#" + old + ")", "url(#" + new + ")")
                if value == "#" + old:
                    value = "#" + new
            e.set(key, value)
        if e.tag == f"{{{SVG}}}style" and e.text:
            css = e.text
            for old, new in classes.items():
                css = re.sub(r"\." + re.escape(old) + r"(?![\w-])", "." + new, css)
            for old, new in identifiers.items():
                css = css.replace("url(#" + old + ")", "url(#" + new + ")")
                css = re.sub(r"#" + re.escape(old) + r"(?![\w-])", "#" + new, css)
            e.text = css
    x, y, width, height = rect
    child.set("x", str(x * page_width))
    child.set("y", str((1 - y - height) * page_height))
    child.set("width", str(width * page_width))
    child.set("height", str(height * page_height))
    child.set("preserveAspectRatio", "xMidYMid meet")
    if source.startswith("original_layout/"):
        # Original PDF crops have an opaque white background. Put them below
        # the freshly generated panel letters and similarity annotations.
        figure_group = tree.find(f"{{{SVG}}}g[@id='figure_1']")
        if figure_group is None:
            raise ValueError("Missing Matplotlib figure group")
        background = next(i for i, node in enumerate(figure_group) if node.get("id") == "patch_1")
        figure_group.insert(background + 1, child)
    else:
        tree.append(child)


def render_svg(source, target, *, png=False, dpi=600, width_pt=None):
    """Use system libraries without depending on a GUI or an editor install."""
    svg = C.CDLL(ctypes.util.find_library("rsvg-2"))
    cairo = C.CDLL(ctypes.util.find_library("cairo"))
    gobject = C.CDLL(ctypes.util.find_library("gobject-2.0"))

    class Rectangle(C.Structure):
        _fields_ = [(name, C.c_double) for name in ("x", "y", "width", "height")]

    svg.rsvg_handle_new_from_data.argtypes = [C.c_void_p, C.c_size_t, C.c_void_p]
    svg.rsvg_handle_new_from_data.restype = C.c_void_p
    svg.rsvg_handle_render_document.argtypes = [
        C.c_void_p,
        C.c_void_p,
        C.POINTER(Rectangle),
        C.c_void_p,
    ]
    svg.rsvg_handle_render_document.restype = C.c_int
    cairo.cairo_image_surface_create.argtypes = [C.c_int, C.c_int, C.c_int]
    cairo.cairo_image_surface_create.restype = C.c_void_p
    cairo.cairo_pdf_surface_create.argtypes = [C.c_char_p, C.c_double, C.c_double]
    cairo.cairo_pdf_surface_create.restype = C.c_void_p
    cairo.cairo_create.argtypes = [C.c_void_p]
    cairo.cairo_create.restype = C.c_void_p
    for method in ("cairo_destroy", "cairo_surface_destroy", "cairo_surface_finish", "cairo_paint"):
        getattr(cairo, method).argtypes = [C.c_void_p]
    cairo.cairo_set_source_rgb.argtypes = [C.c_void_p, C.c_double, C.c_double, C.c_double]
    cairo.cairo_surface_write_to_png.argtypes = [C.c_void_p, C.c_char_p]
    cairo.cairo_surface_write_to_png.restype = C.c_int
    gobject.g_object_unref.argtypes = [C.c_void_p]
    node = ET.parse(source).getroot()
    _, _, width, height = map(float, node.attrib["viewBox"].split())
    node.set("width", f"{width}pt")
    node.set("height", f"{height}pt")
    content = ET.tostring(node)
    buffer = C.create_string_buffer(content)
    handle = svg.rsvg_handle_new_from_data(buffer, len(content), None)
    if not handle:
        raise RuntimeError("SVG parse failed: " + str(source))
    if width_pt is not None:
        height = height * width_pt / width
        width = width_pt
    if png:
        width, height = round(width * dpi / 72), round(height * dpi / 72)
        surface = cairo.cairo_image_surface_create(0, int(width), int(height))
    else:
        surface = cairo.cairo_pdf_surface_create(str(target).encode(), width, height)
    context = cairo.cairo_create(surface)
    cairo.cairo_set_source_rgb(context, 1, 1, 1)
    cairo.cairo_paint(context)
    rect = Rectangle(0, 0, width, height)
    if not svg.rsvg_handle_render_document(handle, context, C.byref(rect), None):
        raise RuntimeError("SVG rendering failed")
    if png and cairo.cairo_surface_write_to_png(surface, str(target).encode()) != 0:
        raise RuntimeError("PNG export failed")
    cairo.cairo_destroy(context)
    cairo.cairo_surface_finish(surface)
    cairo.cairo_surface_destroy(surface)
    gobject.g_object_unref(handle)


def export(fig, figure_id, dpi=600):
    """Write the final PDF, editable SVG, high-res PNG and light preview."""
    destination = OUTPUTS / figure_id
    destination.mkdir(parents=True, exist_ok=True)
    page_width, page_height = fig.get_size_inches() * 72
    compound_scaling = fit_figure5_artwork(
        getattr(fig, "_artwork", []), page_width, page_height, INPUTS
    )
    styling = validate_style(fig)
    styling["figure5_compound_scaling"] = compound_scaling
    styling["growth_labels"] = arrange_growth_labels(fig)
    # Recheck after moving labels and adding their leaders.
    refreshed = validate_style(fig)
    styling.update(refreshed)
    source = destination / (figure_id + ".svg")
    # Fixed page size is essential: artwork locations use normalized figure
    # coordinates, not a moving tight bounding box.
    fig.savefig(source, format="svg", facecolor="white", bbox_inches=None)
    root = ET.parse(source).getroot()
    _, _, width, height = map(float, root.attrib["viewBox"].split())
    styling["original_compound_artwork"] = [
        dict(path=item["path"], rect=item["rect"], bond_strokes_modified=False)
        for item in getattr(fig, "_artwork", [])
        if item["path"].startswith("original_layout/")
    ]
    styling["artwork_text_replacements"] = [
        dict(path=item["path"], patches=item["text_patches"])
        for item in getattr(fig, "_artwork", [])
        if item.get("text_patches")
    ]
    for i, item in enumerate(getattr(fig, "_artwork", [])):
        insert_svg(
            root,
            item["path"],
            item["rect"],
            width,
            height,
            i,
            item.get("chemical_scale"),
            item.get("chemical_stroke"),
            item.get("view_box"),
            item.get("text_patches"),
        )
    ET.ElementTree(root).write(source, encoding="utf-8", xml_declaration=True)
    preview = destination / (figure_id + "_preview.png")
    render_svg(source, preview, png=True, dpi=150)
    styling["centering"] = center_svg(
        source, preview, alignment=getattr(fig, "_horizontal_alignment", "center")
    )
    render_svg(source, preview, png=True, dpi=150)
    styling["centering"]["final_visible_bounds"] = visible_bounds(preview)
    render_svg(source, destination / (figure_id + ".pdf"))
    render_svg(source, destination / (figure_id + ".png"), png=True, dpi=dpi)
    provenance(figure_id, destination, styling)
    if getattr(fig, "_condition_keys", None):
        pd.DataFrame(fig._condition_keys).drop_duplicates().to_csv(
            destination / "condition_keys.csv", index=False
        )
    plt.close(fig)
    return destination / (figure_id + ".pdf")
