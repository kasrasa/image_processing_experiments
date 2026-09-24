"""Metadata and UI control definitions for every explorer operation."""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent
from typing import Any


@dataclass(frozen=True)
class ControlSpec:
    key: str
    label: str
    kind: str
    default: Any
    help: str
    minimum: int | float | None = None
    maximum: int | float | None = None
    step: int | float | None = None
    options: tuple[Any, ...] = ()
    format: str | None = None


@dataclass(frozen=True)
class OperationSpec:
    key: str
    category: str
    name: str
    summary: str
    explanation: str
    best_for: str
    watch_for: str
    controls: tuple[ControlSpec, ...]
    sweep_parameter: str | None = None


CATEGORY_ORDER = (
    "Color spaces",
    "Smoothing",
    "Edges",
    "Thresholding",
    "Contrast",
    "Shapes",
    "Geometry",
    "Augmentation",
)


OPERATION_LIST = (
    OperationSpec(
        key="hsv_adjust",
        category="Color spaces",
        name="HSV controls",
        summary="Shift hue and independently scale color intensity and brightness.",
        explanation=(
            "HSV separates color type (hue), color intensity (saturation), and "
            "brightness (value). This often makes color-based changes easier to reason "
            "about than editing red, green, and blue channels directly."
        ),
        best_for="Color augmentation, segmentation experiments, and lighting studies.",
        watch_for="Large saturation or value gains can clip detail at 0 or 255.",
        controls=(
            ControlSpec(
                "hue_shift",
                "Hue shift",
                "slider",
                0,
                "Rotates colors around the hue wheel; 180° reaches the opposite hue.",
                -180,
                180,
                5,
                format="%d°",
            ),
            ControlSpec(
                "saturation_scale",
                "Saturation",
                "slider",
                1.0,
                "0 removes color; values above 1 intensify it.",
                0.0,
                2.0,
                0.05,
                format="%.2f×",
            ),
            ControlSpec(
                "value_scale",
                "Brightness (value)",
                "slider",
                1.0,
                "Scales the HSV value channel without directly changing hue.",
                0.25,
                1.75,
                0.05,
                format="%.2f×",
            ),
        ),
        sweep_parameter="saturation_scale",
    ),
    OperationSpec(
        key="lab_adjust",
        category="Color spaces",
        name="LAB controls",
        summary="Edit perceived lightness separately from green–red and blue–yellow color axes.",
        explanation=(
            "LAB is designed around human color perception. L represents lightness, while "
            "a and b carry chromatic information. Chroma scaling moves both color axes away "
            "from or toward their neutral midpoint."
        ),
        best_for="Color correction, contrast work, and perceptually meaningful color shifts.",
        watch_for="Strong a/b shifts create color casts and can push channels into clipping.",
        controls=(
            ControlSpec(
                "lightness_scale",
                "Lightness",
                "slider",
                1.0,
                "Scales the L channel.",
                0.4,
                1.6,
                0.05,
                format="%.2f×",
            ),
            ControlSpec(
                "chroma_scale",
                "Chroma",
                "slider",
                1.0,
                "Scales distance from neutral on both a and b axes.",
                0.25,
                2.0,
                0.05,
                format="%.2f×",
            ),
            ControlSpec(
                "a_shift",
                "Green ↔ red shift",
                "slider",
                0,
                "Negative values add green; positive values add red/magenta.",
                -60,
                60,
                2,
            ),
            ControlSpec(
                "b_shift",
                "Blue ↔ yellow shift",
                "slider",
                0,
                "Negative values add blue; positive values add yellow.",
                -60,
                60,
                2,
            ),
        ),
        sweep_parameter="chroma_scale",
    ),
    OperationSpec(
        key="box_blur",
        category="Smoothing",
        name="Box blur",
        summary="Replace every pixel with the average of a square neighborhood.",
        explanation=(
            "A box filter gives every pixel in the kernel equal weight. It is fast and easy "
            "to understand, but it tends to produce a flatter look than Gaussian blur."
        ),
        best_for="Simple denoising, downsampling preparation, and learning kernel behavior.",
        watch_for="Large kernels remove edges and fine structure quickly.",
        controls=(
            ControlSpec(
                "kernel_size",
                "Kernel size",
                "slider",
                7,
                "The width and height of the averaging neighborhood.",
                1,
                31,
                2,
                format="%d px",
            ),
        ),
        sweep_parameter="kernel_size",
    ),
    OperationSpec(
        key="gaussian_blur",
        category="Smoothing",
        name="Gaussian blur",
        summary="Smooth noise with a weighted neighborhood that favors nearby pixels.",
        explanation=(
            "Gaussian blur weights pixels using a bell-shaped curve. Nearby pixels matter "
            "more than distant ones, usually producing a more natural result than box blur."
        ),
        best_for="Noise reduction before edge detection and scale-space experiments.",
        watch_for="Kernel size limits the neighborhood; sigma controls how broadly weight is spread.",
        controls=(
            ControlSpec(
                "kernel_size",
                "Kernel size",
                "slider",
                9,
                "Odd-sized window used for the blur.",
                1,
                41,
                2,
                format="%d px",
            ),
            ControlSpec(
                "sigma",
                "Sigma",
                "slider",
                2.0,
                "Standard deviation of the Gaussian weighting curve.",
                0.0,
                12.0,
                0.25,
                format="%.2f",
            ),
        ),
        sweep_parameter="kernel_size",
    ),
    OperationSpec(
        key="bilateral_filter",
        category="Smoothing",
        name="Bilateral filter",
        summary="Smooth similar regions while preserving strong edges.",
        explanation=(
            "The bilateral filter considers both spatial distance and color difference. Pixels "
            "across a strong edge influence one another less, so edges survive more readily."
        ),
        best_for="Denoising where object boundaries should remain crisp.",
        watch_for="It is slower than basic blur and high sigma values can create a stylized look.",
        controls=(
            ControlSpec(
                "diameter",
                "Neighborhood diameter",
                "slider",
                9,
                "Diameter of each pixel neighborhood.",
                3,
                25,
                2,
                format="%d px",
            ),
            ControlSpec(
                "sigma_color",
                "Color sigma",
                "slider",
                75.0,
                "How different pixel colors may be and still mix.",
                10.0,
                200.0,
                5.0,
            ),
            ControlSpec(
                "sigma_space",
                "Spatial sigma",
                "slider",
                75.0,
                "How far away neighboring pixels may influence the result.",
                10.0,
                200.0,
                5.0,
            ),
        ),
        sweep_parameter="sigma_color",
    ),
    OperationSpec(
        key="sobel_edges",
        category="Edges",
        name="Sobel gradients",
        summary="Measure brightness change along x, y, or both directions.",
        explanation=(
            "Sobel kernels approximate the first image derivative. The x derivative responds "
            "to vertical edges, the y derivative to horizontal edges, and magnitude combines both."
        ),
        best_for="Understanding gradients, feature engineering, and directional edge detection.",
        watch_for="Larger kernels respond over a wider area and can amplify broad transitions.",
        controls=(
            ControlSpec(
                "kernel_size",
                "Kernel size",
                "select",
                3,
                "Sobel supports 1, 3, 5, or 7-pixel derivative kernels.",
                options=(1, 3, 5, 7),
            ),
            ControlSpec(
                "direction",
                "Gradient view",
                "select",
                "Magnitude (dx + dy)",
                "Choose one derivative or their combined strength.",
                options=(
                    "Magnitude (dx + dy)",
                    "Horizontal changes (dx)",
                    "Vertical changes (dy)",
                ),
            ),
        ),
    ),
    OperationSpec(
        key="canny_edges",
        category="Edges",
        name="Canny edges",
        summary="Find thin, connected edges using a pair of confidence thresholds.",
        explanation=(
            "Canny suppresses non-maximum gradient responses, then uses hysteresis: pixels above "
            "the high threshold start edges, while connected pixels above the low threshold may join."
        ),
        best_for="Object outlines, contour preparation, and classical vision pipelines.",
        watch_for="Thresholds that are too low collect noise; values that are too high miss weak edges.",
        controls=(
            ControlSpec(
                "thresholds",
                "Low / high thresholds",
                "range",
                (60, 150),
                "The low threshold follows weak connected edges; the high threshold starts strong ones.",
                0,
                255,
                1,
            ),
            ControlSpec(
                "blur_kernel",
                "Pre-blur kernel",
                "slider",
                5,
                "Gaussian smoothing applied before Canny; 1 disables it.",
                1,
                15,
                2,
                format="%d px",
            ),
        ),
        sweep_parameter="blur_kernel",
    ),
    OperationSpec(
        key="binary_threshold",
        category="Thresholding",
        name="Manual binary threshold",
        summary="Split grayscale pixels into black and white using a chosen cutoff.",
        explanation=(
            "Pixels on one side of the threshold become 0 and pixels on the other become 255. "
            "Inverse mode simply swaps which side is white."
        ),
        best_for="Simple segmentation when foreground and background brightness are predictable.",
        watch_for="One global value struggles when lighting varies across the image.",
        controls=(
            ControlSpec(
                "threshold",
                "Threshold",
                "slider",
                128,
                "Grayscale cutoff between black and white output.",
                0,
                255,
                1,
            ),
            ControlSpec(
                "inverse",
                "Invert output",
                "checkbox",
                False,
                "Swap black and white classes.",
            ),
        ),
        sweep_parameter="threshold",
    ),
    OperationSpec(
        key="otsu_threshold",
        category="Thresholding",
        name="Otsu threshold",
        summary="Let the image histogram choose a global binary cutoff automatically.",
        explanation=(
            "Otsu's method searches for the threshold that minimizes variation within the two "
            "resulting pixel groups. It works best when the histogram has two distinct modes."
        ),
        best_for="Automatic foreground/background separation under even lighting.",
        watch_for="A noisy, multimodal, or unevenly lit image may not have one useful global cutoff.",
        controls=(
            ControlSpec(
                "inverse",
                "Invert output",
                "checkbox",
                False,
                "Swap black and white classes after the automatic cutoff is selected.",
            ),
        ),
    ),
    OperationSpec(
        key="histogram_equalization",
        category="Contrast",
        name="Histogram equalization",
        summary="Redistribute intensity values to use more of the available tonal range.",
        explanation=(
            "Global histogram equalization remaps brightness using its cumulative distribution. "
            "Color mode edits only luminance to avoid independently distorting RGB channels."
        ),
        best_for="Low-contrast images with a narrow, broadly consistent brightness range.",
        watch_for="Global equalization can over-amplify noise or make local regions look harsh.",
        controls=(
            ControlSpec(
                "output_mode",
                "Output",
                "select",
                "Color (luminance)",
                "Apply equalization to color luminance or return a grayscale result.",
                options=("Color (luminance)", "Grayscale"),
            ),
        ),
    ),
    OperationSpec(
        key="clahe",
        category="Contrast",
        name="CLAHE",
        summary="Improve local contrast while limiting excessive amplification.",
        explanation=(
            "CLAHE equalizes small tiles rather than the whole image at once. Its clip limit caps "
            "large histogram peaks, reducing the tendency to amplify noise without bound."
        ),
        best_for="Uneven lighting, medical images, and recovering local detail.",
        watch_for="Tiny tiles or a high clip limit can emphasize noise and tile boundaries.",
        controls=(
            ControlSpec(
                "clip_limit",
                "Clip limit",
                "slider",
                2.0,
                "Limits how strongly any local histogram peak is amplified.",
                1.0,
                8.0,
                0.25,
            ),
            ControlSpec(
                "grid_size",
                "Tile grid",
                "slider",
                8,
                "Number of local tiles along each image dimension.",
                2,
                16,
                1,
                format="%d tiles",
            ),
            ControlSpec(
                "output_mode",
                "Output",
                "select",
                "Color (luminance)",
                "Apply CLAHE to LAB lightness or return a grayscale result.",
                options=("Color (luminance)", "Grayscale"),
            ),
        ),
        sweep_parameter="clip_limit",
    ),
    OperationSpec(
        key="contours",
        category="Shapes",
        name="Contour detection",
        summary="Trace connected edge boundaries and filter them by enclosed area.",
        explanation=(
            "This pipeline blurs the image, applies Canny, extracts external contours, and keeps "
            "only contours above the selected area. Green lines trace contours; yellow boxes show bounds."
        ),
        best_for="Counting objects, extracting regions, and inspecting shape candidates.",
        watch_for="Contours depend heavily on the edge map; broken edges can split one object into pieces.",
        controls=(
            ControlSpec(
                "thresholds",
                "Canny thresholds",
                "range",
                (50, 140),
                "Low and high thresholds used to create the contour edge map.",
                0,
                255,
                1,
            ),
            ControlSpec(
                "min_area",
                "Minimum area",
                "slider",
                300,
                "Ignore small contours below this enclosed pixel area.",
                0,
                15000,
                50,
                format="%d px²",
            ),
            ControlSpec(
                "line_width",
                "Line width",
                "slider",
                3,
                "Thickness of the displayed contour.",
                1,
                9,
                1,
                format="%d px",
            ),
        ),
        sweep_parameter="min_area",
    ),
    OperationSpec(
        key="resize",
        category="Geometry",
        name="Resize & interpolation",
        summary="Change image scale and compare the resampling methods used to invent or discard pixels.",
        explanation=(
            "Interpolation estimates pixel values on the new grid. Area is strong for shrinking, "
            "linear is a fast default, cubic and Lanczos are smoother, and nearest preserves hard blocks."
        ),
        best_for="Preparing model inputs, thumbnails, pyramids, and learning resampling artifacts.",
        watch_for="Upscaling cannot recreate missing detail, even when the result looks smoother.",
        controls=(
            ControlSpec(
                "scale",
                "Scale",
                "slider",
                0.75,
                "Multiplier applied to width and height.",
                0.25,
                2.0,
                0.05,
                format="%.2f×",
            ),
            ControlSpec(
                "interpolation",
                "Interpolation",
                "select",
                "Area",
                "How OpenCV calculates pixels on the resized grid.",
                options=("Nearest", "Linear", "Cubic", "Area", "Lanczos"),
            ),
        ),
        sweep_parameter="scale",
    ),
    OperationSpec(
        key="rotate",
        category="Geometry",
        name="Rotation",
        summary="Rotate around the image center with optional canvas expansion.",
        explanation=(
            "OpenCV builds a 2×3 affine matrix combining rotation, scale, and translation. "
            "Expanding the canvas prevents corners from being cropped."
        ),
        best_for="Deskewing, augmentation, and understanding affine transforms.",
        watch_for="Repeated rotations resample pixels repeatedly and gradually soften detail.",
        controls=(
            ControlSpec(
                "angle",
                "Angle",
                "slider",
                20.0,
                "Counter-clockwise rotation in degrees.",
                -180.0,
                180.0,
                1.0,
                format="%.0f°",
            ),
            ControlSpec(
                "scale",
                "Scale",
                "slider",
                1.0,
                "Scale applied within the rotation matrix.",
                0.5,
                1.5,
                0.05,
                format="%.2f×",
            ),
            ControlSpec(
                "expand_canvas",
                "Expand canvas",
                "checkbox",
                True,
                "Grow the output so rotated corners remain visible.",
            ),
        ),
        sweep_parameter="angle",
    ),
    OperationSpec(
        key="flip",
        category="Geometry",
        name="Flip",
        summary="Mirror an image across its horizontal axis, vertical axis, or both.",
        explanation=(
            "Flipping rearranges pixels without interpolation: horizontal uses OpenCV code 1, "
            "vertical uses 0, and both axes use −1."
        ),
        best_for="Data augmentation, correcting mirrored images, and symmetry experiments.",
        watch_for="Text, handedness, and directional labels may become semantically incorrect.",
        controls=(
            ControlSpec(
                "direction",
                "Direction",
                "select",
                "Horizontal",
                "Select the axis or axes across which the image is mirrored.",
                options=("Horizontal", "Vertical", "Both axes"),
            ),
        ),
    ),
    OperationSpec(
        key="shear",
        category="Geometry",
        name="Shear",
        summary="Slant the image by shifting coordinates in proportion to the other axis.",
        explanation=(
            "A shear is an affine transform. Horizontal shear moves x according to y; vertical "
            "shear moves y according to x. The output canvas is expanded to retain the image."
        ),
        best_for="Perspective-like augmentation and learning affine matrix behavior.",
        watch_for="Strong shear distorts object geometry and increases the output canvas substantially.",
        controls=(
            ControlSpec(
                "horizontal",
                "Horizontal shear",
                "slider",
                0.2,
                "Horizontal displacement per unit of vertical position.",
                -0.6,
                0.6,
                0.05,
            ),
            ControlSpec(
                "vertical",
                "Vertical shear",
                "slider",
                0.0,
                "Vertical displacement per unit of horizontal position.",
                -0.6,
                0.6,
                0.05,
            ),
        ),
        sweep_parameter="horizontal",
    ),
    OperationSpec(
        key="gaussian_noise",
        category="Augmentation",
        name="Gaussian noise",
        summary="Add normally distributed random variation with a controlled standard deviation.",
        explanation=(
            "Zero-mean Gaussian noise perturbs pixels around their original values. A fixed random "
            "seed keeps the preview stable while parameters change. Values are clipped safely to 0–255."
        ),
        best_for="Robustness testing, augmentation, and studying denoising algorithms.",
        watch_for="Noise that is much stronger than the expected real sensor noise can hurt training.",
        controls=(
            ControlSpec(
                "sigma",
                "Noise sigma",
                "slider",
                20.0,
                "Standard deviation of the added pixel noise.",
                0.0,
                80.0,
                1.0,
            ),
            ControlSpec(
                "noise_mode",
                "Noise channels",
                "select",
                "Color",
                "Use independent noise per color channel or the same noise across all channels.",
                options=("Color", "Monochrome"),
            ),
        ),
        sweep_parameter="sigma",
    ),
    OperationSpec(
        key="cutout",
        category="Augmentation",
        name="Cutout / random erasing",
        summary="Mask a square region so a model cannot rely on every visible feature.",
        explanation=(
            "Cutout replaces one patch with a constant or mean value. Random placement uses a "
            "fixed seed here so comparisons remain reproducible."
        ),
        best_for="Image-classification augmentation and occlusion robustness experiments.",
        watch_for="A patch that routinely removes the whole subject changes the label semantics.",
        controls=(
            ControlSpec(
                "size_ratio",
                "Patch size",
                "slider",
                0.25,
                "Fraction of image width and height covered by the square patch.",
                0.05,
                0.7,
                0.05,
                format="%.2f",
            ),
            ControlSpec(
                "location",
                "Location",
                "select",
                "Center",
                "Place the patch in the center or at a reproducible random position.",
                options=("Center", "Random (fixed seed)"),
            ),
            ControlSpec(
                "fill",
                "Fill",
                "select",
                "Image mean",
                "Value used inside the erased patch.",
                options=("Image mean", "Black", "White"),
            ),
        ),
        sweep_parameter="size_ratio",
    ),
    OperationSpec(
        key="sharpen",
        category="Augmentation",
        name="Unsharp masking",
        summary="Increase local edge contrast by subtracting a blurred copy from the original.",
        explanation=(
            "Despite its name, unsharp masking sharpens: the blur estimates low-frequency content, "
            "and subtracting it boosts high-frequency detail and edges."
        ),
        best_for="Improving perceived crispness and studying high-frequency enhancement.",
        watch_for="High amounts exaggerate noise and create bright or dark halos around edges.",
        controls=(
            ControlSpec(
                "amount",
                "Sharpen amount",
                "slider",
                1.0,
                "Strength of the high-frequency detail added back.",
                0.0,
                3.0,
                0.1,
                format="%.1f×",
            ),
            ControlSpec(
                "radius",
                "Blur radius",
                "slider",
                1.5,
                "Gaussian sigma that determines the feature size being sharpened.",
                0.3,
                6.0,
                0.1,
            ),
        ),
        sweep_parameter="amount",
    ),
)


OPERATIONS = {operation.key: operation for operation in OPERATION_LIST}


def operations_for_category(category: str) -> tuple[OperationSpec, ...]:
    return tuple(
        operation for operation in OPERATION_LIST if operation.category == category
    )


def default_parameters(operation: OperationSpec) -> dict[str, Any]:
    return {control.key: control.default for control in operation.controls}


def code_snippet(key: str, parameters: dict[str, Any]) -> str:
    """Return a copy-ready OpenCV example matching the current controls."""
    p = parameters
    header = (
        "import cv2\n"
        "import numpy as np\n\n"
        "# image_rgb is a uint8 NumPy array in RGB order\n"
    )

    if key == "hsv_adjust":
        body = f"""
hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
hsv[:, :, 0] = np.mod(hsv[:, :, 0] + {p["hue_shift"]} / 2.0, 180)
hsv[:, :, 1] = np.clip(hsv[:, :, 1] * {p["saturation_scale"]}, 0, 255)
hsv[:, :, 2] = np.clip(hsv[:, :, 2] * {p["value_scale"]}, 0, 255)
result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
"""
    elif key == "lab_adjust":
        body = f"""
lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
lab[:, :, 0] = np.clip(lab[:, :, 0] * {p["lightness_scale"]}, 0, 255)
lab[:, :, 1] = np.clip((lab[:, :, 1] - 128) * {p["chroma_scale"]} + 128 + {p["a_shift"]}, 0, 255)
lab[:, :, 2] = np.clip((lab[:, :, 2] - 128) * {p["chroma_scale"]} + 128 + {p["b_shift"]}, 0, 255)
result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
"""
    elif key == "box_blur":
        body = f"result = cv2.blur(image_rgb, ({p['kernel_size']}, {p['kernel_size']}))"
    elif key == "gaussian_blur":
        body = (
            f"result = cv2.GaussianBlur(image_rgb, "
            f"({p['kernel_size']}, {p['kernel_size']}), sigmaX={p['sigma']})"
        )
    elif key == "bilateral_filter":
        body = (
            "result = cv2.bilateralFilter(\n"
            f"    image_rgb, {p['diameter']}, {p['sigma_color']}, {p['sigma_space']}\n)"
        )
    elif key == "sobel_edges":
        if p["direction"] == "Horizontal changes (dx)":
            gradient_code = "edges = np.abs(dx)"
        elif p["direction"] == "Vertical changes (dy)":
            gradient_code = "edges = np.abs(dy)"
        else:
            gradient_code = "edges = cv2.magnitude(dx, dy)"
        body = f"""
gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize={p["kernel_size"]})
dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize={p["kernel_size"]})
{gradient_code}
result = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
"""
    elif key == "canny_edges":
        low, high = p["thresholds"]
        body = f"""
gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
gray = cv2.GaussianBlur(gray, ({p["blur_kernel"]}, {p["blur_kernel"]}), 0)
result = cv2.Canny(gray, {low}, {high})
"""
    elif key == "binary_threshold":
        threshold_type = (
            "cv2.THRESH_BINARY_INV" if p["inverse"] else "cv2.THRESH_BINARY"
        )
        body = f"""
gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
_, result = cv2.threshold(gray, {p["threshold"]}, 255, {threshold_type})
"""
    elif key == "otsu_threshold":
        threshold_type = (
            "cv2.THRESH_BINARY_INV" if p["inverse"] else "cv2.THRESH_BINARY"
        )
        body = f"""
gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
chosen_threshold, result = cv2.threshold(
    gray, 0, 255, {threshold_type} | cv2.THRESH_OTSU
)
"""
    elif key == "histogram_equalization":
        if p["output_mode"] == "Grayscale":
            body = """
gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
result = cv2.equalizeHist(gray)
"""
        else:
            body = """
ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
"""
    elif key == "clahe":
        if p["output_mode"] == "Grayscale":
            body = f"""
gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
clahe = cv2.createCLAHE(
    clipLimit={p["clip_limit"]}, tileGridSize=({p["grid_size"]}, {p["grid_size"]})
)
result = clahe.apply(gray)
"""
        else:
            body = f"""
lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
clahe = cv2.createCLAHE(
    clipLimit={p["clip_limit"]}, tileGridSize=({p["grid_size"]}, {p["grid_size"]})
)
lab[:, :, 0] = clahe.apply(lab[:, :, 0])
result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
"""
    elif key == "contours":
        low, high = p["thresholds"]
        body = f"""
gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), {low}, {high})
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = [c for c in contours if cv2.contourArea(c) >= {p["min_area"]}]
result = image_rgb.copy()
cv2.drawContours(result, contours, -1, (28, 238, 177), {p["line_width"]}, cv2.LINE_AA)
for contour in contours:
    x, y, width, height = cv2.boundingRect(contour)
    cv2.rectangle(result, (x, y), (x + width, y + height), (255, 196, 67), 1)
"""
    elif key == "resize":
        interpolation = {
            "Nearest": "cv2.INTER_NEAREST",
            "Linear": "cv2.INTER_LINEAR",
            "Cubic": "cv2.INTER_CUBIC",
            "Area": "cv2.INTER_AREA",
            "Lanczos": "cv2.INTER_LANCZOS4",
        }[p["interpolation"]]
        body = f"""
height, width = image_rgb.shape[:2]
size = (max(1, round(width * {p["scale"]})), max(1, round(height * {p["scale"]})))
result = cv2.resize(image_rgb, size, interpolation={interpolation})
"""
    elif key == "rotate":
        if p["expand_canvas"]:
            canvas_code = """
cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
output_width = int(height * sine + width * cosine)
output_height = int(height * cosine + width * sine)
matrix[0, 2] += output_width / 2 - center[0]
matrix[1, 2] += output_height / 2 - center[1]
""".strip()
        else:
            canvas_code = "output_width, output_height = width, height"
        body = f"""
height, width = image_rgb.shape[:2]
center = (width / 2, height / 2)
matrix = cv2.getRotationMatrix2D(center, {p["angle"]}, {p["scale"]})
{canvas_code}
result = cv2.warpAffine(
    image_rgb,
    matrix,
    (output_width, output_height),
    borderMode=cv2.BORDER_REFLECT_101,
)
"""
    elif key == "flip":
        code = {"Horizontal": 1, "Vertical": 0, "Both axes": -1}[p["direction"]]
        body = f"result = cv2.flip(image_rgb, {code})"
    elif key == "shear":
        body = f"""
height, width = image_rgb.shape[:2]
matrix = np.float32([[1, {p["horizontal"]}, 0], [{p["vertical"]}, 1, 0]])
corners = np.float32([[[0, 0], [width, 0], [0, height], [width, height]]])
transformed = cv2.transform(corners, matrix)[0]
minimum, maximum = transformed.min(axis=0), transformed.max(axis=0)
matrix[:, 2] -= minimum
output_size = tuple(np.ceil(maximum - minimum).astype(int))
result = cv2.warpAffine(
    image_rgb, matrix, output_size, borderMode=cv2.BORDER_REFLECT_101
)
"""
    elif key == "gaussian_noise":
        noise_shape = (
            "image_rgb.shape"
            if p["noise_mode"] == "Color"
            else "(*image_rgb.shape[:2], 1)"
        )
        body = f"""
rng = np.random.default_rng(42)
noise = rng.normal(0, {p["sigma"]}, size={noise_shape})
result = np.clip(image_rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)
"""
    elif key == "cutout":
        if p["location"] == "Random (fixed seed)":
            position_code = """
rng = np.random.default_rng(42)
x0 = int(rng.integers(0, max(1, width - patch_width + 1)))
y0 = int(rng.integers(0, max(1, height - patch_height + 1)))
""".strip()
        else:
            position_code = """
x0 = (width - patch_width) // 2
y0 = (height - patch_height) // 2
""".strip()

        fill_code = {
            "Image mean": "fill_value = image_rgb.mean(axis=(0, 1)).astype(np.uint8)",
            "Black": "fill_value = 0",
            "White": "fill_value = 255",
        }[p["fill"]]
        body = f"""
result = image_rgb.copy()
height, width = result.shape[:2]
patch_width = max(1, round(width * {p["size_ratio"]}))
patch_height = max(1, round(height * {p["size_ratio"]}))
{position_code}
{fill_code}
result[y0:y0 + patch_height, x0:x0 + patch_width] = fill_value
"""
    elif key == "sharpen":
        body = f"""
blurred = cv2.GaussianBlur(image_rgb, (0, 0), sigmaX={p["radius"]})
result = cv2.addWeighted(image_rgb, {1.0 + p["amount"]:.1f}, blurred, {-p["amount"]:.1f}, 0)
result = np.clip(result, 0, 255).astype(np.uint8)
"""
    else:
        raise KeyError(f"Unknown operation: {key}")

    return header + dedent(body).strip() + "\n"
