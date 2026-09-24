"""Interactive Streamlit interface for exploring classical OpenCV techniques."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.image_utils import (
    create_sample_image,
    image_dimensions,
    load_uploaded_image,
    mean_luminance,
    resize_to_limit,
)
from src.operation_registry import (
    CATEGORY_ORDER,
    ControlSpec,
    OperationSpec,
    code_snippet,
    operations_for_category,
)
from src.operations import OperationResult, apply_operation

st.set_page_config(
    page_title="OpenCV Parameter Explorer",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        :root {
            --lab-ink: #172033;
            --lab-muted: #64748b;
            --lab-accent: #0f766e;
            --lab-accent-soft: #ccfbf1;
            --lab-warm: #f59e0b;
            --lab-line: rgba(100, 116, 139, 0.22);
        }
        .block-container {
            max-width: 1440px;
            padding-top: 2.2rem;
            padding-bottom: 4rem;
        }
        .lab-hero {
            padding: 1.45rem 1.6rem;
            margin-bottom: 1.25rem;
            border: 1px solid var(--lab-line);
            border-radius: 18px;
            background:
                radial-gradient(circle at 92% 18%, rgba(20, 184, 166, 0.20), transparent 26%),
                linear-gradient(125deg, rgba(15, 118, 110, 0.08), rgba(245, 158, 11, 0.07));
        }
        .lab-eyebrow {
            color: var(--lab-accent);
            font-size: 0.76rem;
            font-weight: 750;
            letter-spacing: 0.13em;
            margin-bottom: 0.45rem;
            text-transform: uppercase;
        }
        .lab-hero h1 {
            color: var(--lab-ink);
            font-size: clamp(2rem, 4vw, 3.45rem);
            letter-spacing: -0.045em;
            line-height: 1.02;
            margin: 0;
        }
        .lab-hero p {
            color: var(--lab-muted);
            font-size: 1.03rem;
            line-height: 1.6;
            margin: 0.7rem 0 0;
            max-width: 760px;
        }
        .lab-technique {
            border-left: 4px solid var(--lab-accent);
            margin: 0.55rem 0 1.1rem;
            padding: 0.15rem 0 0.15rem 1rem;
        }
        .lab-technique strong {
            color: var(--lab-ink);
            display: block;
            font-size: 1.2rem;
            margin-bottom: 0.15rem;
        }
        .lab-technique span {
            color: var(--lab-muted);
            line-height: 1.5;
        }
        [data-testid="stImage"] img {
            border-radius: 10px;
        }
        [data-testid="stMetric"] {
            background: rgba(248, 250, 252, 0.72);
            border: 1px solid var(--lab-line);
            border-radius: 12px;
            padding: 0.75rem 0.9rem;
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid var(--lab-line);
        }
        .lab-callout {
            background: rgba(204, 251, 241, 0.35);
            border: 1px solid rgba(15, 118, 110, 0.18);
            border-radius: 12px;
            color: var(--lab-ink);
            line-height: 1.55;
            padding: 0.9rem 1rem;
        }
        @media (prefers-color-scheme: dark) {
            .lab-hero h1, .lab-technique strong, .lab-callout { color: #f8fafc; }
            .lab-hero p, .lab-technique span { color: #cbd5e1; }
            [data-testid="stMetric"] { background: rgba(15, 23, 42, 0.55); }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def sample_image() -> np.ndarray:
    return create_sample_image()


def image_as_png(image: np.ndarray) -> bytes:
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def render_control(control: ControlSpec, operation_key: str) -> Any:
    widget_key = f"control_{operation_key}_{control.key}"
    if control.kind == "slider":
        options: dict[str, Any] = {
            "label": control.label,
            "min_value": control.minimum,
            "max_value": control.maximum,
            "value": control.default,
            "step": control.step,
            "help": control.help,
            "key": widget_key,
        }
        if control.format:
            options["format"] = control.format
        return st.sidebar.slider(**options)
    if control.kind == "range":
        return st.sidebar.slider(
            control.label,
            min_value=control.minimum,
            max_value=control.maximum,
            value=control.default,
            step=control.step,
            help=control.help,
            key=widget_key,
        )
    if control.kind == "select":
        return st.sidebar.selectbox(
            control.label,
            options=control.options,
            index=control.options.index(control.default),
            help=control.help,
            key=widget_key,
        )
    if control.kind == "checkbox":
        return st.sidebar.checkbox(
            control.label,
            value=control.default,
            help=control.help,
            key=widget_key,
        )
    raise ValueError(f"Unsupported control kind: {control.kind}")


def format_parameter_value(value: Any) -> str:
    if isinstance(value, tuple):
        return " – ".join(str(item) for item in value)
    if isinstance(value, bool):
        return "On" if value else "Off"
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def _snap_to_control(value: float, control: ControlSpec) -> int | float:
    minimum = float(control.minimum)
    maximum = float(control.maximum)
    step = float(control.step or 1)
    snapped = minimum + round((value - minimum) / step) * step
    snapped = max(minimum, min(maximum, snapped))
    if isinstance(control.default, int):
        return round(snapped)
    return round(snapped, 4)


def sweep_variants(
    operation: OperationSpec,
    parameters: dict[str, Any],
) -> tuple[ControlSpec, list[tuple[str, dict[str, Any]]]] | None:
    if operation.sweep_parameter is None:
        return None

    control = next(
        item for item in operation.controls if item.key == operation.sweep_parameter
    )
    current = float(parameters[control.key])
    span = float(control.maximum) - float(control.minimum)
    offset = max(float(control.step or 1), span * 0.18)
    values = [
        _snap_to_control(current - offset, control),
        _snap_to_control(current, control),
        _snap_to_control(current + offset, control),
    ]

    unique_values: list[int | float] = []
    for value in values:
        if value not in unique_values:
            unique_values.append(value)

    variants: list[tuple[str, dict[str, Any]]] = []
    for value in unique_values:
        variant = parameters.copy()
        variant[control.key] = value
        variants.append((format_parameter_value(value), variant))
    return control, variants


def luminance_histogram(image: np.ndarray) -> np.ndarray:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    counts, _ = np.histogram(gray, bins=256, range=(0, 256))
    total = max(1, counts.sum())
    return counts / total


st.markdown(
    """
    <div class="lab-hero">
        <div class="lab-eyebrow">Interactive computer vision playground</div>
        <h1>OpenCV Parameter Explorer</h1>
        <p>
            See what each parameter changes, compare settings side by side, and copy the
            exact Python behind the result. Start with the built-in test card or upload your own image.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("## Experiment setup")
st.sidebar.caption("Your uploaded image is processed only in the running app session.")
image_source = st.sidebar.radio(
    "Image source",
    ("Built-in test image", "Upload an image"),
    key="image_source",
)

input_name = "opencv-sample"
was_resized = False
if image_source == "Upload an image":
    uploaded_file = st.sidebar.file_uploader(
        "Choose a JPG, PNG, or WebP file",
        type=("jpg", "jpeg", "png", "webp"),
        help="Large images are resized to a maximum side length of 1,400 pixels for responsiveness.",
    )
    if uploaded_file is None:
        st.sidebar.info(
            "Upload an image when ready. The test image is shown in the meantime."
        )
        source_image = sample_image().copy()
    else:
        try:
            source_image = load_uploaded_image(uploaded_file)
            source_image, was_resized = resize_to_limit(source_image)
            input_name = uploaded_file.name.rsplit(".", 1)[0]
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            st.sidebar.error(f"That image could not be read: {error}")
            source_image = sample_image().copy()
else:
    source_image = sample_image().copy()

if was_resized:
    st.sidebar.warning("The preview was resized to keep interactions fast.")

st.sidebar.divider()
selected_category = st.sidebar.selectbox(
    "Technique family",
    CATEGORY_ORDER,
    key="selected_category",
)
category_operations = operations_for_category(selected_category)
operation_names = tuple(operation.name for operation in category_operations)
selected_name = st.sidebar.selectbox(
    "Technique",
    operation_names,
    key="selected_operation_name",
)
operation = next(item for item in category_operations if item.name == selected_name)
st.sidebar.caption(operation.summary)

st.sidebar.markdown("### Parameters")
parameters = {
    control.key: render_control(control, operation.key)
    for control in operation.controls
}

try:
    result: OperationResult = apply_operation(operation.key, source_image, parameters)
except (cv2.error, KeyError, TypeError, ValueError) as error:
    st.error(f"The operation could not be rendered: {error}")
    st.stop()

st.markdown(
    f"""
    <div class="lab-technique">
        <strong>{selected_category} / {operation.name}</strong>
        <span>{operation.summary}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

original_column, result_column = st.columns(2, gap="large")
with original_column, st.container(border=True):
    st.subheader("Original")
    st.image(source_image, width="stretch")
    st.caption(f"RGB input · {image_dimensions(source_image)}")

with result_column, st.container(border=True):
    st.subheader("Result")
    st.image(result.image, width="stretch", clamp=True)
    output_kind = "Grayscale" if result.image.ndim == 2 else "RGB"
    st.caption(f"{output_kind} output · {image_dimensions(result.image)}")
    st.download_button(
        "Download result as PNG",
        data=image_as_png(result.image),
        file_name=f"{input_name}-{operation.key}.png",
        mime="image/png",
        width="stretch",
    )

metric_items = [
    ("Input size", image_dimensions(source_image)),
    ("Output size", image_dimensions(result.image)),
    ("Mean brightness", f"{mean_luminance(result.image):.1f} / 255"),
]
metric_items.extend(result.metrics.items())
metric_columns = st.columns(len(metric_items))
for column, (label, value) in zip(metric_columns, metric_items):
    column.metric(label, value)

learn_tab, code_tab, compare_tab, histogram_tab = st.tabs(
    ("Learn", "Copy the code", "Compare settings", "Histogram")
)

with learn_tab:
    st.markdown(f"### How {operation.name.lower()} works")
    st.write(operation.explanation)
    use_column, warning_column = st.columns(2, gap="large")
    with use_column:
        st.success(f"**Good for**  \n{operation.best_for}")
    with warning_column:
        st.warning(f"**Watch for**  \n{operation.watch_for}")

    st.markdown("#### Current parameters")
    parameter_rows = [
        {
            "Parameter": control.label,
            "Current value": format_parameter_value(parameters[control.key]),
            "What it controls": control.help,
        }
        for control in operation.controls
    ]
    st.dataframe(parameter_rows, hide_index=True, width="stretch")

with code_tab:
    st.markdown("### Reproduce this result")
    st.caption(
        "The snippet assumes `image_rgb` has already been loaded as an RGB uint8 NumPy array."
    )
    st.code(
        code_snippet(operation.key, parameters), language="python", line_numbers=True
    )
    st.markdown(
        """
        <div class="lab-callout">
            OpenCV often reads files in BGR order. If you used <code>cv2.imread</code>, convert once with
            <code>cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)</code> before using this snippet.
        </div>
        """,
        unsafe_allow_html=True,
    )

with compare_tab:
    comparison = sweep_variants(operation, parameters)
    if comparison is None:
        st.info(
            "This technique uses a categorical or automatic choice. Change its sidebar setting "
            "and compare it directly with the original above."
        )
    else:
        sweep_control, variants = comparison
        st.markdown(f"### Sweep: {sweep_control.label}")
        st.caption(
            "The middle view is your current setting; the neighboring views show a lower and "
            "higher value while every other parameter stays fixed."
        )
        comparison_columns = st.columns(len(variants), gap="medium")
        for column, (value_label, variant_parameters) in zip(
            comparison_columns, variants
        ):
            variant_result = apply_operation(
                operation.key,
                source_image,
                variant_parameters,
            )
            with column:
                st.image(variant_result.image, width="stretch", clamp=True)
                st.markdown(f"**{sweep_control.label}: {value_label}**")

with histogram_tab:
    st.markdown("### Luminance distribution")
    st.caption(
        "Each line shows the fraction of pixels at every brightness level from black (0) to white (255)."
    )
    histogram_data = pd.DataFrame(
        {
            "Original": luminance_histogram(source_image),
            "Result": luminance_histogram(result.image),
        }
    )
    histogram_data.index.name = "Brightness"
    st.line_chart(histogram_data, color=["#64748b", "#0f766e"])

st.divider()
st.caption(
    "Built from the experiments in this repository. Processing is deterministic so parameter comparisons stay meaningful."
)
