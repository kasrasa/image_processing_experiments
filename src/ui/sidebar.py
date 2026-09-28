"""Sidebar controls for choosing the image, technique, and parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image

from src.image_utils import create_sample_image, load_uploaded_image, resize_to_limit
from src.operation_registry import (
    CATEGORY_ORDER,
    ControlSpec,
    OperationSpec,
    operations_for_category,
)


@dataclass(frozen=True)
class ExplorerSelection:
    """All user selections needed to render one explorer result."""

    source_image: np.ndarray
    input_name: str
    category: str
    operation: OperationSpec
    parameters: dict[str, Any]


@st.cache_data(show_spinner=False)
def _sample_image() -> np.ndarray:
    return create_sample_image()


def render_control(control: ControlSpec, operation_key: str) -> Any:
    """Render the Streamlit widget described by a control specification."""

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


def _render_image_source() -> tuple[np.ndarray, str]:
    st.sidebar.caption(
        "Your uploaded image is processed only in the running app session."
    )
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
            help=(
                "Large images are resized to a maximum side length of 1,400 pixels "
                "for responsiveness."
            ),
        )
        if uploaded_file is None:
            st.sidebar.info(
                "Upload an image when ready. The test image is shown in the meantime."
            )
            source_image = _sample_image().copy()
        else:
            try:
                source_image = load_uploaded_image(uploaded_file)
                source_image, was_resized = resize_to_limit(source_image)
                input_name = uploaded_file.name.rsplit(".", 1)[0]
            except (OSError, ValueError, Image.DecompressionBombError) as error:
                st.sidebar.error(f"That image could not be read: {error}")
                source_image = _sample_image().copy()
    else:
        source_image = _sample_image().copy()

    if was_resized:
        st.sidebar.warning("The preview was resized to keep interactions fast.")
    return source_image, input_name


def render_sidebar() -> ExplorerSelection:
    """Render all experiment controls and return their current values."""

    source_image, input_name = _render_image_source()

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
    return ExplorerSelection(
        source_image=source_image,
        input_name=input_name,
        category=selected_category,
        operation=operation,
        parameters=parameters,
    )
