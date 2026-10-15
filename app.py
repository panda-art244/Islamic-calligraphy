
import json
from datetime import datetime, timezone

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Coin & Inscription Explorer",
    layout="wide",
)

st.title("🪙 Coin & Inscription Explorer")

st.caption(
    "Computer vision for medieval Islamic coins "
    "and cultural heritage research"
)

st.markdown(
    """
    Upload a photograph of a coin to explore its visual features.

    This prototype can enhance an image, detect edges, and attempt
    to locate a circular coin outline. It does not identify coins
    or automatically read and translate inscriptions.
    """
)


# --------------------------------------------------
# Image loading and export helpers
# --------------------------------------------------

def load_image(uploaded_file):
    """Load an image, correct its orientation, and resize if needed."""
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((2000, 2000))
    return np.array(image)


def encode_png(image):
    """Encode a grayscale or RGB image as PNG bytes."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    success, buffer = cv2.imencode(".png", image)

    if not success:
        raise ValueError("Could not encode image as PNG.")

    return buffer.tobytes()


# --------------------------------------------------
# Coin detection
# --------------------------------------------------

def detect_coin(rgb_image):
    """
    Attempt to detect a circular object using Hough Circle Transform.
    The largest detected circle is treated as a candidate coin outline.
    """
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    height, width = gray.shape
    min_dimension = min(height, width)

    min_radius = max(10, min_dimension // 10)
    max_radius = max(min_radius + 1, min_dimension // 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, min_dimension // 3),
        param1=100,
        param2=35,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    marked_image = rgb_image.copy()

    if circles is None:
        return marked_image, None

    circles = np.round(circles[0]).astype(int)

    x, y, radius = max(circles, key=lambda circle: circle[2])

    # Draw the detected candidate circle.
    cv2.circle(marked_image, (x, y), radius, (0, 255, 0), 4)
    cv2.circle(marked_image, (x, y), 4, (255, 0, 0), -1)

    return marked_image, {
        "center_x": int(x),
        "center_y": int(y),
        "radius": int(radius),
    }


# --------------------------------------------------
# Image processing
# --------------------------------------------------

def compare_processing_methods(rgb_image, contrast, edge_threshold):
    """Create different visual representations of the same image."""
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    # Method 1: improve local contrast.
    clahe = cv2.createCLAHE(
        clipLimit=float(contrast),
        tileGridSize=(8, 8),
    )
    enhanced = clahe.apply(gray)

    # Method 2: adaptive thresholding.
    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )

    # Method 3: sharpen the image.
    blurred = cv2.GaussianBlur(gray, (0, 0), 2)
    sharpened = cv2.addWeighted(
        gray, 1.8,
        blurred, -0.8,
        0,
    )

    # Method 4: detect edges.
    edges = cv2.Canny(
        enhanced,
        int(edge_threshold),
        int(edge_threshold * 2),
    )

    # Calculate simple image statistics.
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    image_area = int(gray.shape[0] * gray.shape[1])
    edge_pixel_count = int(np.count_nonzero(edges))

    statistics = {
        "width": int(gray.shape[1]),
        "height": int(gray.shape[0]),
        "mean_brightness": round(float(gray.mean()), 2),
        "edge_pixel_count": edge_pixel_count,
        "edge_pixel_percentage": round(
            edge_pixel_count / image_area * 100, 3
        ),
        "contour_count": len(contours),
    }

    processed_images = {
        "CLAHE contrast": enhanced,
        "Adaptive threshold": adaptive,
        "Sharpening": sharpened,
        "Edge detection": edges,
    }

    return processed_images, statistics


# --------------------------------------------------
# Upload and settings
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a coin photograph",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is None:
    st.info("Upload a photograph to start the analysis.")
    st.stop()

try:
    original = load_image(uploaded_file)
except Exception as exc:
    st.error(f"Could not read this image: {exc}")
    st.stop()


with st.sidebar:
    st.header("Analysis settings")

    contrast = st.slider(
        "Local contrast enhancement",
        min_value=1.0,
        max_value=5.0,
        value=2.0,
        step=0.5,
        help="Higher values may make faint details more visible.",
    )

    edge_threshold = st.slider(
        "Edge detection threshold",
        min_value=10,
        max_value=200,
        value=80,
        step=5,
        help="Lower values detect more subtle edges.",
    )

    st.caption(
        "Change the settings to compare the visual results."
    )


# --------------------------------------------------
# Run the analysis
# --------------------------------------------------

try:
    detected_image, coin_circle = detect_coin(original)

    processed_images, statistics = compare_processing_methods(
        original,
        contrast,
        edge_threshold,
    )
except Exception as exc:
    st.error(f"Image processing failed: {exc}")
    st.stop()


# --------------------------------------------------
# Section 1: original photograph
# --------------------------------------------------

st.header("1. Original photograph")

st.image(
    original,
    caption="Uploaded coin photograph",
    use_container_width=True,
)


# --------------------------------------------------
# Section 2: coin detection
# --------------------------------------------------

st.header("2. Automatic circular-object detection")

if coin_circle is not None:
    st.success("A possible circular outline was detected.")

    st.image(
        detected_image,
        caption="Green circle: detected candidate outline",
        use_container_width=True,
    )

    st.write(
        f'Centre coordinates: '
        f'({coin_circle["center_x"]}, '
        f'{coin_circle["center_y"]}) pixels'
    )

    st.write(
        f'Estimated radius: {coin_circle["radius"]} pixels'
    )
else:
    st.warning(
        "No circular outline was detected. "
        "Try a photograph with a clearly visible coin edge."
    )

    st.image(
        detected_image,
        caption="No candidate circle found",
        use_container_width=True,
    )

st.caption(
    "Circle detection is approximate. The algorithm may miss a worn "
    "coin or detect another circular object in the photograph."
)


# --------------------------------------------------
# Section 3: compare image processing methods
# --------------------------------------------------

st.header("3. Compare image processing methods")

st.markdown(
    """
    Each method changes the visual representation of the same image.
    Compare them to see which makes surface details easier to inspect.
    """
)

columns = st.columns(2)

for index, (method_name, processed_image) in enumerate(
    processed_images.items()
):
    with columns[index % 2]:
        st.subheader(method_name)

        st.image(
            processed_image,
            caption=method_name,
            use_container_width=True,
        )


# --------------------------------------------------
# Section 4: statistics
# --------------------------------------------------

st.header("4. Image statistics")

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric("Width", f'{statistics["width"]} px')
metric2.metric("Height", f'{statistics["height"]} px')
metric3.metric(
    "Mean brightness",
    statistics["mean_brightness"],
)
metric4.metric(
    "Edge pixels",
    f'{statistics["edge_pixel_percentage"]}%',
)

st.write(
    f'Number of detected contours: '
    f'{statistics["contour_count"]}'
)

st.caption(
    "Contours and edge pixels are image-processing measurements. "
    "They do not directly correspond to letters, symbols, or "
    "archaeological features."
)


# --------------------------------------------------
# Section 5: export
# --------------------------------------------------

report = {
    "project": "Coin & Inscription Explorer",
    "analysis_time_utc": datetime.now(timezone.utc).isoformat(),
    "input_filename": uploaded_file.name,
    "image_dimensions": {
        "width": statistics["width"],
        "height": statistics["height"],
    },
    "parameters": {
        "contrast_clip_limit": contrast,
        "edge_threshold": edge_threshold,
    },
    "statistics": statistics,
    "coin_detection": coin_circle,
    "methods": list(processed_images.keys()),
    "limitations": [
        "Coin identification is not implemented.",
        "Inscription transcription and translation are not implemented.",
        "Circular-object detection can fail on worn or irregular coins.",
        "Results depend on lighting, image quality, and surface wear.",
    ],
}

st.header("5. Export results")

download1, download2, download3 = st.columns(3)

with download1:
    st.download_button(
        "Download JSON report",
        data=json.dumps(report, ensure_ascii=False, indent=2),
        file_name="coin_analysis.json",
        mime="application/json",
        use_container_width=True,
    )

with download2:
    st.download_button(
        "Download enhanced image",
        data=encode_png(processed_images["CLAHE contrast"]),
        file_name="coin_enhanced.png",
        mime="image/png",
        use_container_width=True,
    )

with download3:
    st.download_button(
        "Download edge map",
        data=encode_png(processed_images["Edge detection"]),
        file_name="coin_edges.png",
        mime="image/png",
        use_container_width=True,
    )

with st.expander("View complete analysis metadata"):
    st.json(report)




