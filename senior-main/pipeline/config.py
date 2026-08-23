"""Pipeline-wide configuration constants."""

# Stage 7 volume calibration
# The ArUco marker is identified as the "obj" cluster (non-box).
# REFERENCE_REAL_SIZE_CM is the real-world linear size of the ArUco marker cube.
REFERENCE_REAL_SIZE_CM = 14.0     # real linear size of reference in cm

# Stage 1 frame limits
DEFAULT_MAX_FRAMES_MPS = 6

# Model weights
VGGT_MODEL_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"

# Image extensions accepted by the input loader
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp", ".heic", ".heif")
