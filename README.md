# Nanofiber Fold Detection

MATLAB code for detecting fold-like boundary features in nanofiber microscopy images.

The script reads an input image, enhances contrast, detects edges, traces connected
boundaries, and marks candidate fold points using curvature and angle-change criteria.
Local data paths and sample-specific storage addresses have been removed. Provide the
image path at runtime.

## Usage

```matlab
results = nanofiber_fold_detection("path/to/input_image.tif");
```

The returned `results` structure includes the number of connected regions, total fold
count, average fold count per region, and per-region fold counts.

## Requirements

- MATLAB
- Image Processing Toolbox

## Notes

No raw microscopy images or local storage addresses are included in this code release.
