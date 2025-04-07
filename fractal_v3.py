import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import time
from pathlib import Path
import imageio

np.set_printoptions(threshold=np.inf)


def smoothstep(x, edge0, edge1):
    # Normalize x to [0, 1] between edge0 and edge1
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def palette(t):
    a = np.array([0.5, 0.5, 0.5])[np.newaxis, :]
    b = np.array([0.5, 0.5, 0.5])[np.newaxis, :]
    c = np.array([1.0, 1.0, 1.0])[np.newaxis, :]
    d = np.array([0.263, 0.416, 0.557])[np.newaxis, :]
    t = np.array(t)
    return a + b * np.cos(6.28318 * (c * t[..., np.newaxis] + d))


# Configuration for the GIF export
fps = 30  # Frames per second in the output GIF
duration = 5.0  # Total duration of the GIF in seconds
total_frames = int(fps * duration)
canvas_height, canvas_width = 500, 500  # Smaller size for faster rendering and smaller file

# Create an empty list to store frames
frames = []

# Simulate elapsed time without actually waiting
for frame_num in range(total_frames):
    # Calculate simulated elapsed time
    elapsed_time = frame_num / fps

    # Create a fresh canvas for this frame
    canvas = np.ones([canvas_height, canvas_width, 3], dtype=np.uint8) * 255

    # Initialize coordinate grids
    yy, xx = np.meshgrid(np.arange(canvas_height), np.arange(canvas_width), indexing='ij')
    yy, xx = 2.0 * (canvas_height - yy) / canvas_height, 2.0 * xx / canvas_width

    # Original coordinates for base coloring
    xx0, yy0 = xx - 1, yy - 1
    length0 = np.sqrt((xx0) ** 2 + (yy0) ** 2)

    # Reset color accumulator
    finalcolor = [0, 0, 0]

    # Apply transformations for fractal effect
    iterations = 3  # Change this to control fractal depth

    for i in range(iterations):
        # Apply fractal transformation
        xx = 3.0 * xx - 1.0
        yy = 3.0 * yy - 1.0

        # Take fractional part to create repetition
        xx = np.abs(xx) % 2.0 - 1.0
        yy = np.abs(yy) % 2.0 - 1.0

        # Calculate length for this iteration
        length = np.sqrt(xx ** 2 + yy ** 2)

        # Calculate function value for this iteration
        func_plot = np.abs(np.sin(length * np.exp(-length0) * 8.0 + elapsed_time) / 8.0)
        func_plot = 0.02 / (func_plot + 0.0000001)

        # Scale effect by iteration level (smaller contribution from deeper iterations)
        scale_factor = 1.0 / (i + 1)

        # Get colors for this iteration
        color_values = palette(length0 + elapsed_time * 0.4)
        blue = color_values[:, :, 2] * func_plot * scale_factor
        green = color_values[:, :, 1] * func_plot * scale_factor
        red = color_values[:, :, 0] * func_plot * scale_factor

        # Accumulate colors
        finalcolor[2] += blue
        finalcolor[1] += green
        finalcolor[0] += red

    # Apply accumulated colors to canvas
    canvas[:, :, 0] = np.clip(finalcolor[2] * 255, 0, 255).astype(np.uint8)
    canvas[:, :, 1] = np.clip(finalcolor[1] * 255, 0, 255).astype(np.uint8)
    canvas[:, :, 2] = np.clip(finalcolor[0] * 255, 0, 255).astype(np.uint8)

    # Convert from BGR (OpenCV default) to RGB for the GIF
    rgb_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    # Make sure the frame has the expected dimensions before appending
    if rgb_canvas.shape == (canvas_height, canvas_width, 3):
        frames.append(rgb_canvas)
    else:
        print(
            f"Warning: Frame {frame_num} has unexpected shape {rgb_canvas.shape}, expected {(canvas_height, canvas_width, 3)}")
        # Resize to expected dimensions if needed
        rgb_canvas = cv2.resize(rgb_canvas, (canvas_width, canvas_height))
        frames.append(rgb_canvas)

    # Optional: Display progress
    if frame_num % 10 == 0:
        print(f"Rendered frame {frame_num}/{total_frames} ({frame_num / total_frames * 100:.1f}%)")

    # Optional: Show the current frame
    cv2.namedWindow("Preview")
    cv2.imshow("Preview", canvas)
    ch = cv2.waitKey(1)
    if ch & 0xFF == ord('q'):
        break

# Close any open windows
cv2.destroyAllWindows()

# Debug check: verify all frames have identical shape
frame_shapes = [frame.shape for frame in frames]
if len(set(frame_shapes)) != 1:
    print("Warning: Not all frames have the same shape!")
    print("Shapes found:", set(frame_shapes))

    # Fix by resizing all frames to the expected dimensions
    print("Resizing all frames to match expected dimensions...")
    for i in range(len(frames)):
        if frames[i].shape != (canvas_height, canvas_width, 3):
            frames[i] = cv2.resize(frames[i], (canvas_width, canvas_height))

# Export the GIF
output_path = 'fractal_animation2.gif'
print(f"Saving GIF to {output_path}...")

# Save with imageio - convert frames to uint8 if needed
frames_uint8 = [np.asarray(frame, dtype=np.uint8) for frame in frames]
imageio.mimsave(output_path, frames_uint8, fps=fps)

print(f"GIF saved successfully: {output_path}")