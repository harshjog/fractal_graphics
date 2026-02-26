import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import time
from pathlib import Path
import imageio
from PIL import Image, ImageSequence
import pygifsicle
import pyopencl as cl

# Set up OpenCL for AMD GPU
def setup_opencl():
    """Initialize OpenCL context for AMD GPU"""
    platforms = cl.get_platforms()
    
    # Find AMD platform
    amd_platform = None
    for platform in platforms:
        if 'AMD' in platform.name or 'amd' in platform.name.lower():
            amd_platform = platform
            break
    
    if amd_platform is None:
        # Fallback to first platform if AMD not found
        amd_platform = platforms[0]
    
    print(f"Using platform: {amd_platform.name}")
    
    # Get GPU devices
    devices = amd_platform.get_devices(cl.device_type.GPU)
    if not devices:
        devices = amd_platform.get_devices()
    
    device = devices[0]
    print(f"Using device: {device.name}")
    
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    
    return ctx, queue, device


# OpenCL Kernel for fractal computation
FRACTAL_KERNEL = """
float3 palette(float t) {
    float a[] = {0.5f, 0.5f, 0.5f};
    float b[] = {0.5f, 0.5f, 0.5f};
    float c[] = {1.0f, 1.0f, 1.0f};
    float d[] = {0.263f, 0.416f, 0.557f};
    
    float angle = 6.28318f * (c[0] * t + d[0]);
    float3 result;
    result.x = a[0] + b[0] * cos(angle);
    result.y = a[1] + b[1] * cos(6.28318f * (c[1] * t + d[1]));
    result.z = a[2] + b[2] * cos(6.28318f * (c[2] * t + d[2]));
    return result;
}

__kernel void compute_fractal(
    __global float *xx,
    __global float *yy,
    __global float *xx0,
    __global float *yy0,
    __global float *length0,
    float elapsed_time,
    int iterations,
    __global float *out_red,
    __global float *out_green,
    __global float *out_blue
) {
    int idx = get_global_id(0);
    
    float x = xx[idx];
    float y = yy[idx];
    float x0 = xx0[idx];
    float y0 = yy0[idx];
    float len0 = length0[idx];
    
    float finalcolor_r = 0.0f;
    float finalcolor_g = 0.0f;
    float finalcolor_b = 0.0f;
    
    for (int i = 0; i < iterations; i++) {
        // Apply fractal transformation
        x = 4.0f * x - 2.0f;
        y = 4.0f * y - 2.0f;
        
        // Take fractional part to create repetition
        x = fabs(x) - 2.0f * floor(fabs(x) / 2.0f) - 1.0f;
        y = fabs(y) - 2.0f * floor(fabs(y) / 2.0f) - 1.0f;
        
        // Calculate length for this iteration
        float length = sqrt(x * x + y * y);
        
        // Calculate function value for this iteration
        float func_plot = fabs(sin(length * exp(-len0) * 8.0f + elapsed_time*0.01) / 8.0f);
        func_plot = 0.01f / (func_plot + 0.0000001f);
        
        // Scale effect by iteration level
        float scale_factor = 1.0f / (i + 1);
        
        // Get colors for this iteration
        float3 color_values = palette(len0 + elapsed_time * 0.4f);
        float blue = color_values.z * func_plot * scale_factor;
        float green = color_values.y * func_plot * scale_factor;
        float red = color_values.x * func_plot * scale_factor;
        
        // Accumulate colors
        finalcolor_b += blue;
        finalcolor_g += green;
        finalcolor_r += red;
    }
    
    out_red[idx] = finalcolor_r;
    out_green[idx] = finalcolor_g;
    out_blue[idx] = finalcolor_b;
}
"""

np.set_printoptions(threshold=np.inf)

# Initialize OpenCL
ctx, queue, device = setup_opencl()
program = cl.Program(ctx, FRACTAL_KERNEL).build()

# Configuration for the GIF export
fps = 24  # Frames per second in the output GIF
duration = 10  # Total duration of the GIF in seconds
total_frames = int(fps * duration)
canvas_height, canvas_width = 1000, 1000  # Dimensions

# Color reduction parameters
num_colors = 64  # Reduce colors to this many (lower = smaller file size)

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

    # Flatten arrays for GPU processing
    xx_flat = xx.astype(np.float32).flatten()
    yy_flat = yy.astype(np.float32).flatten()
    xx0_flat = xx0.astype(np.float32).flatten()
    yy0_flat = yy0.astype(np.float32).flatten()
    length0_flat = length0.astype(np.float32).flatten()
    
    # Apply transformations for fractal effect
    iterations = 3  # Change this to control fractal depth
    
    # Allocate GPU memory
    xx_gpu = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=xx_flat)
    yy_gpu = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=yy_flat)
    xx0_gpu = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=xx0_flat)
    yy0_gpu = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=yy0_flat)
    length0_gpu = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=length0_flat)
    
    red_gpu = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=xx_flat.nbytes)
    green_gpu = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=xx_flat.nbytes)
    blue_gpu = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=xx_flat.nbytes)
    
    # Execute kernel
    program.compute_fractal(
        queue, xx_flat.shape, None,
        xx_gpu, yy_gpu, xx0_gpu, yy0_gpu, length0_gpu,
        np.float32(elapsed_time), np.int32(iterations),
        red_gpu, green_gpu, blue_gpu
    )
    
    # Read results from GPU
    finalcolor_r = np.empty_like(xx_flat)
    finalcolor_g = np.empty_like(xx_flat)
    finalcolor_b = np.empty_like(xx_flat)
    
    cl.enqueue_copy(queue, finalcolor_r, red_gpu)
    cl.enqueue_copy(queue, finalcolor_g, green_gpu)
    cl.enqueue_copy(queue, finalcolor_b, blue_gpu)
    queue.finish()
    
    # Reshape back to 2D
    finalcolor_r = finalcolor_r.reshape(canvas_height, canvas_width)
    finalcolor_g = finalcolor_g.reshape(canvas_height, canvas_width)
    finalcolor_b = finalcolor_b.reshape(canvas_height, canvas_width)
    
    # Apply accumulated colors to canvas
    canvas[:, :, 0] = np.clip(finalcolor_b * 255, 0, 255).astype(np.uint8)
    canvas[:, :, 1] = np.clip(finalcolor_g * 255, 0, 255).astype(np.uint8)
    canvas[:, :, 2] = np.clip(finalcolor_r * 255, 0, 255).astype(np.uint8)

    # Convert from BGR to RGB for the GIF
    rgb_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    # Append frame to our list
    frames.append(rgb_canvas)

    # Display progress
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

# Save the raw GIF first (we'll optimize it after)
temp_output = 'temp_fractal.gif'
output_path = 'fractal_animation_optimized5_v2.gif'

print(f"Converting frames to PIL images...")
frames_pil = [Image.fromarray(frame) for frame in frames]

print(f"Saving initial GIF to {temp_output}...")
# # Save with PIL for better color quantization control
# frames_pil[0].save(
#     temp_output,
#     save_all=True,
#     append_images=frames_pil[1:],
#     optimize=True,
#     duration=int(1000 / fps),  # Duration in ms
#     loop=0,  # 0 means loop forever
#     colors=num_colors  # Reduce to specified number of colors
# )

# print(f"Saving optimized GIF to {output_path}...")
# frames_pil[0].save(
#     output_path,
#     save_all=True,
#     append_images=frames_pil[1:],
#     optimize=True,
#     duration=int(1000/fps),
#     loop=0,
#     colors=num_colors,
#     disposal=2  # Only update regions that change between frames
# )

# Clean up temporary file
Path(temp_output).unlink(missing_ok=True)
print(f"GIF saved to: {output_path}")