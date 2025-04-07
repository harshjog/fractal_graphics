# Fractal Animation Generator

A Python tool for creating mesmerizing animated fractal patterns and exporting them as optimized GIFs.

## Overview

This project generates dynamic fractal patterns using coordinate transformations and mathematical functions. The animations feature self-similar patterns that evolve over time, creating visually interesting effects. The tool supports real-time preview and exports high-quality, size-optimized GIF animations.

## Features

- Real-time fractal visualization
- Customizable animation parameters (FPS, duration, complexity)
- GIF export with size optimization techniques
- Adjustable fractal iterations for different complexity levels
- Color palette based on smooth cosine transitions

## Requirements

- Python 3.6+
- Required Python packages:
  - numpy
  - opencv-python (cv2)
  - matplotlib
  - imageio
  - Pillow (PIL)
  - pygifsicle

- External dependencies:
  - gifsicle (for advanced GIF optimization)

## Installation

1. Clone this repository or download the source files

2. Install the required Python packages:
   ```bash
   pip install numpy opencv-python matplotlib imageio Pillow pygifsicle
   ```

3. Install gifsicle (for optimized GIFs):
   - Windows: Download from the [gifsicle website](https://www.lcdf.org/gifsicle/)
   - macOS: `brew install gifsicle`
   - Linux: `apt install gifsicle` or equivalent for your distribution

## Usage

### Basic Usage

Run the script to generate and preview the fractal animation in real-time:

```bash
python fractal_animation.py
```

Press 'q' to exit the preview.

### Configuration Options

Adjust these parameters in the script to customize your animation:

- `fps`: Frames per second in the output GIF (default: 30)
- `duration`: Total animation duration in seconds (default: 5.0)
- `canvas_height` / `canvas_width`: Dimensions of the animation (default: 500×500)
- `iterations`: Number of fractal iterations - higher values create more detail (default: 3)
- `num_colors`: Color palette size for optimization (default: 64, lower = smaller file)

### Customizing the Visualization

- Modify the `palette()` function to change the color scheme
- Adjust the mathematical function in the loop to create different patterns:
  ```python
  # Try different functions here for varied effects
  func_plot = np.abs(np.sin(length * 8.0 + elapsed_time)/ 8.0)
  ```

## Optimization Techniques

This tool employs several techniques to reduce GIF file sizes:

1. **Color Palette Reduction**: Limits the number of colors per frame
2. **Frame Difference Encoding**: Only stores changes between consecutive frames
3. **Gifsicle Optimization**: Advanced compression and redundancy removal
4. **Dithering Control**: Optional quality vs. size tradeoff

## Examples

Experiment with different parameters to create unique visualizations:

- Change the `iterations` value to control fractal complexity
- Modify the time-based functions for different animation speeds
- Adjust the transformation equations to create different fractal types

## Troubleshooting

- **Memory Issues**: If you encounter memory problems with large animations, try:
  - Reducing the canvas dimensions
  - Shortening the animation duration
  - Exporting frames in batches

- **Gifsicle Not Found**: Ensure gifsicle is properly installed and in your system PATH

## License

This project is open source and available under the GNU License.

## Acknowledgements

This project was created using mathematical concepts from fractal geometry and leverages several open-source libraries for image processing and animation.