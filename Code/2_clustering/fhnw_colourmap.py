
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import rgb2hex


# Define the color scheme
fhnw_colour = (252/255, 230/255, 14/255)

#rgb colours
rgb_colors = [
    (0.9882352941176471, 0.9019607843137255, 0.054901960784313725), # Original Yellow
    (0.7905882352941177, 0.7215686274509805, 0.04392156862745098),
    (0.5929411764705882, 0.5411764705882353, 0.03294117647058823),
    (0.39529411764705885, 0.36078431372549025, 0.02196078431372549),
    (0.19764705882352937, 0.18039215686274507, 0.010980392156862742),
    (0.0, 0.0, 0.0) # Black
]

#hex colors
hex_colors = [rgb2hex(color) for color in rgb_colors]

# Create a colormap
fhnw_colourmap = LinearSegmentedColormap.from_list("custom_colormap", rgb_colors)

