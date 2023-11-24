
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
# Define your colors
#%%
fhnw_colour = (252/255, 230/255, 14/255)


hex_colors = [
    "#fce60e",  # Original Color
    "#e3cf0c",  # Slightly Darker
    "#cab80a",  # Even Darker
    "#fceb40",  # Lighter
    "#fcf072",  # Lighter
    "#9966cc"   # RGB Color
]

rgb_colors = [(252/255, 230/255, 14/255),  # Original Color
          (227/255, 207/255, 12/255),  # Slightly Darker
          (202/255, 184/255, 10/255),  # Even Darker
          (252/255, 235/255, 64/255),  # Lighter
          (252/255, 240/255, 114/255),
          (153, 102, 204)]

# palette_3 = [(252/255, 230/255, 14/255),  # Original Color
#           (227/255, 207/255, 12/255),  # Slightly Darker
#           (202/255, 184/255, 10/255)]
# palette_4 = [(252/255, 230/255, 14/255),  # Original Color
#           (227/255, 207/255, 12/255),  # Slightly Darker
#           (202/255, 184/255, 10/255),  # Even Darker
#           (252/255, 235/255, 64/255)]
# palette_5 =




#%%


# Create a colormap
fhnw_colourmap = LinearSegmentedColormap.from_list("custom_colormap", rgb_colors)

#%%
# import colorsys
# def generate_color_palette(base_color, n_colors):
#     """
#     Generate a color palette based on the base color.
#     Adjusts the lightness to create different shades of the base color.

#     :param base_color: Tuple (R, G, B) as values in the range [0, 255]
#     :param n_colors: Number of colors to generate in the palette
#     :return: List of RGB tuples
#     """
#     # Convert the base color from RGB to HSL
#     base_hsl = colorsys.rgb_to_hls(base_color[0]/255, base_color[1]/255, base_color[2]/255)

#     # Create variations by adjusting the lightness
#     variations = [(base_hsl[0], base_hsl[1] + (i / (n_colors - 1) - 0.5) * 0.4, base_hsl[2]) for i in range(n_colors)]

#     # Convert back to RGB
#     rgb_colors = [colorsys.hls_to_rgb(*hsl) for hsl in variations]

#     # Adjusting the RGB values to the range [0, 255]
#     rgb_colors = [(round(r*255), round(g*255), round(b*255)) for r, g, b in rgb_colors]

#     return rgb_colors

# palette = generate_color_palette(fhnw_colour, 5)
# palette