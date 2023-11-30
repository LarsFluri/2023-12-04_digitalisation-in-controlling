
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
# Define your colors
#%%
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

# hex colors
hex_colors = [rgb2hex(color) for color in rgb_colors]

# Create a colormap
fhnw_colourmap = LinearSegmentedColormap.from_list("custom_colormap", rgb_colors)




# palette_3 = [(252/255, 230/255, 14/255),  # Original Color
#           (227/255, 207/255, 12/255),  # Slightly Darker
#           (202/255, 184/255, 10/255)]
# palette_4 = [(252/255, 230/255, 14/255),  # Original Color
#           (227/255, 207/255, 12/255),  # Slightly Darker
#           (202/255, 184/255, 10/255),  # Even Darker
#           (252/255, 235/255, 64/255)]
# palette_5 =




#%%


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
# rgb_colors = [(252/255, 230/255, 14/255),  # Original Color
#           (227/255, 207/255, 12/255),  # Slightly Darker
# #           (202/255, 184/255, 10/255),  # Even Darker
# #           (252/255, 235/255, 64/255),  # Lighter
# #           (252/255, 240/255, 114/255),
# #           (153, 102, 204)]rgb_colors = [(252/255, 230/255, 14/255),  # Original Color
#           (227/255, 207/255, 12/255),  # Slightly Darker
#           (202/255, 184/255, 10/255),  # Even Darker
#           (252/255, 235/255, 64/255),  # Lighter
#           (252/255, 240/255, 114/255),
#           (153, 102, 204)]
