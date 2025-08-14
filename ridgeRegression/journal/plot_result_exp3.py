# 分组柱状图
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Data
# languages = ['EN','FR','ZH']
# layers = ['1', '4', '12', '16', '24']
#
# # Corresponding PC values for each language in each layer
# pc_values = {
#     'EN': [0.0585, 0.1168, 0.1060, 0.1067, 0.1036],
#     'FR': [0.1450, 0.1836, 0.1747, 0.1790, 0.1806],
#     'ZH': [0.1879, 0.1492, 0.1528, 0.1202, 0.1832],
# }
#
# # Set positions of bars on X axis
# num_languages = len(languages)
# bar_width = 0.15  # Set width for each bar
# x_indexes = np.arange(num_languages) * 1.5  # Adding spacing between groups
#
# # Improved color palette for the five layers
# layer_colors = ['#FFB6C1', '#ADD8E6', '#98FB98', '#FFD700', '#9370DB']
#
# # Plot
# fig, ax = plt.subplots(figsize=(10, 6))
#
# # Plot each layer as separate bars for each language
# for i, layer in enumerate(layers):
#     layer_pc_values = [pc_values[lang][i] for lang in languages]
#     ax.bar(x_indexes + i * bar_width, layer_pc_values, color=layer_colors[i], width=bar_width, label=f'Layer {layer}')
#
# # Customizing the plot
# ax.set_xlabel('Languages', fontweight='bold')
# ax.set_ylabel('PC Values', fontweight='bold')
# ax.set_title('PC Values by Layer for Different Languages')
#
# # Set xticks in the middle of group bars
# ax.set_xticks(x_indexes + 2 * bar_width)
# ax.set_xticklabels(languages)
#
# # Add a legend for the layers
# ax.legend(title='Layers')
#
# # Add space between groups of languages
# ax.margins(x=0.2)
#
# # Show plot with tight layout
# plt.tight_layout()
# plt.show()
#
import matplotlib.pyplot as plt
import numpy as np

# Data
languages = ['EN', 'FR', 'ZH']
layers = ['1', '4', '12', '16', '24']

# Corresponding PC values for each language in each layer
pc_values = {
    'EN': [0.0585, 0.1168, 0.1060, 0.1067, 0.1036],
    'FR': [0.1450, 0.1836, 0.1747, 0.1790, 0.1806],
    'ZH': [0.1879, 0.1492, 0.1528, 0.1202, 0.1832],
}

# Improved color palette for the three languages
line_colors = ['#FF6347', '#4682B4', '#32CD32']  # EN - Red, FR - Blue, ZH - Green
markers = ['o', 's', 'D']  # Different markers for each language

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each language's line graph
for i, lang in enumerate(languages):
    ax.plot(layers, pc_values[lang], marker=markers[i], color=line_colors[i], label=lang,
            linestyle='--', linewidth=2.5, markersize=8)  # Line thickness, marker size increased

# Customizing the plot
ax.set_xlabel('Layers', fontweight='bold')
ax.set_ylabel('PC Values', fontweight='bold')
ax.set_title('PC Values by Layer for Different Languages', fontweight='bold', fontsize=14)

# Add grid for better readability
ax.grid(True, which='both', linestyle='--', linewidth=0.7)

# Customize tick parameters for visibility
ax.tick_params(axis='both', which='major', labelsize=12)

# Add a legend with better positioning and larger font size
ax.legend(title='Languages', loc='lower right', bbox_to_anchor=(1, 0.5), fontsize=10, title_fontsize=12, frameon=True)

# Add tight layout for better padding and margin control
plt.tight_layout()

# Show plot
plt.show()