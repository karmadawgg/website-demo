
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
from matplotlib.patches import FancyArrowPatch

# --------------------------------------------------
# Create figure
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(18, 7))

ax.set_xlim(0, 18)
ax.set_ylim(0, 7)
ax.axis("off")

fig.patch.set_facecolor("#252525")
ax.set_facecolor("#252525")

# --------------------------------------------------
# Helper function to draw text
# --------------------------------------------------
def draw_text(x, y, text, size=10, color="black",
              weight="normal", ha="center"):
    ax.text(
        x, y, text,
        fontsize=size,
        color=color,
        fontweight=weight,
        ha=ha,
        va="center"
    )


# --------------------------------------------------
# Helper function to draw arrows
# --------------------------------------------------
def draw_arrow(x1, y1, x2, y2):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=1.5,
        color="black"
    )
    ax.add_patch(arrow)


# --------------------------------------------------
# Helper function to draw stacked feature maps
# --------------------------------------------------
def draw_feature_maps(x, y, count=4,
                      color="#78B5F9",
                      width=1.1, height=1.1):

    for i in range(count):
        offset = i * 0.12

        rect = Rectangle(
            (x + offset, y + offset),
            width, height,
            facecolor=color,
            edgecolor="#0756B8",
            linewidth=1.3
        )

        ax.add_patch(rect)


# --------------------------------------------------
# Helper function to draw pooling grid
# --------------------------------------------------
def draw_pooling(x, y, size=1.0):
    rect = Rectangle(
        (x, y), size, size,
        facecolor="#FFB347",
        edgecolor="#D97706",
        linewidth=1.5
    )
    ax.add_patch(rect)

    # Divide into 4 sections
    ax.plot(
        [x + size / 2, x + size / 2],
        [y, y + size],
        color="white",
        linewidth=1.5
    )

    ax.plot(
        [x, x + size],
        [y + size / 2, y + size / 2],
        color="white",
        linewidth=1.5
    )


# --------------------------------------------------
# Helper function to draw neurons
# --------------------------------------------------
def draw_neurons(x, y, label, neurons):

    # Layer container
    box = FancyBboxPatch(
        (x, y), 0.6, 2.0,
        boxstyle="round,pad=0.05",
        facecolor="#F4F8FF",
        edgecolor="#1683E8",
        linewidth=1.5
    )
    ax.add_patch(box)

    # Neurons
    positions = [y + 1.65, y + 1.05, y + 0.25]

    for pos in positions:
        neuron = Circle(
            (x + 0.3, pos),
            0.12,
            facecolor="#3E7CCB",
            edgecolor="#1557B0",
            linewidth=1
        )
        ax.add_patch(neuron)

    # Dots represent omitted neurons
    draw_text(x + 0.3, y + 0.65, "•••", 12)

    draw_text(x + 0.3, y + 2.4, label, 9)
    draw_text(x + 0.3, y - 0.25, neurons, 9)


# --------------------------------------------------
# Title
# --------------------------------------------------
draw_text(
    9, 6.5,
    "Typical Structure of a Convolutional Neural Network (CNN)",
    17, "white", "bold"
)

draw_text(
    9, 6.05,
    "Image Input → Convolution → ReLU → Pooling → "
    "Flatten → Dense Layers → Output",
    11, "white"
)


# --------------------------------------------------
# Input image
# --------------------------------------------------
draw_text(0.75, 5.25, "Input Image", 11, "white", "bold")

# Image placeholder
input_box = Rectangle(
    (0.2, 3.6), 1.2, 1.2,
    facecolor="#D8E8F5",
    edgecolor="white",
    linewidth=1.5
)
ax.add_patch(input_box)

# Draw simple cat-like face
draw_text(0.8, 4.2, "🐱", 35)

draw_text(0.8, 3.25, "224 × 224 × 3", 10, "white")
draw_text(0.8, 3.0, "(RGB)", 9, "white")


# --------------------------------------------------
# First convolution + ReLU block
# --------------------------------------------------
block1 = FancyBboxPatch(
    (1.7, 2.9), 3.4, 2.8,
    boxstyle="round,pad=0.15",
    facecolor="white",
    edgecolor="#1683E8",
    linestyle="--",
    linewidth=1.5
)
ax.add_patch(block1)

draw_text(3.4, 5.4, "Convolution + ReLU", 12, "black", "bold")

draw_text(2.6, 4.95, "Conv 3 × 3", 10)
draw_text(2.6, 4.7, "64 Filters", 9)

draw_feature_maps(2.0, 3.6)

draw_text(3.0, 4.95, "", 9)

# ReLU feature maps
draw_text(4.3, 4.95, "ReLU", 10)
draw_feature_maps(
    3.9, 3.7,
    count=3,
    color="#81D49A"
)

draw_text(3.4, 3.1, "224 × 224 × 64", 10)
draw_text(3.4, 2.75, "Feature Maps", 9)


# --------------------------------------------------
# First max pooling
# --------------------------------------------------
draw_text(5.8, 5.0, "Max Pooling", 10)

draw_pooling(5.4, 3.8, 0.9)

draw_text(5.85, 3.4, "112 × 112 × 64", 9)
draw_text(5.85, 3.15, "Feature Maps", 8)


# --------------------------------------------------
# Second convolution + ReLU block
# --------------------------------------------------
block2 = FancyBboxPatch(
    (6.6, 2.9), 3.4, 2.8,
    boxstyle="round,pad=0.15",
    facecolor="white",
    edgecolor="#1683E8",
    linestyle="--",
    linewidth=1.5
)
ax.add_patch(block2)

draw_text(8.3, 5.4, "Convolution + ReLU", 12, "black", "bold")

draw_text(7.5, 4.95, "Conv 3 × 3", 10)
draw_text(7.5, 4.7, "128 Filters", 9)

draw_feature_maps(
    6.9, 3.6,
    color="#78B5F9"
)

draw_text(9.2, 4.95, "ReLU", 10)

draw_feature_maps(
    8.9, 3.7,
    count=3,
    color="#81D49A"
)

draw_text(8.3, 3.1, "112 × 112 × 128", 10)
draw_text(8.3, 2.75, "Feature Maps", 9)


# --------------------------------------------------
# Second max pooling
# --------------------------------------------------
draw_text(10.7, 5.0, "Max Pooling", 10)

draw_pooling(10.3, 3.8, 0.9)

draw_text(10.75, 3.4, "56 × 56 × 128", 9)
draw_text(10.75, 3.15, "Feature Maps", 8)


# --------------------------------------------------
# Flatten layer
# --------------------------------------------------
draw_text(12.0, 5.4, "Flatten", 11, "white", "bold")

flatten = Rectangle(
    (11.8, 3.0), 0.5, 2.2,
    facecolor="#D0D5DB",
    edgecolor="#737B85",
    linewidth=1.5
)
ax.add_patch(flatten)

# Lines inside flatten vector
for i in range(1, 8):
    y = 3.0 + i * 2.2 / 8
    ax.plot(
        [11.8, 12.3], [y, y],
        color="#737B85",
        linewidth=0.7
    )

draw_text(12.05, 2.6, "1 × 401,408", 9, "white")
draw_text(12.05, 2.35, "Vector", 9, "white")


# --------------------------------------------------
# Fully connected layers
# --------------------------------------------------
draw_text(
    14.5, 5.7,
    "Fully Connected Layers",
    12, "white", "bold"
)

draw_neurons(12.9, 3.3, "Dense", "512")
draw_neurons(14.0, 3.3, "Dense", "256")
draw_neurons(15.1, 3.3, "Dense", "128")
draw_neurons(16.2, 3.3, "Output", "Classes")


# --------------------------------------------------
# Dropout blocks
# --------------------------------------------------
for x in [13.65, 14.75, 15.85]:

    dropout = Rectangle(
        (x, 3.6), 0.3, 1.5,
        facecolor="white",
        edgecolor="#777777",
        linewidth=1.2
    )

    ax.add_patch(dropout)

    draw_text(x + 0.15, 5.35, "Dropout", 8)
    draw_text(x + 0.15, 3.35, "50%", 8)


# --------------------------------------------------
# Final output
# --------------------------------------------------
draw_text(17.3, 5.0, "Output", 11, "white", "bold")

output_box = FancyBboxPatch(
    (16.9, 4.0), 0.9, 0.7,
    boxstyle="round,pad=0.08",
    facecolor="#EAF4FF",
    edgecolor="#1683E8",
    linewidth=1.5
)

ax.add_patch(output_box)

draw_text(17.35, 4.35, "Class", 10)


# --------------------------------------------------
# Connect all stages with arrows
# --------------------------------------------------
arrow_y = 4.25

connections = [
    (1.4, 1.7),
    (5.1, 5.4),
    (6.3, 6.6),
    (10.0, 10.3),
    (11.2, 11.8),
    (12.3, 12.9),
    (13.5, 13.65),
    (13.95, 14.0),
    (14.6, 14.75),
    (15.3, 15.85),
    (16.5, 16.9)
]

for x1, x2 in connections:
    draw_arrow(x1, arrow_y, x2, arrow_y)


# --------------------------------------------------
# Save and display image
# --------------------------------------------------
plt.tight_layout()

plt.savefig(
    "cnn_architecture.png",
    dpi=300,
    bbox_inches="tight",
    facecolor=fig.get_facecolor()
)

plt.show()
