import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageDraw
import os

# Directory to save images and masks
dataset_dir = 'data'
os.makedirs(dataset_dir, exist_ok=True)


image_dir = 'node_images'

def get_image(name):
    img_path = os.path.join(image_dir, f"{name}.png")
    return plt.imread(img_path)

def apply_color_overlay(image, color):
    colored_image = np.zeros_like(image)
    colored_image[..., :3] = color[:3]
    colored_image[..., 3] = image[..., 3]
    return colored_image

def generate_random_ip():
    return f"{random.randint(192, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"

def create_random_network_diagram(file_index):
    G = nx.Graph()
    
    # Increase number of nodes for a larger diagram
    num_switches = random.randint(3, 6)
    num_computers = random.randint(5, 10)
    num_firewalls = random.randint(1, 3)
    num_servers = random.randint(1, 3)
    
    # Add nodes for router, switches, computers, firewalls, and servers
    G.add_node("Router", label=f"Router\n{generate_random_ip()}", device_type="router")
    for i in range(1, num_switches + 1):
        G.add_node(f"Switch {i}", label=f"Switch {i}\n{generate_random_ip()}", device_type="switch")
    for i in range(1, num_computers + 1):
        G.add_node(f"Computer {i}", label=f"Computer {i}\n{generate_random_ip()}", device_type="computer")
    for i in range(1, num_firewalls + 1):
        G.add_node(f"Firewall {i}", label=f"Firewall {i}\n{generate_random_ip()}", device_type="firewall")
    for i in range(1, num_servers + 1):
        G.add_node(f"Server {i}", label=f"Server {i}\n{generate_random_ip()}", device_type="server")

    switches = [f"Switch {i}" for i in range(1, num_switches + 1)]
    firewalls = [f"Firewall {i}" for i in range(1, num_firewalls + 1)]
    servers = [f"Server {i}" for i in range(1, num_servers + 1)]
    
    # Connect the router to each switch and firewall
    for switch in switches:
        G.add_edge("Router", switch)
    for firewall in firewalls:
        G.add_edge("Router", firewall)
    
    # Connect switches to a subset of computers and servers
    computers = [f"Computer {i}" for i in range(1, num_computers + 1)]
    for switch in switches:
        connected_computers = random.sample(computers, random.randint(2, min(4, len(computers))))
        for computer in connected_computers:
            G.add_edge(switch, computer)
        connected_servers = random.sample(servers, random.randint(1, len(servers)))
        for server in connected_servers:
            G.add_edge(switch, server)

    pos = nx.spring_layout(G, seed=random.randint(0, 100), k=0.8, iterations=20)
    
    # Create figure for the network diagram
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    ax.axis("off")

    # Create a blank mask image
    mask = Image.new("L", (1000, 800), color=0)
    draw_mask = ImageDraw.Draw(mask)

    for node, (x, y) in pos.items():
        device_type = G.nodes[node]['device_type']
        image = get_image(device_type)
        label = G.nodes[node]['label']
        
        # Adjust coordinates to fit in mask space (0-1000, 0-800)
        node_x, node_y = int((x + 0.5) * 900), int((y + 0.5) * 700)  # Coordinates are scaled
        
        # Draw the device on the network diagram
        im = OffsetImage(image, zoom=0.08)
        ab = AnnotationBbox(im, (x, y), frameon=False, zorder=3)
        ax.add_artist(ab)
        
        # Offset the label position slightly
        ax.text(x, y - 0.1, label, ha="center", fontsize=8, clip_on=True, zorder=4)

        # Draw mask with a filled circle for each device type
        device_label_value = device_labels[device_type]
        draw_mask.ellipse((node_x - 20, node_y - 20, node_x + 20, node_y + 20), fill=device_label_value)

    # Save the network diagram
    plt.savefig(os.path.join(dataset_dir, "images", f"network_diagram_{file_index}.png"), bbox_inches="tight")
    plt.close(fig)

for i in range(2000):
    create_random_network_diagram(i)

print(f"Generated 10 unique network diagrams with custom images in the '{dataset_dir}' directory.")






