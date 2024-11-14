import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os

dataset_dir = 'data'
os.makedirs(dataset_dir+'_in', exist_ok=True)
os.makedirs(dataset_dir+'_out', exist_ok=True)

image_dir = 'node_images'

device_colors = {
    "router": [0, 0, 1, 1],
    "switch": [1, 0.5, 0, 1],
    "computer": [0, 1, 0, 1],
    "firewall": [1, 0, 0, 1],
    "server": [0.5, 0, 0.5, 1]
}

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
    
    num_switches = random.randint(1, 3)
    num_computers = random.randint(2, 5)
    num_firewalls = random.randint(1, 2)
    num_servers = random.randint(1, 2) 
    
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
    
    for switch in switches:
        G.add_edge("Router", switch)
    for firewall in firewalls:
        G.add_edge("Router", firewall)
    
    computers = [f"Computer {i}" for i in range(1, num_computers + 1)]
    for switch in switches:
        connected_computers = random.sample(computers, random.randint(1, min(3, len(computers))))
        for computer in connected_computers:
            G.add_edge(switch, computer)
        
        connected_servers = random.sample(servers, random.randint(1, len(servers)))
        for server in connected_servers:
            G.add_edge(switch, server)

    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(1, len(components)):
            node_from_component = random.choice(list(components[i]))
            node_from_main = random.choice(list(components[0]))
            G.add_edge(node_from_component, node_from_main)


    pos = nx.spring_layout(G, seed=random.randint(0, 100), k=0.8, iterations=20)
    
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    for node, (x, y) in pos.items():
        device_type = G.nodes[node]['device_type']
        image = get_image(device_type)

        im = OffsetImage(image, zoom=0.08)
        ab = AnnotationBbox(im, (x, y), frameon=False, zorder=3)
        ax.add_artist(ab)
        
        label_y_offset = -0.15 if y > 0 else 0.15
        ax.text(x, y + label_y_offset, G.nodes[node]['label'], ha="center", fontsize=8, clip_on=True, zorder=4)

    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)


    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="gray")

    plt.axis('off')
    plt.savefig(f"{dataset_dir}_in/network_diagram_{file_index}.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    for node, (x, y) in pos.items():
        device_type = G.nodes[node]['device_type']
        image = get_image(device_type)
        
        color_overlay_image = apply_color_overlay(image, device_colors[device_type])

        im = OffsetImage(color_overlay_image, zoom=0.08)
        ab = AnnotationBbox(im, (x, y), frameon=False, zorder=3)
        ax.add_artist(ab)
        
        label_y_offset = -0.15 if y > 0 else 0.15
        ax.text(x, y + label_y_offset, G.nodes[node]['label'], ha="center", fontsize=8, clip_on=True, zorder=4)

    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="gray")

    plt.axis('off')
    plt.savefig(f"{dataset_dir}_out/network_diagram_{file_index}.png", bbox_inches="tight")
    plt.close()

num_diagrams = 20

for i in range(num_diagrams):
    create_random_network_diagram(i+1)

print(f"Generated {num_diagrams} unique network diagrams with custom images in the '{dataset_dir}' directory.")






