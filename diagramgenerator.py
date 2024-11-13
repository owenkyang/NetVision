import networkx as nx
import matplotlib.pyplot as plt
import random
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os

dataset_dir = 'data'
os.makedirs(dataset_dir, exist_ok=True)


image_dir = 'node_images'

def get_image(name):
    img_path = os.path.join(image_dir, f"{name}.png")
    return plt.imread(img_path)

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
    plt.savefig(f"{dataset_dir}/network_diagram_{file_index}.png", bbox_inches="tight")
    plt.close()

for i in range(2000):
    create_random_network_diagram(i)

print(f"Generated 10 unique network diagrams with custom images in the '{dataset_dir}' directory.")






