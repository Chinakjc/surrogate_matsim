#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  

import gzip  
import xml.etree.ElementTree as ET  
import matplotlib.pyplot as plt  
import argparse  
import os  
import sys  

def parse_args():  
    p = argparse.ArgumentParser(  
        description="Draw a MATSim network and highlight specified sections."  
    )  
    p.add_argument(  
        "-n", "--network",  
        required=True,  
        help="Input MATSim network file (.xml or .xml.gz)"  
    )  
    p.add_argument(  
        "-p", "--policy",  
        required=True,  
        help="Text file containing link IDs to be highlighted, one ID per line."  
    )  
    p.add_argument(  
        "-o", "--output",  
        default=None,  
        help="Optional: Save the image to the specified path (e.g., output.png). If not specified, it will be displayed directly in a pop-up window."  
    )  
    return p.parse_args()  

def parse_matsim_network(gz_path):  
    """Parse network.xml or network.xml.gz, and return nodes and links"""  
    open_fn = gzip.open if gz_path.endswith(".gz") else open  
    with open_fn(gz_path, 'rt', encoding='utf-8') as f:  
        tree = ET.parse(f)  
    root = tree.getroot()  

    nodes = {}  
    for node in root.find('nodes').findall('node'):  
        nid = node.get('id')  
        x = float(node.get('x')); y = float(node.get('y'))  
        nodes[nid] = (x, y)  

    links = {}  
    for link in root.find('links').findall('link'):  
        lid = link.get('id')  
        links[lid] = (link.get('from'), link.get('to'))  
    return nodes, links  

def load_policy_link_ids(txt_path):  
    """Read a text file with one ID per line and return a set."""  
    ids = set()  
    with open(txt_path, 'r', encoding='utf-8') as f:  
        for line in f:  
            id_ = line.strip()  
            if id_:  
                ids.add(id_)  
    return ids  

def plot_network(nodes, links, policy_ids, output=None, figsize=(10,10)):  
    plt.figure(figsize=figsize)  
    ax = plt.gca()  
    ax.set_aspect('equal'); ax.axis('off')  

    for lid, (u, v) in links.items():  
        x1, y1 = nodes[u]; x2, y2 = nodes[v]  
        if lid in policy_ids:  
            ax.plot([x1,x2], [y1,y2],  
                    color='red', linewidth=1.5, zorder=2)  
        else:  
            ax.plot([x1,x2], [y1,y2],  
                    color='#999', linewidth=0.5, zorder=1)  

    if output:  
        plt.tight_layout()  
        plt.savefig(output, dpi=300)  
        print(f"Saved image to: {output}")  
    else:  
        plt.show()  

def main():  
    args = parse_args()  

    if not os.path.exists(args.network):  
        print("Error: Network file not found", args.network, file=sys.stderr)  
        sys.exit(1)  
    if not os.path.exists(args.policy):  
        print("Error: Policy file not found", args.policy, file=sys.stderr)  
        sys.exit(1)  

    print("Parsing network file …")  
    nodes, links = parse_matsim_network(args.network)  

    print("Read Policy link IDs …")  
    policy_ids = load_policy_link_ids(args.policy)  
    print(f"Number of nodes: {len(nodes)}, Number of links: {len(links)}, Number of links to highlight: {len(policy_ids)}")  

    print("Start drawing …")  
    plot_network(nodes, links, policy_ids, output=args.output)  

if __name__ == "__main__":  
    main()