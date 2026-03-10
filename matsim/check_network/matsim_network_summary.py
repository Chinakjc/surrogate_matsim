#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  

import argparse  
import csv  
import gzip  
import json  
import re  
import statistics  
import xml.etree.ElementTree as ET  
from collections import Counter, defaultdict  


def open_network_file(file_path):  
    """Open a MATSim network file (.xml or .xml.gz)."""  
    if file_path.endswith(".gz"):  
        return gzip.open(file_path, "rb")  
    return open(file_path, "rb")  


def to_float(value, default=0.0):  
    """Safely convert a value to float."""  
    try:  
        return float(value)  
    except (TypeError, ValueError):  
        return default  


def parse_modes(modes_text):  
    """Split MATSim modes string into a list."""  
    if not modes_text:  
        return []  
    return [mode for mode in re.split(r"[,;\s]+", modes_text.strip()) if mode]  


def format_number(value):  
    """Format a number with commas and 2 decimal places."""  
    return f"{value:,.2f}"  


def write_json(output_path, result):  
    with open(output_path, "w", encoding="utf-8") as f:  
        json.dump(result, f, indent=2, ensure_ascii=False)  


def write_csv(output_path, result):  
    rows = []  

    def add_row(section, key, value):  
        rows.append({"section": section, "key": key, "value": value})  

    for key, value in result["network_attributes"].items():  
        add_row("network_attributes", key, value)  

    for key, value in result["size_statistics"].items():  
        add_row("size_statistics", key, value)  

    for key, value in result["link_statistics"].items():  
        add_row("link_statistics", key, value)  

    for key, value in result["node_coordinate_range"].items():  
        add_row("node_coordinate_range", key, value)  

    for key, value in result["topology_statistics"].items():  
        add_row("topology_statistics", key, value)  

    for key, value in result["allowed_modes_by_link_count"].items():  
        add_row("allowed_modes_by_link_count", key, value)  

    for key, value in result["mode_length_statistics_m"].items():  
        add_row("mode_length_statistics_m", key, value)  

    with open(output_path, "w", newline="", encoding="utf-8") as f:  
        writer = csv.DictWriter(f, fieldnames=["section", "key", "value"])  
        writer.writeheader()  
        writer.writerows(rows)  


def main():  
    parser = argparse.ArgumentParser(  
        description="Read a MATSim network.xml or network.xml.gz file and summarize the network."  
    )  
    parser.add_argument(  
        "network_file",  
        help="Path to the MATSim network file (.xml or .xml.gz)"  
    )  
    parser.add_argument(  
        "--json",  
        dest="json_output",  
        help="Write summary to a JSON file"  
    )  
    parser.add_argument(  
        "--csv",  
        dest="csv_output",  
        help="Write summary to a CSV file"  
    )  
    args = parser.parse_args()  

    node_count = 0  
    link_count = 0  

    total_link_length = 0.0  
    total_link_capacity = 0.0  

    link_lengths = []  
    link_capacities = []  
    free_speeds = []  
    lane_counts = []  

    allowed_mode_counter = Counter()  
    mode_length_counter = Counter()  

    node_ids = set()  
    in_degree = defaultdict(int)  
    out_degree = defaultdict(int)  
    directed_edges = set()  

    min_x = float("inf")  
    max_x = float("-inf")  
    min_y = float("inf")  
    max_y = float("-inf")  

    network_attributes = {}  
    network_tag_seen = False  

    with open_network_file(args.network_file) as file_obj:  
        xml_context = ET.iterparse(file_obj, events=("start", "end"))  

        for event, element in xml_context:  
            tag_name = element.tag.split("}")[-1]  

            if event == "start" and tag_name == "network" and not network_tag_seen:  
                network_attributes = dict(element.attrib)  
                network_tag_seen = True  

            elif event == "end":  
                if tag_name == "node":  
                    node_count += 1  

                    node_id = element.get("id")  
                    if node_id is not None:  
                        node_ids.add(node_id)  

                    x = to_float(element.get("x"), None)  
                    y = to_float(element.get("y"), None)  

                    if x is not None and y is not None:  
                        min_x = min(min_x, x)  
                        max_x = max(max_x, x)  
                        min_y = min(min_y, y)  
                        max_y = max(max_y, y)  

                    element.clear()  

                elif tag_name == "link":  
                    link_count += 1  

                    from_node = element.get("from")  
                    to_node = element.get("to")  

                    if from_node is not None:  
                        out_degree[from_node] += 1  
                    if to_node is not None:  
                        in_degree[to_node] += 1  
                    if from_node is not None and to_node is not None and from_node != to_node:  
                        directed_edges.add((from_node, to_node))  

                    length = to_float(element.get("length"), 0.0)  
                    capacity = to_float(element.get("capacity"), 0.0)  
                    free_speed = to_float(element.get("freespeed"), 0.0)  
                    lanes = to_float(element.get("permlanes"), 0.0)  

                    total_link_length += length  
                    total_link_capacity += capacity  

                    link_lengths.append(length)  
                    link_capacities.append(capacity)  
                    free_speeds.append(free_speed)  
                    lane_counts.append(lanes)  

                    modes = parse_modes(element.get("modes"))  
                    for mode in modes:  
                        allowed_mode_counter[mode] += 1  
                        mode_length_counter[mode] += length  

                    element.clear()  

                else:  
                    element.clear()  

    isolated_node_count = sum(  
        1 for node_id in node_ids  
        if in_degree[node_id] == 0 and out_degree[node_id] == 0  
    )  

    checked_pairs = set()  
    bidirectional_pair_count = 0  
    one_way_pair_count = 0  

    for u, v in directed_edges:  
        undirected_pair = tuple(sorted((u, v)))  
        if undirected_pair in checked_pairs:  
            continue  
        checked_pairs.add(undirected_pair)  

        if (u, v) in directed_edges and (v, u) in directed_edges:  
            bidirectional_pair_count += 1  
        else:  
            one_way_pair_count += 1  

    result = {  
        "file": args.network_file,  
        "network_attributes": network_attributes,  
        "size_statistics": {  
            "number_of_nodes": node_count,  
            "number_of_links": link_count,  
        },  
        "link_statistics": {  
            "total_length_m": total_link_length,  
            "average_length_m": (total_link_length / link_count) if link_count > 0 else 0.0,  
            "minimum_length_m": min(link_lengths) if link_lengths else 0.0,  
            "maximum_length_m": max(link_lengths) if link_lengths else 0.0,  
            "total_capacity": total_link_capacity,  
            "average_capacity": (total_link_capacity / link_count) if link_count > 0 else 0.0,  
            "average_free_speed_mps": statistics.mean(free_speeds) if free_speeds else 0.0,  
            "average_number_of_lanes": statistics.mean(lane_counts) if lane_counts else 0.0,  
        },  
        "node_coordinate_range": {  
            "min_x": min_x if min_x != float("inf") else None,  
            "max_x": max_x if max_x != float("-inf") else None,  
            "min_y": min_y if min_y != float("inf") else None,  
            "max_y": max_y if max_y != float("-inf") else None,  
        },  
        "topology_statistics": {  
            "isolated_nodes": isolated_node_count,  
            "bidirectional_node_pairs": bidirectional_pair_count,  
            "one_way_node_pairs": one_way_pair_count,  
        },  
        "allowed_modes_by_link_count": dict(allowed_mode_counter),  
        "mode_length_statistics_m": dict(mode_length_counter),  
    }  

    print("=" * 60)  
    print("MATSim Network Summary")  
    print("=" * 60)  
    print(f"File: {result['file']}")  

    if result["network_attributes"]:  
        print("\n[Network Attributes]")  
        for key, value in result["network_attributes"].items():  
            print(f"  {key}: {value}")  

    print("\n[Size Statistics]")  
    print(f"  Number of nodes: {result['size_statistics']['number_of_nodes']:,}")  
    print(f"  Number of links: {result['size_statistics']['number_of_links']:,}")  

    print("\n[Link Statistics]")  
    print(f"  Total length: {format_number(result['link_statistics']['total_length_m'])} m")  
    print(f"  Average length: {format_number(result['link_statistics']['average_length_m'])} m")  
    print(f"  Minimum length: {format_number(result['link_statistics']['minimum_length_m'])} m")  
    print(f"  Maximum length: {format_number(result['link_statistics']['maximum_length_m'])} m")  
    print(f"  Total capacity: {format_number(result['link_statistics']['total_capacity'])}")  
    print(f"  Average capacity: {format_number(result['link_statistics']['average_capacity'])}")  
    print(f"  Average free speed: {format_number(result['link_statistics']['average_free_speed_mps'])} m/s")  
    print(f"  Average number of lanes: {format_number(result['link_statistics']['average_number_of_lanes'])}")  

    if result["node_coordinate_range"]["min_x"] is not None:  
        print("\n[Node Coordinate Range]")  
        print(  
            f"  x: {format_number(result['node_coordinate_range']['min_x'])} "  
            f"~ {format_number(result['node_coordinate_range']['max_x'])}"  
        )  
        print(  
            f"  y: {format_number(result['node_coordinate_range']['min_y'])} "  
            f"~ {format_number(result['node_coordinate_range']['max_y'])}"  
        )  

    print("\n[Topology Statistics]")  
    print(f"  Isolated nodes: {result['topology_statistics']['isolated_nodes']:,}")  
    print(f"  Bidirectional node pairs: {result['topology_statistics']['bidirectional_node_pairs']:,}")  
    print(f"  One-way node pairs: {result['topology_statistics']['one_way_node_pairs']:,}")  

    if result["allowed_modes_by_link_count"]:  
        print("\n[Allowed Modes by Link Count]")  
        for mode, count in sorted(result["allowed_modes_by_link_count"].items()):  
            print(f"  {mode}: {count:,} links")  

    if result["mode_length_statistics_m"]:  
        print("\n[Mode Length Statistics]")  
        for mode, length in sorted(result["mode_length_statistics_m"].items()):  
            print(f"  {mode}: {format_number(length)} m")  

    if args.json_output:  
        write_json(args.json_output, result)  
        print(f"\nJSON written to: {args.json_output}")  

    if args.csv_output:  
        write_csv(args.csv_output, result)  
        print(f"CSV written to: {args.csv_output}")  

    print("\nNotes:")  
    print("  - In MATSim, a link usually represents a directed road segment.")  
    print("  - Therefore, the 'number of roads' is usually approximated by the number of links.")  
    print("  - Bidirectional node pairs mean at least one link exists in both directions.")  
    print("=" * 60)  


if __name__ == "__main__":  
    main()