#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  

import argparse  
import re  
from pathlib import Path  
import sys  

def replace_first_value(xml_text: str, param_name: str, new_value: str) -> str:  
    """  
    In the XML text, replace the value of the first <param name="param_name" value="..."> with new_value.  
    Uses non-greedy and capturing groups, replaces only once, does not change other content or formatting.  
    """  
    pattern = re.compile(r'(<param\s+[^>]*\bname\s*=\s*"' + re.escape(param_name) + r'"\s+[^>]*\bvalue\s*=\s*")[^"]*(")',  
                         flags=re.IGNORECASE)  
    # Replace only the first occurrence  
    replaced, count = pattern.subn(r'\1' + new_value + r'\2', xml_text, count=1)  
    if count == 0:  
        raise RuntimeError(f'param name="{param_name}" not found in config (or this param does not have a value attribute)')  
    return replaced  

def main():  
    parser = argparse.ArgumentParser(  
        description="Generate MATSim batch config_n.xml files by pure text replacement (no XML parsing, minimal changes)"  
    )  
    parser.add_argument("--config", required=True, help="Path to the original runnable MATSim config file")  
    parser.add_argument("--networks-dir", required=True, help="Directory containing network_scenario_n.xml.gz files")  
    parser.add_argument("--output-configs-dir", required=True, help="Directory to output config_n.xml files")  
    parser.add_argument("--network-prefix", default="networks_idf/scenario",  
                        help="Network path prefix to write into config (default: networks_idf/scenario)")  
    args = parser.parse_args()  

    config_path = Path(args.config)  
    networks_dir = Path(args.networks_dir)  
    out_dir = Path(args.output_configs_dir)  
    net_prefix = args.network_prefix.rstrip("/")  

    if not config_path.is_file():  
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)  
        sys.exit(1)  
    if not networks_dir.is_dir():  
        print(f"Error: Network directory not found: {networks_dir}", file=sys.stderr)  
        sys.exit(1)  

    out_dir.mkdir(parents=True, exist_ok=True)  

    # Read the original config text (UTF-8, do not change encoding or BOM)  
    src = config_path.read_text(encoding="utf-8")  

    # List network files, identify n  
    pattern = re.compile(r"^network_scenario_(\d+)\.xml\.gz$")  
    items = []  
    for p in networks_dir.iterdir():  
        if p.is_file():  
            m = pattern.match(p.name)  
            if m:  
                n = int(m.group(1))  
                items.append((n, p.name))  

    if not items:  
        print(f"Warning: No network_scenario_n.xml.gz found in directory {networks_dir}", file=sys.stderr)  
        sys.exit(0)  

    items.sort(key=lambda x: x[0])  

    for n, filename in items:  
        # For each n, start from the original text to ensure independent generation  
        text = src  
  
        # Replace outputDirectory (first match)  
        text = replace_first_value(text, "outputDirectory", f"simulation_output/{n}")  
  
        # Replace inputNetworkFile (first match)  
        text = replace_first_value(text, "inputNetworkFile", f"{net_prefix}/{filename}")  
  
        out_file = out_dir / f"config_{n}.xml"  
        out_file.write_text(text, encoding="utf-8")  
        print(f"Generated: {out_file}")  

if __name__ == "__main__":  
    main()