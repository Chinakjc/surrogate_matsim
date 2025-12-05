#!/usr/bin/env python3  
import os  
import re  
import gzip  
import argparse  

def load_policy_ids(fn):  
    """Return the set of link IDs from the policy file"""  
    ids = set()  
    with open(fn, 'r', encoding='utf-8') as f:  
        for line in f:  
            line = line.strip()  
            if line and not line.startswith('#'):  
                ids.add(line)  
    return ids  

def apply_policy_stream(in_gz, out_gz, ids_to_half):  
    """  
    Read in_gz as a stream, write each line to out_gz;
    If a line is `<link ...>` and its id is in `ids_to_half`, then replace its `capacity` attribute with its original value * 0.5. 
    """  
    # Regular expression to match id and capacity. 
    id_re = re.compile(r'\bid="([^"]+)"')  
    cap_re = re.compile(r'\bcapacity="([^"]+)"')  
    
    with gzip.open(in_gz, 'rt', encoding='utf-8') as src, \
         gzip.open(out_gz, 'wt', encoding='utf-8') as dst:  
        
        for line in src:  
            if '<link ' in line:  
                m_id = id_re.search(line)  
                m_cap = cap_re.search(line)  
                if m_id and m_cap:  
                    lid = m_id.group(1)  
                    if lid in ids_to_half:  
                        # Calculate new capacity
                        try:  
                            cap_val = float(m_cap.group(1))  
                        except:  
                            raise RuntimeError(f"Could not parse capacity value: {m_cap.group(1)}")  
                        new_cap = cap_val * 0.5  
                        # Replace the first capacity="...":  
                        line = cap_re.sub(f'capacity="{new_cap}"', line, count=1)  
            dst.write(line)  

def main():  
    p = argparse.ArgumentParser(  
        description="Modify the capacity of a specified link in a MATSim network.xml.gz file by 50%, processing line by line.")  
    p.add_argument('--network', '-n', required=True,  
                   help="Original network file (network.xml.gz)")  
    p.add_argument('--policy-dir', '-p', required=True,  
                   help="policy folder, containing policy_roads_id_i.txt")  
    p.add_argument('--out-dir', '-o', default='scenario',  
                   help="Output directory (default: scenario/)")  
    args = p.parse_args()  

    os.makedirs(args.out_dir, exist_ok=True)  
    pattern = re.compile(r'policy_roads_id_(.+?)\.txt$')  

    for fn in os.listdir(args.policy_dir):  
        m = pattern.match(fn)  
        if not m: continue  
        idx = m.group(1)  
        policy_file = os.path.join(args.policy_dir, fn)  
        out_net = os.path.join(args.out_dir, f'network_scenario_{idx}.xml.gz')  
        
        print(f">> Policy #{idx}: Loading {policy_file}")  
        ids = load_policy_ids(policy_file)  
        print(f">> Write network file {out_net} (a total of {len(ids)} links, 50% need to be hit).")  
        apply_policy_stream(args.network, out_net, ids)  

    print("All done.")  

if __name__ == '__main__':  
    main()