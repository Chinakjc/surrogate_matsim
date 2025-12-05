import sys  
import os  
import random  

def main():  
    if len(sys.argv) < 4:  
        print("Usage: python sample_policy_ids.py N M higher-order_roads_id.txt")  
        sys.exit(1)  
    N = int(sys.argv[1])  
    M = int(sys.argv[2])  
    in_file = sys.argv[3]  

    # Read original ID list  
    with open(in_file, 'r', encoding='utf-8') as fin:  
        ids = [line.strip() for line in fin if line.strip()]  
    total = len(ids)  
    if M > total:  
        print(f"Error: The number to extract M ({M}) is greater than the number of available IDs ({total})! Please reduce M or provide more IDs.")  
        sys.exit(1)  

    out_dir = 'policy'  
    os.makedirs(out_dir, exist_ok=True)  

    for i in range(1, N+1):  
        sampled = random.sample(ids, M)  # Ensure that each extraction retrieves M unique IDs.  
        out_path = os.path.join(out_dir, f'policy_roads_id_{i}.txt')  
        with open(out_path, 'w', encoding='utf-8') as fout:  
            for lid in sampled:  
                fout.write(f"{lid}\n")  
        print(f"File generated: {out_path}")  

if __name__ == '__main__':  
    main()