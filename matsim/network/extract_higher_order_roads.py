import sys  
import gzip  
import xml.etree.ElementTree as ET  

def extract_links(infile, lanes_threshold, outfile="higher-order_roads_id.txt"):  
    # Determine if it is a .gz or a regular XML file 
    if infile.endswith('.gz'):  
        open_file = lambda f: gzip.open(f, 'rt', encoding='utf-8')  
    else:  
        open_file = lambda f: open(f, 'rt', encoding='utf-8')  
    ids = []  
    with open_file(infile) as fin:  
        # Only parse the links node to improve efficiency, can handle larger XML files.
        for event, elem in ET.iterparse(fin, events=('end',)):  
            if elem.tag == 'link':  
                permlanes = elem.get('permlanes')  
                link_id = elem.get('id')  
                if permlanes is not None and link_id is not None:  
                    try:  
                        if float(permlanes) >= lanes_threshold:  
                            ids.append(link_id)  
                    except Exception:  
                        pass  
                elem.clear()  
    # Write the result.
    with open(outfile, 'w', encoding='utf-8') as fout:  
        for lid in ids:  
            fout.write(f"{lid}\n")  
    print(f"Extracted {len(ids)} links with permlanes >={lanes_threshold} to {outfile}")  

if __name__ == '__main__':  
    if len(sys.argv) < 3:  
        print("Usage: python extract_higher_order_roads.py network.xml[.gz] permlanes_threshold")  
        sys.exit(1)  
    xmlfile = sys.argv[1]  
    thresh = float(sys.argv[2])  
    extract_links(xmlfile, thresh)