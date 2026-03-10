1. Only print summary
python matsim_network_summary.py network.xml
python matsim_network_summary.py network.xml.gz
2. Export JSON
python matsim_network_summary.py network.xml.gz --json summary.json
3. Export CSV
python matsim_network_summary.py network.xml.gz --csv summary.csv
4. Export both
python matsim_network_summary.py network.xml.gz --json summary.json --csv summary.csv

Output example
============================================================  
MATSim Network Summary  
============================================================  
File: network.xml.gz  

[Size Statistics]  
  Number of nodes: 12,345  
  Number of links: 28,901  

[Link Statistics]  
  Total length: 1,234,567.89 m  
  Average length: 42.72 m  
  Minimum length: 1.00 m  
  Maximum length: 2,345.67 m  

[Topology Statistics]  
  Isolated nodes: 12  
  Bidirectional node pairs: 10,245  
  One-way node pairs: 3,901