./generate_configs.sh config_withinday.xml networks/



tmux new -s matsim_berlin_p
 
CONCURRENCY=65 \
JAVA_XMX=12G \
OUTPUT_BASE=./output_11_2025/berlin \
LINKSTATS_INTERVAL=10 \
LINKSTATS_AVERAGE=5 \
./run_matsim_berlin_parallel.sh \
  matsim-example-project-0.0.1-SNAPSHOT.jar \
  configs/

tmux attach -t matsim_berlin_p

pgrep -fa java


Usage: extract_berlin_linkstats.sh [OPTIONS]  

OPTIONS:  
  -i, --input DIR       Input simulation directory (default: ./output)  
  -o, --output DIR      Output directory (default: ./extracted_linkstats)  
  -n, --iteration N     Iteration to extract:  
                          'last' - last available iteration (default)  
                          'all'  - all iterations  
                          NUMBER - specific iteration number  
  -m, --multiple N      Only process iterations that are multiples of N (default: 10)  
  -v, --verbose         Verbose output  
  -h, --help            Show this help message

./extract_berlin_linkstats.sh -i ./output_11_2025 -o ./results_11_2025