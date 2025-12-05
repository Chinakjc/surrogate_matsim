# 1. Create the log folder (if it doesn't exist).
mkdir -p logs  

# 2. Start a tmux session 
tmux new -s training_idf  

# 3. Activate conda environment 
conda activate tfgnn  

# 4. Run and save logs to the logs folder, with a timestamp.
ipython -i idf_pipeline.py 2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log  

# 5. Detach: Ctrl+B then press D 
# 6. Reconnect: tmux attach -t training_idf