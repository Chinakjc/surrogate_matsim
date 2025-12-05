python3 extract_higher_order_roads.py ile_de_france_network.xml.gz 3

python3 sample_policy_ids.py 500 1000 higher-order_roads_id.txt

python3 apply_policy_to_network.py ile_de_france_network.xml.gz policy

python3 apply_policy_to_network.py  --network ile_de_france_network.xml.gz  --policy-dir policy  --out-dir scenario


python plot_matsim_network.py --network path/to/network.xml.gz --policy path/to/policy_roads_id.txt

python3 plot_matsim_network.py  -n ile_de_france_network.xml.gz -p policy/policy_roads_id_1.txt  -o map.png


python generate_matsim_configs.py \
  --config ile_de_france_config.xml \
  --networks-dir scenario \
  --output-configs-dir configs \
  --network-prefix ../scenario  