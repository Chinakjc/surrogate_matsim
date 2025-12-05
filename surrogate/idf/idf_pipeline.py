from my_model import *
from my_data import *
from my_plot import *
import seaborn as sns
from sklearn.metrics import explained_variance_score, r2_score
from datetime import datetime
def tprint(*args, **kwargs):  
    """  
    Print function with timestamp prefix.  
    
    Args:  
        *args: Variable length argument list to print  
        **kwargs: Arbitrary keyword arguments (same as print function)  
    
    Example:  
        tprint("Hello, World!")  
        tprint("Processing item:", 42)  
    """  
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  
    print(f"[{timestamp}]", *args, **kwargs)  


# 1. Set a global seed.
SEED = 1234  

# 2. OS Level: Guaranteeing that Python hashes are also reproducible  
os.environ['PYTHONHASHSEED'] = str(SEED)  

# 3. Make the underlying ops of TF use deterministic implementations as much as possible (2.x). 
os.environ['TF_DETERMINISTIC_OPS']    = '1'  
os.environ['TF_CUDNN_DETERMINISM']    = '1'  

# 4. Python comes with random built-in.
random.seed(SEED)  
# 5. NumPy  
np.random.seed(SEED)  
# 6. TensorFlow  
tf.random.set_seed(SEED) 

#parametre
batch_size = 1


tprint('Read data')

tprint('trainingset')
#Read data
data = prepare_training_data("idf_linkstats/train",
network_pattern_mid="network_idf", ls_pattern_mid=".linkstats_idf")  
tprint('validation set')
data_val = prepare_training_data("idf_linkstats/val",
network_pattern_mid="network_idf", ls_pattern_mid=".linkstats_idf")  

tprint('Read data complete')

tprint('Data Preprocessing')
#Data Normalization
#info = get_data_info(data) #cr2

dico_transf = dict()
dico_transf['length'] = tf.math.log1p
dico_transf['freespeed'] = lambda x : 1 - 1/x
dico_transf['capacity'] = tf.math.log1p
dico_transf['permlanes'] = tf.math.log1p
dico_transf['base_hrs_avg'] = tf.math.log1p
dico_transf['label'] = tf.math.log1p

data_t = transform_data(data=data,dico_transformer=dico_transf)
data_t
info_t = get_data_info(data_t)

dico_scaler = dict()
dico_scaler['length'] = lambda gt, stats, feat_name: linear_scaler_feat(data=gt, stats=stats, center='min', scale='range', feat_name=feat_name)  
dico_scaler['freespeed'] = lambda gt, stats, feat_name: linear_scaler_feat(data=gt, stats=stats, center='min', scale='range', feat_name=feat_name)  
dico_scaler['capacity'] = lambda gt, stats, feat_name: linear_scaler_feat(data=gt, stats=stats, center='min', scale='range', feat_name=feat_name)  
dico_scaler['permlanes'] = lambda gt, stats, feat_name: linear_scaler_feat(data=gt, stats=stats, center='min', scale='range', feat_name=feat_name)  
dico_scaler['xf'] = lambda gt, stats, feat_name: linear_scaler_feat(data=gt, stats=stats, center='avg', scale='std', feat_name=feat_name)  
dico_scaler['yf'] = lambda gt, stats, feat_name: linear_scaler_feat(data=gt, stats=stats, center='avg', scale='std', feat_name=feat_name)  
dico_scaler['xt'] = lambda gt, stats, feat_name: linear_scaler_feat(data=gt, stats=stats, center='avg', scale='std', feat_name=feat_name)  
dico_scaler['yt'] = lambda gt, stats, feat_name: linear_scaler_feat(data=gt, stats=stats, center='avg', scale='std', feat_name=feat_name)  
dico_scaler['base_hrs_avg'] = lambda gt, stats, feat_name: linear_scaler_feat(data=gt, stats=stats, center='min', scale='range', feat_name=feat_name)  
dico_scaler['label'] = lambda lt, stats: linear_scaler_lab(data=lt,stats=stats,center='min',scale='range')

data_s = scale_data(data=data_t,dico_scaler=dico_scaler,dico_info=info_t)
#info_s = get_data_info(data_s)

data_val_t = transform_data(data=data_val,dico_transformer=dico_transf) #cr1
info_val_t = get_data_info(data_val_t)
data_val_s = scale_data(data=data_val_t,dico_scaler=dico_scaler,dico_info=info_val_t)
#info_val_s = get_data_info(data_val_s)



#mini batch
ds_nor = build_dataset(data_s)
ds_val_nor = build_dataset(data_val_s)
train_dataset_nor = ds_nor
train_ds_batched_nor = train_dataset_nor.shuffle(buffer_size=3,seed=SEED).batch(batch_size=batch_size).repeat()

tprint('Data preprocessing completed')

#model
model_input_graph_spec, label_spec = train_dataset_nor.element_spec
del label_spec  # Delete unused tag specifications

# Build model
model_nor = build_regression_model(
    graph_tensor_spec=model_input_graph_spec,
    node_dim=128,
    edge_dim=32,
    message_dim=128,
    next_state_dim=128,
    output_dim=25,  
    num_message_passing=3,
    l2_regularization=5e-5,
    dropout_rate=0.01, )

# Compile model
model_nor.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[
        tf.keras.metrics.MeanAbsoluteError(),
        tf.keras.metrics.RootMeanSquaredError()
    ]
)

# Print model structure
model_nor.summary()



tprint('Model Training')
# Train model
history_nor = model_nor.fit(train_ds_batched_nor, steps_per_epoch=10,epochs=200,validation_data=ds_val_nor)

tprint('Model training completed')


history_plot = history_nor
metrics = [m for m in history_plot.history if not m.startswith("val_")]  
n = len(metrics)  

fig, axes = plt.subplots(n, 1, figsize=(8, 4 * n), squeeze=False)  
for i, metric in enumerate(metrics):  
    val_metric = f"val_{metric}"  
    if val_metric not in history_plot.history:  
        continue  

    epochs = range(1, len(history_plot.history[metric]) + 1)  
    ax1 = axes[i, 0]  
    ax2 = ax1.twinx()  

    # train  
    ax1.plot(epochs, history_plot.history[metric],  
             color='blue',  label=f"train {metric}")  
    ax1.set_ylabel(f"train {metric}", color='blue')  
    ax1.tick_params(axis='y', colors='blue')  

    # val  
    ax2.plot(epochs, history_plot.history[val_metric],  
             color='orange',  label=f"val {metric}",linestyle="--")  
    ax2.set_ylabel(f"val {metric}", color='orange')  
    ax2.tick_params(axis='y', colors='orange')  

     
    ax1.set_xlabel("Epoch")  
    ax1.grid(True, linestyle="--", alpha=0.5)  
    lines = ax1.get_lines() + ax2.get_lines()  
    labels = [l.get_label() for l in lines]  
    ax1.legend(lines, labels, loc="best")  
    ax1.set_title(f"{metric} Train vs Validation")  

plt.tight_layout()  
plt.savefig('training_history.png')


N_sample_policies = 15
predict_graphs_nor = []
actual_labels_nor = []
val_ds_nor = ds_nor.take(N_sample_policies) #ds_val_nor.take(10)
cpu_time = 0
# Generate prediction
for graph, labels in val_ds_nor:
    t1 = time.perf_counter()
    predict_graph = model_nor(graph)  # Prediction
    t2 = time.perf_counter()
    cpu_time +=  (t2-t1)
    predict_graphs_nor.append(predict_graph.numpy())  # Save prediction results
    actual_labels_nor.append([labels.numpy()])  # Save actual labels
print(f'CPU time for {N_sample_policies} predictions = {cpu_time} s.')
# Convert to NumPy array
predict_graphs_nor = np.concatenate(predict_graphs_nor, axis=0)
actual_labels_nor = np.concatenate(actual_labels_nor, axis=0)

save_plot_real_vs_pred_subsample(y_pred=predict_graphs_nor,y_real=actual_labels_nor,n_samples=800,filename='real_vs_pred.png')


predict_graphs_nor = []
actual_labels_nor = []
val_ds_nor = ds_val_nor
cpu_time = 0
# Generate prediction
for graph, labels in val_ds_nor:
    t1 = time.perf_counter()
    predict_graph = model_nor(graph)  # Prediction
    t2 = time.perf_counter()
    cpu_time +=  (t2-t1)
    predict_graphs_nor.append(predict_graph.numpy())  # Save prediction results
    actual_labels_nor.append([labels.numpy()])  # Save actual labels
print(f'CPU time for {len(data_val)} predictions = {cpu_time} s.')
# Convert to NumPy array
predict_graphs_nor = np.concatenate(predict_graphs_nor, axis=0)
actual_labels_nor = np.concatenate(actual_labels_nor, axis=0)






y_true = np.asarray(actual_labels_nor).ravel()  
y_pred = np.asarray(predict_graphs_nor).ravel()
eps = 1e-8
mask = y_true != 0  
mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
print(f"MAPE = {mape:.3f}%")  
smape = 2 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-6)) * 100  
print(f"SMAPE = {smape:.3f}%")
ape = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100    
median_ape = np.median(ape)  
print(f"Median APE = {median_ape:.3f}%")
wape = np.sum(np.abs(y_pred - y_true)) / np.sum(np.abs(y_true)) * 100  
print(f"WAPE = {wape:.3f}%")

fig = plt.figure(figsize=(8, 6))    
plt.hist(ape, bins=50)  
plt.xlabel("Absolute Percentage Error (%)")  
plt.ylabel("Count")  
plt.savefig('APE.png')


num = np.linalg.norm(y_true - y_pred, ord=2)  
den = np.linalg.norm(y_true,        ord=2)  
accuracy_l2 = 1 - num/(den + eps )     
print(f"Relative Accuracy (L2) = {accuracy_l2:.4f}  ({accuracy_l2*100:.2f}%)")  
from sklearn.metrics import explained_variance_score, r2_score
ev = explained_variance_score(y_true, y_pred)  
print(f"Explained Variance (sklearn) = {ev:.4f}  ({ev*100:.2f}%)")
r2 = r2_score(y_true, y_pred)  
print(f"R² on test set: {r2:.4f} ({r2*100:.2f}%)")  


def l2_error_ts(pt,tl):
    return np.linalg.norm(pt-tl, ord=2, axis=0) 
def autocorr(x):  
    x = np.asarray(x)  
    x = x - x.mean()  
    corr = np.correlate(x, x, mode='full')  
    mid  = len(corr)//2  
    acf  = corr[mid:] / corr[mid]             
    return acf  
def compute_acf(x, nlags=None):  
  
    x = np.asarray(x)  
    N = x.size  
    if nlags is None:  
        nlags = N - 1   
    x = x - x.mean()  
    corr = np.correlate(x, x, mode='full')       
    mid = len(corr) // 2  
    acf_full = corr[mid: mid + nlags + 1]  
    
    acf_full = acf_full / acf_full[0]  
    return acf_full 


errors_nor = list()
for i in range(len(actual_labels_nor)):
    pt = predict_graphs_nor[i]
    tl = actual_labels_nor[i]
    errors_nor.append(l2_error_ts(pt,tl))
errors_nor = np.array(errors_nor)
mean_err = errors_nor.mean(axis=0)  
std_err  = errors_nor.std(axis=0)  


start_hour = 0  
time_labels = [f"{(start_hour+i)%24:02d}:00–{(start_hour+i+1)%24:02d}:00"  
               for i in range(24)] + ["avg"]  
fig, ax = plt.subplots(figsize=(10,5))  
plt.style.use('ggplot')  
ax.plot(range(25), mean_err,  
        color='blue', linewidth=2.5, marker='o',  
        markerfacecolor='white', label='Mean L2 Error')  
ax.fill_between(range(25),  
                mean_err - std_err,  
                mean_err + std_err,  
                color='blue', alpha=0.2, label='±1 Std Dev')  
ax.set_xticks(range(25))  
ax.set_xticklabels(time_labels, rotation=45)  
ax.set_ylim(bottom=0)  
ax.set_xlabel("Time Interval")  
ax.set_ylabel("L2 Error")  
ax.set_title("Mean L2 Error with Std Dev over Time")  
ax.legend()  
ax.grid(True, linestyle='--', alpha=0.5)  
plt.tight_layout()  
plt.savefig('l2errors.png')


start_hour = 4 
t_range = 20
time_labels = [f"{(start_hour+i)%24:02d}:00–{(start_hour+i+1)%24:02d}:00"  
               for i in range(t_range)] 

errors_nor = list()
for i in range(len(actual_labels_nor)):
    pt = predict_graphs_nor[i][:,start_hour:start_hour+t_range]
    tl = actual_labels_nor[i][:,start_hour:start_hour+t_range]
    errors_nor.append(l2_error_ts(pt,tl))
errors_nor = np.array(errors_nor)

mean_err = errors_nor.mean(axis=0)  
std_err  = errors_nor.std(axis=0)  
fig, ax = plt.subplots(figsize=(10,5))  
plt.style.use('ggplot')  
ax.plot(range(t_range), mean_err,  
        color='blue', linewidth=2.5, marker='o',  
        markerfacecolor='white', label='Mean L2 Error')  
ax.fill_between(range(t_range),  
                mean_err - std_err,  
                mean_err + std_err,  
                color='blue', alpha=0.2, label='±1 Std Dev')  
ax.set_xticks(range(t_range))  
ax.set_xticklabels(time_labels, rotation=45)  
ax.set_ylim(bottom=0)  
ax.set_xlabel("Time Interval")  
ax.set_ylabel("L2 Error")  
ax.set_title("Mean L2 Error with Std Dev over Time")  
ax.legend()  
ax.grid(True, linestyle='--', alpha=0.5)  
plt.tight_layout()  
plt.savefig('l2errors_4_20.png')


N = mean_err.size  
nlags = N - 1  
# ACF  
acf_vals = compute_acf(mean_err, nlags=nlags)  
lags     = np.arange(nlags + 1)  
#start_hour = 7  
time_labels = [  
    f"{(start_hour + i) % 24:02d}:00–{(start_hour + i + 1) % 24:02d}:00"  
    for i in range(N)  
]  
plt.style.use('ggplot')  
fig, ax = plt.subplots(figsize=(10, 5))  
markerline, stemlines, baseline = ax.stem(  
    lags, acf_vals,  
    linefmt='teal', markerfmt='o', basefmt='k-'  
)  
plt.setp(stemlines, 'linewidth', 1.2)  
plt.setp(markerline, 'markersize', 6)  
ax.set_xticks(lags)  
ax.set_xticklabels(time_labels, rotation=45, ha='right')  
plt.plot(lags,lags*0+0.15,color='blue',linestyle='--')
plt.plot(lags,lags*0-0.15,color='blue',linestyle='--')
ax.set_xlabel('Time Interval')  
ax.set_ylabel('Autocorrelation')  
ax.set_title('Autocorrelation of Mean L2 Error')    
ax.set_xlim(-1, nlags + 1)  
#ax.set_ylim(-1.1, 1.1)  
ax.grid(True, linestyle='--', alpha=0.4)  
plt.tight_layout()  
plt.savefig('acf.png') 


#network error
index_t = [8,12,17,-1]
gt = next(iter(val_ds_nor))[0]
for ti in index_t:
    c_title = f" average {ti}:00 - {ti+1}:00"
    if ti==-1 or ti==25:
        c_title = " daily average"
    #print(f"ti = {ti}")
    hsr_np = gt.node_sets['links']['base_hrs_avg'].numpy()[:,ti]
    hrs_pt = predict_graphs_nor[0][:,ti]
    hrs_lb = actual_labels_nor[0][:,ti]
    
    save_plot_policy_network_2panels(  
        net_xml_path="idf_linkstats/network_idf_321.xml",  
        policy_links_txt="idf_linkstats/policy_roads_id_321.txt",  
        hrs_no_policy=hsr_np,   
        hrs_pred=hrs_pt,  
        hrs_real=hrs_lb,
        c_title=c_title,
        file_name= f'network_2panels_error_{c_title}.png' 
    )  

#network error reverse of logp1 (expm1)
index_t = [8,12,17,-1]
gt = next(iter(val_ds_nor))[0]
for ti in index_t:
    c_title = f" average {ti}:00 - {ti+1}:00"
    if ti==-1 or ti==25:
        c_title = " daily average"
    #print(f"ti = {ti}")
    hsr_np = gt.node_sets['links']['base_hrs_avg'].numpy()[:,ti]
    hrs_pt = predict_graphs_nor[0][:,ti]
    hrs_lb = actual_labels_nor[0][:,ti]
    
    save_plot_policy_network_2panels(  
        net_xml_path="idf_linkstats/network_idf_321.xml",  
        policy_links_txt="idf_linkstats/policy_roads_id_321.txt",  
        hrs_no_policy=hsr_np,   
        hrs_pred=np.expm1(hrs_pt),  
        hrs_real=np.expm1(hrs_lb),
        c_title=c_title,
        file_name= f'network_2panels_expm1_error_{c_title}.png' 
    )  


#RNN

data_rnn = prepare_rnn_data(data_s)
data_val_rnn = prepare_rnn_data(data_val_s)


ds_rnn = build_dataset_rnn(data_rnn)
ds_val_rnn = build_dataset_rnn(data_val_rnn)

# --------- Training set ----------
train_ds_batched_rnn = (
    ds_rnn
    .shuffle(buffer_size=3, seed=SEED)
    .batch(batch_size)          
    .repeat()                   
    .prefetch(tf.data.AUTOTUNE)
)

# --------- Validation set ----------
val_ds_batched_rnn = (
    ds_val_rnn
    .batch(batch_size)          
    .prefetch(tf.data.AUTOTUNE)
)
input_spec, label_spec = train_ds_batched_rnn.element_spec


#LSTM model
model_lstm = build_regression_model_lstm( input_tensor_spec = input_spec,output_tensor_spec =label_spec,hidden_dim = 8)
# Compile model
model_lstm.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[
        tf.keras.metrics.MeanAbsoluteError(),
        tf.keras.metrics.RootMeanSquaredError()
    ]
)
model_lstm.summary()

# Train model
history_lstm = model_lstm.fit(train_ds_batched_rnn, steps_per_epoch=10,epochs=200,validation_data=val_ds_batched_rnn)

#plot
history_plot = history_lstm 
metrics = [m for m in history_plot.history if not m.startswith("val_")]  
n = len(metrics)  

fig, axes = plt.subplots(n, 1, figsize=(8, 4 * n), squeeze=False)  
for i, metric in enumerate(metrics):  
    val_metric = f"val_{metric}"  
    if val_metric not in history_plot.history:  
        continue  

    epochs = range(1, len(history_plot.history[metric]) + 1)  
    ax1 = axes[i, 0]  
    ax2 = ax1.twinx()  

    # train  
    ax1.plot(epochs, history_plot.history[metric],  
             color='blue',  label=f"train {metric}")  
    ax1.set_ylabel(f"train {metric}", color='blue')  
    ax1.tick_params(axis='y', colors='blue')  

    # val  
    ax2.plot(epochs, history_plot.history[val_metric],  
             color='orange',  label=f"val {metric}",linestyle="--")  
    ax2.set_ylabel(f"val {metric}", color='orange')  
    ax2.tick_params(axis='y', colors='orange')  

     
    ax1.set_xlabel("Epoch")  
    ax1.grid(True, linestyle="--", alpha=0.5)  
    lines = ax1.get_lines() + ax2.get_lines()  
    labels = [l.get_label() for l in lines]  
    ax1.legend(lines, labels, loc="best")  
    ax1.set_title(f"{metric} Train vs Validation")  

plt.tight_layout()  
plt.savefig('training_history_lstm.png')


predict_graphs_lstm = []
actual_labels_lstm = []
val_ds_rnn = ds_val_rnn.take(10).batch(1).prefetch(tf.data.AUTOTUNE)
cpu_time = 0
# Generate prediction
for graph, labels in val_ds_rnn:
    t1 = time.perf_counter()
    predict_graph = model_lstm(graph)  # Prediction
    t2 = time.perf_counter()
    cpu_time +=  (t2-t1)
    predict_graphs_lstm.append(predict_graph.numpy())  # Save prediction results
    actual_labels_lstm.append(labels.numpy())  # Save actual labels
print(f'CPU time for 10 predictions = {cpu_time} s.')
# Convert to NumPy array
predict_graphs_lstm = np.concatenate(predict_graphs_lstm, axis=0)
actual_labels_lstm = np.concatenate(actual_labels_lstm, axis=0)
print(predict_graphs_lstm.shape)
print(actual_labels_lstm.shape)

save_plot_real_vs_pred_subsample(y_pred=predict_graphs_lstm,y_real=actual_labels_lstm,n_samples=800,filename='real_vs_pred_lstm.png')


predict_graphs_lstm = []
actual_labels_lstm = []
val_ds_lstm = (
    ds_val_rnn       # (23, 25, 9)
    .batch(1)         # -> (1, 23, 25, 9)
    .prefetch(tf.data.AUTOTUNE)
)
cpu_time = 0
# Generate prediction
for inputs, labels in val_ds_lstm:
    t1 = time.perf_counter()
    predict_graph = model_lstm(inputs)  # Prediction
    t2 = time.perf_counter()
    cpu_time +=  (t2-t1)
    predict_graphs_lstm.append(predict_graph.numpy())  # Save prediction results
    actual_labels_lstm.append(labels.numpy())  # Save actual labels
print(f'CPU time for {len(data_val)} predictions = {cpu_time} s.')
# Convert to NumPy array
predict_graphs_lstm = np.concatenate(predict_graphs_lstm, axis=0)
actual_labels_lstm = np.concatenate(actual_labels_lstm, axis=0)

predict_graphs_lstm = np.squeeze(predict_graphs_lstm,axis=-1)
actual_labels_lstm = np.squeeze(actual_labels_lstm,axis=-1)


#error
y_true = np.asarray(actual_labels_lstm).ravel()  
y_pred = np.asarray(predict_graphs_lstm).ravel()
eps = 1e-8
mask = y_true != 0  
mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
print(f"MAPE = {mape:.3f}%")  
smape = 2 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-6)) * 100  
print(f"SMAPE = {smape:.3f}%")
ape = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100    
median_ape = np.median(ape)  
print(f"Median APE = {median_ape:.3f}%")
wape = np.sum(np.abs(y_pred - y_true)) / np.sum(np.abs(y_true)) * 100  
print(f"WAPE = {wape:.3f}%")

fig = plt.figure(figsize=(8, 6))    
plt.hist(ape, bins=50)  
plt.xlabel("Absolute Percentage Error (%)")  
plt.ylabel("Count")  
plt.savefig('APE_lstm.png')


num = np.linalg.norm(y_true - y_pred, ord=2)  
den = np.linalg.norm(y_true,        ord=2)  
accuracy_l2 = 1 - num/(den + eps )     
print(f"Relative Accuracy (L2) = {accuracy_l2:.4f}  ({accuracy_l2*100:.2f}%)")  
ev = explained_variance_score(y_true, y_pred)  
print(f"Explained Variance (sklearn) = {ev:.4f}  ({ev*100:.2f}%)")
r2 = r2_score(y_true, y_pred)  
print(f"R² on test set: {r2:.4f} ({r2*100:.2f}%)")  


errors_lstm = list()
for i in range(len(actual_labels_lstm)):
    pt = predict_graphs_lstm[i]
    tl = actual_labels_lstm[i]
    errors_lstm.append(l2_error_ts(pt,tl))
errors_lstm = np.array(errors_lstm)
mean_err = errors_lstm.mean(axis=0)  
std_err  = errors_lstm.std(axis=0)  


start_hour = 0  
time_labels = [f"{(start_hour+i)%24:02d}:00–{(start_hour+i+1)%24:02d}:00"  
               for i in range(24)] + ["avg"]  
fig, ax = plt.subplots(figsize=(10,5))  
plt.style.use('ggplot')  
ax.plot(range(25), mean_err,  
        color='blue', linewidth=2.5, marker='o',  
        markerfacecolor='white', label='Mean L2 Error')  
ax.fill_between(range(25),  
                mean_err - std_err,  
                mean_err + std_err,  
                color='blue', alpha=0.2, label='±1 Std Dev')  
ax.set_xticks(range(25))  
ax.set_xticklabels(time_labels, rotation=45)  
ax.set_ylim(bottom=0)  
ax.set_xlabel("Time Interval")  
ax.set_ylabel("L2 Error")  
ax.set_title("Mean L2 Error with Std Dev over Time")  
ax.legend()  
ax.grid(True, linestyle='--', alpha=0.5)  
plt.tight_layout()  
plt.savefig('l2errors_lstm.png')


start_hour = 4 
t_range = 20
time_labels = [f"{(start_hour+i)%24:02d}:00–{(start_hour+i+1)%24:02d}:00"  
               for i in range(t_range)] 

errors_lstm = list()
for i in range(len(actual_labels_lstm)):
    pt = predict_graphs_lstm[i][:,start_hour:start_hour+t_range]
    tl = actual_labels_lstm[i][:,start_hour:start_hour+t_range]
    errors_lstm.append(l2_error_ts(pt,tl))
errors_lstm = np.array(errors_lstm)

mean_err = errors_lstm.mean(axis=0)  
std_err  = errors_lstm.std(axis=0)  
fig, ax = plt.subplots(figsize=(10,5))  
plt.style.use('ggplot')  
ax.plot(range(t_range), mean_err,  
        color='blue', linewidth=2.5, marker='o',  
        markerfacecolor='white', label='Mean L2 Error')  
ax.fill_between(range(t_range),  
                mean_err - std_err,  
                mean_err + std_err,  
                color='blue', alpha=0.2, label='±1 Std Dev')  
ax.set_xticks(range(t_range))  
ax.set_xticklabels(time_labels, rotation=45)  
ax.set_ylim(bottom=0)  
ax.set_xlabel("Time Interval")  
ax.set_ylabel("L2 Error")  
ax.set_title("Mean L2 Error with Std Dev over Time")  
ax.legend()  
ax.grid(True, linestyle='--', alpha=0.5)  
plt.tight_layout()  
plt.savefig('l2errors_4_20_lstm.png')


N = mean_err.size  
nlags = N - 1  
# ACF  
acf_vals = compute_acf(mean_err, nlags=nlags)  
lags     = np.arange(nlags + 1)  
#start_hour = 7  
time_labels = [  
    f"{(start_hour + i) % 24:02d}:00–{(start_hour + i + 1) % 24:02d}:00"  
    for i in range(N)  
]  
plt.style.use('ggplot')  
fig, ax = plt.subplots(figsize=(10, 5))  
markerline, stemlines, baseline = ax.stem(  
    lags, acf_vals,  
    linefmt='teal', markerfmt='o', basefmt='k-'  
)  
plt.setp(stemlines, 'linewidth', 1.2)  
plt.setp(markerline, 'markersize', 6)  
ax.set_xticks(lags)  
ax.set_xticklabels(time_labels, rotation=45, ha='right')  
plt.plot(lags,lags*0+0.15,color='blue',linestyle='--')
plt.plot(lags,lags*0-0.15,color='blue',linestyle='--')
ax.set_xlabel('Time Interval')  
ax.set_ylabel('Autocorrelation')  
ax.set_title('Autocorrelation of Mean L2 Error')    
ax.set_xlim(-1, nlags + 1)  
#ax.set_ylim(-1.1, 1.1)  
ax.grid(True, linestyle='--', alpha=0.4)  
plt.tight_layout()  
plt.savefig('acf_lstm.png') 


#network error reverse of logp1 (expm1)
index_t = [8,12,17,-1]
#gt = next(iter(val_ds_lstm))[0]
for ti in index_t:
    c_title = f" average {ti}:00 - {ti+1}:00"
    if ti==-1 or ti==25:
        c_title = " daily average"
    #print(f"ti = {ti}")
    hsr_np = gt.node_sets['links']['base_hrs_avg'].numpy()[:,ti]
    hrs_pt = predict_graphs_lstm[0][:,ti]
    hrs_lb = actual_labels_lstm[0][:,ti]
    
    save_plot_policy_network_2panels(  
        net_xml_path="idf_linkstats/network_idf_321.xml",  
        policy_links_txt="idf_linkstats/policy_roads_id_321.txt",  
        hrs_no_policy=hsr_np,   
        hrs_pred=np.expm1(hrs_pt),  
        hrs_real=np.expm1(hrs_lb),
        c_title=c_title,
        file_name= f'network_2panels_expm1_error_{c_title}.png' 
    )  


#GRU
 # Function to build a regression model
def build_regression_model_gru(input_tensor_spec,
                                output_tensor_spec,
                                hidden_dim=64):

    # ------------- Input -------------
    # input_spec.shape = (None, num_node, T, feat)
    inp = tf.keras.layers.Input(type_spec=input_tensor_spec)
    num_node = inp.shape[1]           
    T        = inp.shape[2]           
    feat_in  = inp.shape[3]

    # ---------- Merge node into batch ----------
    x = tf.keras.layers.Lambda(
            lambda z: tf.reshape(z, (-1, T, feat_in)),
            output_shape=(T, feat_in)               
        )(inp)                                       # (batch*num_node, T, feat)

    # ----------------- GRU -----------------
    x = tf.keras.layers.GRU(hidden_dim,
                             return_sequences=True)(x)  # (batch*num_node, T, hidden)

    # ------ Restore the node dimension again. ------
    def split_nodes(z):
        b = tf.shape(z)[0] // num_node                # Dynamic batch
        return tf.reshape(z, (b, num_node, T, hidden_dim))

    x = tf.keras.layers.Lambda(
            split_nodes,
            output_shape=(num_node, T, hidden_dim)    
        )(x)                                          # (batch, num_node, T, hidden)

    # ------------- Head -------------
    target_dim = output_tensor_spec.shape[-1]
    out = tf.keras.layers.TimeDistributed(            # node dim
            tf.keras.layers.TimeDistributed(          # time dim
                tf.keras.layers.Dense(target_dim))
          )(x)                                        # (batch, num_node, T, target_dim)

    model = tf.keras.Model(inp, out)
    return model
#GRU model
model_gru = build_regression_model_gru( input_tensor_spec = input_spec,output_tensor_spec =label_spec,hidden_dim = 8)
# Compile model
model_gru.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[
        tf.keras.metrics.MeanAbsoluteError(),
        tf.keras.metrics.RootMeanSquaredError()
    ]
)
model_gru.summary()


# Train model
history_gru = model_gru.fit(train_ds_batched_rnn, steps_per_epoch=10,epochs=200,validation_data=val_ds_batched_rnn)


#plot
history_plot = history_gru 
metrics = [m for m in history_plot.history if not m.startswith("val_")]  
n = len(metrics)  

fig, axes = plt.subplots(n, 1, figsize=(8, 4 * n), squeeze=False)  
for i, metric in enumerate(metrics):  
    val_metric = f"val_{metric}"  
    if val_metric not in history_plot.history:  
        continue  

    epochs = range(1, len(history_plot.history[metric]) + 1)  
    ax1 = axes[i, 0]  
    ax2 = ax1.twinx()  

    # train  
    ax1.plot(epochs, history_plot.history[metric],  
             color='blue',  label=f"train {metric}")  
    ax1.set_ylabel(f"train {metric}", color='blue')  
    ax1.tick_params(axis='y', colors='blue')  

    # val  
    ax2.plot(epochs, history_plot.history[val_metric],  
             color='orange',  label=f"val {metric}",linestyle="--")  
    ax2.set_ylabel(f"val {metric}", color='orange')  
    ax2.tick_params(axis='y', colors='orange')  

     
    ax1.set_xlabel("Epoch")  
    ax1.grid(True, linestyle="--", alpha=0.5)  
    lines = ax1.get_lines() + ax2.get_lines()  
    labels = [l.get_label() for l in lines]  
    ax1.legend(lines, labels, loc="best")  
    ax1.set_title(f"{metric} Train vs Validation")  

plt.tight_layout()  
plt.savefig('training_history_gru.png')


predict_graphs_gru = []
actual_labels_gru = []
val_ds_rnn = ds_val_rnn.take(10).batch(1).prefetch(tf.data.AUTOTUNE)
cpu_time = 0
# Generate prediction
for graph, labels in val_ds_rnn:
    t1 = time.perf_counter()
    predict_graph = model_gru(graph)  # Prediction
    t2 = time.perf_counter()
    cpu_time +=  (t2-t1)
    predict_graphs_gru.append(predict_graph.numpy())  # Save prediction results
    actual_labels_gru.append(labels.numpy())  # Save actual labels
print(f'CPU time for 10 predictions = {cpu_time} s.')
# Convert to NumPy array
predict_graphs_gru = np.concatenate(predict_graphs_gru, axis=0)
actual_labels_gru = np.concatenate(actual_labels_gru, axis=0)
print(predict_graphs_gru.shape)
print(actual_labels_gru.shape)

save_plot_real_vs_pred_subsample(y_pred=predict_graphs_gru,y_real=actual_labels_gru,n_samples=800,filename='real_vs_pred_gru.png')

predict_graphs_gru = []
actual_labels_gru = []
val_ds_gru = (
    ds_val_rnn       # (23, 25, 9)
    .batch(1)         # -> (1, 23, 25, 9)
    .prefetch(tf.data.AUTOTUNE)
)
cpu_time = 0
# Generate prediction
for inputs, labels in val_ds_gru:
    t1 = time.perf_counter()
    predict_graph = model_gru(inputs)  # Prediction
    t2 = time.perf_counter()
    cpu_time +=  (t2-t1)
    predict_graphs_gru.append(predict_graph.numpy())  # Save prediction results
    actual_labels_gru.append(labels.numpy())  # Save actual labels
print(f'CPU time for {len(data_val)} predictions = {cpu_time} s.')
# Convert to NumPy array
predict_graphs_gru = np.concatenate(predict_graphs_gru, axis=0)
actual_labels_gru = np.concatenate(actual_labels_gru, axis=0)

predict_graphs_gru = np.squeeze(predict_graphs_gru,axis=-1)
actual_labels_gru = np.squeeze(actual_labels_gru,axis=-1)

#error
y_true = np.asarray(actual_labels_gru).ravel()  
y_pred = np.asarray(predict_graphs_gru).ravel()
eps = 1e-8
mask = y_true != 0  
mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
print(f"MAPE = {mape:.3f}%")  
smape = 2 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-6)) * 100  
print(f"SMAPE = {smape:.3f}%")
ape = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100    
median_ape = np.median(ape)  
print(f"Median APE = {median_ape:.3f}%")
wape = np.sum(np.abs(y_pred - y_true)) / np.sum(np.abs(y_true)) * 100  
print(f"WAPE = {wape:.3f}%")

fig = plt.figure(figsize=(8, 6))    
plt.hist(ape, bins=50)  
plt.xlabel("Absolute Percentage Error (%)")  
plt.ylabel("Count")  
plt.savefig('APE_gru.png')

num = np.linalg.norm(y_true - y_pred, ord=2)  
den = np.linalg.norm(y_true,        ord=2)  
accuracy_l2 = 1 - num/(den + eps )     
print(f"Relative Accuracy (L2) = {accuracy_l2:.4f}  ({accuracy_l2*100:.2f}%)")  
ev = explained_variance_score(y_true, y_pred)  
print(f"Explained Variance (sklearn) = {ev:.4f}  ({ev*100:.2f}%)")
r2 = r2_score(y_true, y_pred)  
print(f"R² on test set: {r2:.4f} ({r2*100:.2f}%)") 

errors_gru = list()
for i in range(len(actual_labels_gru)):
    pt = predict_graphs_gru[i]
    tl = actual_labels_gru[i]
    errors_gru.append(l2_error_ts(pt,tl))
errors_gru = np.array(errors_gru)
mean_err = errors_gru.mean(axis=0)  
std_err  = errors_gru.std(axis=0)  

start_hour = 0  
time_labels = [f"{(start_hour+i)%24:02d}:00–{(start_hour+i+1)%24:02d}:00"  
               for i in range(24)] + ["avg"]  
fig, ax = plt.subplots(figsize=(10,5))  
plt.style.use('ggplot')  
ax.plot(range(25), mean_err,  
        color='blue', linewidth=2.5, marker='o',  
        markerfacecolor='white', label='Mean L2 Error')  
ax.fill_between(range(25),  
                mean_err - std_err,  
                mean_err + std_err,  
                color='blue', alpha=0.2, label='±1 Std Dev')  
ax.set_xticks(range(25))  
ax.set_xticklabels(time_labels, rotation=45)  
ax.set_ylim(bottom=0)  
ax.set_xlabel("Time Interval")  
ax.set_ylabel("L2 Error")  
ax.set_title("Mean L2 Error with Std Dev over Time")  
ax.legend()  
ax.grid(True, linestyle='--', alpha=0.5)  
plt.tight_layout()  
plt.savefig('l2errors_gru.png')

start_hour = 4 
t_range = 20
time_labels = [f"{(start_hour+i)%24:02d}:00–{(start_hour+i+1)%24:02d}:00"  
               for i in range(t_range)] 
errors_gru = list()
for i in range(len(actual_labels_gru)):
    pt = predict_graphs_gru[i][:,start_hour:start_hour+t_range]
    tl = actual_labels_gru[i][:,start_hour:start_hour+t_range]
    errors_gru.append(l2_error_ts(pt,tl))
errors_gru = np.array(errors_gru)

mean_err = errors_gru.mean(axis=0)  
std_err  = errors_gru.std(axis=0)  
fig, ax = plt.subplots(figsize=(10,5))  
plt.style.use('ggplot')  
ax.plot(range(t_range), mean_err,  
        color='blue', linewidth=2.5, marker='o',  
        markerfacecolor='white', label='Mean L2 Error')  
ax.fill_between(range(t_range),  
                mean_err - std_err,  
                mean_err + std_err,  
                color='blue', alpha=0.2, label='±1 Std Dev')  
ax.set_xticks(range(t_range))  
ax.set_xticklabels(time_labels, rotation=45)  
ax.set_ylim(bottom=0)  
ax.set_xlabel("Time Interval")  
ax.set_ylabel("L2 Error")  
ax.set_title("Mean L2 Error with Std Dev over Time")  
ax.legend()  
ax.grid(True, linestyle='--', alpha=0.5)  
plt.tight_layout()  
plt.savefig('l2errors_4_20_gru.png')

N = mean_err.size  
nlags = N - 1  
# ACF  
acf_vals = compute_acf(mean_err, nlags=nlags)  
lags     = np.arange(nlags + 1)  
#start_hour = 7  
time_labels = [  
    f"{(start_hour + i) % 24:02d}:00–{(start_hour + i + 1) % 24:02d}:00"  
    for i in range(N)  
]  
plt.style.use('ggplot')  
fig, ax = plt.subplots(figsize=(10, 5))  
markerline, stemlines, baseline = ax.stem(  
    lags, acf_vals,  
    linefmt='teal', markerfmt='o', basefmt='k-'  
)  
plt.setp(stemlines, 'linewidth', 1.2)  
plt.setp(markerline, 'markersize', 6)  
ax.set_xticks(lags)  
ax.set_xticklabels(time_labels, rotation=45, ha='right')  
plt.plot(lags,lags*0+0.15,color='blue',linestyle='--')
plt.plot(lags,lags*0-0.15,color='blue',linestyle='--')
ax.set_xlabel('Time Interval')  
ax.set_ylabel('Autocorrelation')  
ax.set_title('Autocorrelation of Mean L2 Error')    
ax.set_xlim(-1, nlags + 1)  
#ax.set_ylim(-1.1, 1.1)  
ax.grid(True, linestyle='--', alpha=0.4)  
plt.tight_layout()  
plt.savefig('acf_gru.png') 

#network error reverse of logp1 (expm1)
index_t = [8,12,17,-1]
#gt = next(iter(val_ds_gru))[0]
for ti in index_t:
    c_title = f" average {ti}:00 - {ti+1}:00"
    if ti==-1 or ti==25:
        c_title = " daily average"
    #print(f"ti = {ti}")
    hsr_np = gt.node_sets['links']['base_hrs_avg'].numpy()[:,ti]
    hrs_pt = predict_graphs_gru[0][:,ti]
    hrs_lb = actual_labels_gru[0][:,ti]
    
    save_plot_policy_network_2panels(  
        net_xml_path="idf_linkstats/network_idf_321.xml",  
        policy_links_txt="idf_linkstats/policy_roads_id_321.txt",  
        hrs_no_policy=hsr_np,   
        hrs_pred=np.expm1(hrs_pt),  
        hrs_real=np.expm1(hrs_lb),
        c_title=c_title,
        file_name= f'network_2panels_expm1_error_{c_title}.png' 
    )  

# NN

def build_regression_model_dense(input_tensor_spec,  
                                 output_tensor_spec,  
                                 hidden_dims=(64, 32),   # Number of neurons per layer 
                                 activation='relu',   
                                 dropout_rate=None):  
    """  
    A multi-layer Dense version of a time-series/multi-node regression model.

    Parameters
    ----
    input_tensor_spec : tf.TensorSpec
        Specification of the input tensor; shape = (None, num_node, T, feat)
    output_tensor_spec : tf.TensorSpec
        Specification of the output tensor; shape = (None, num_node, T, target_dim)
    hidden_dims : tuple
        Number of neurons in the hidden layers, e.g., (128, 64, 64)
    activation : str or callable
        Activation function for the Dense layers
    dropout_rate : float or None
        If not None, adds Dropout(dropout_rate) after each hidden layer
    """  
    
    
    # ---------- Input ----------  
    inp = tf.keras.layers.Input(type_spec=input_tensor_spec)  
    num_node = inp.shape[1]        # nodes  
    T        = inp.shape[2]        # times  
    feat_in  = inp.shape[3]        # input features  

    # ---------- Merge node into batch ----------  
    # (batch, num_node, T, feat) -> (batch*num_node, T, feat)  
    x = tf.keras.layers.Lambda(  
            lambda z: tf.reshape(z, (-1, T, feat_in)),  
            output_shape=(T, feat_in)  
        )(inp)  
    
    # ---------- Flatten time and features ----------  
    # (batch*num_node, T, feat) -> (batch*num_node, T*feat)  
    flat_dim = T * feat_in  
    x = tf.keras.layers.Flatten()(x)        # Equivalent to tf.reshape(-1, flat_dim)  

    # ---------- Stacking Dense hidden layers ----------  
    for units in hidden_dims:  
        x = tf.keras.layers.Dense(units, activation=activation)(x)  
        if dropout_rate is not None:  
            x = tf.keras.layers.Dropout(dropout_rate)(x)  
    
    # ---------- Output layer ----------  
    target_dim   = output_tensor_spec.shape[-1]  
    out_units    = T * target_dim            # Output target_dim at each time step.
    x = tf.keras.layers.Dense(out_units)(x)  # (batch*num_node, T*target_dim)  

    # ---------- Restore node/time dimension ----------  
    def split_nodes_and_time(z):  
        # Dynamic calculation batch_size  
        b = tf.shape(z)[0] // num_node  
        # (batch*num_node, T*target_dim) ->  
        # (batch, num_node, T, target_dim)  
        return tf.reshape(z, (b, num_node, T, target_dim))  

    out = tf.keras.layers.Lambda(  
            split_nodes_and_time,  
            output_shape=(num_node, T, target_dim)  
        )(x)  

    model = tf.keras.Model(inp, out)  
    return model  


#Dense
model_nn = build_regression_model_dense( input_tensor_spec = input_spec,output_tensor_spec =label_spec,hidden_dims = (4,4))
# Compile model
model_nn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[
        tf.keras.metrics.MeanAbsoluteError(),
        tf.keras.metrics.RootMeanSquaredError()
    ]
)
model_nn.summary()

# Train model
history_nn = model_nn.fit(train_ds_batched_rnn, steps_per_epoch=10,epochs=200,validation_data=val_ds_batched_rnn)

#plot
history_plot = history_nn 
metrics = [m for m in history_plot.history if not m.startswith("val_")]  
n = len(metrics)  

fig, axes = plt.subplots(n, 1, figsize=(8, 4 * n), squeeze=False)  
for i, metric in enumerate(metrics):  
    val_metric = f"val_{metric}"  
    if val_metric not in history_plot.history:  
        continue  

    epochs = range(1, len(history_plot.history[metric]) + 1)  
    ax1 = axes[i, 0]  
    ax2 = ax1.twinx()  

    # train  
    ax1.plot(epochs, history_plot.history[metric],  
             color='blue',  label=f"train {metric}")  
    ax1.set_ylabel(f"train {metric}", color='blue')  
    ax1.tick_params(axis='y', colors='blue')  

    # val  
    ax2.plot(epochs, history_plot.history[val_metric],  
             color='orange',  label=f"val {metric}",linestyle="--")  
    ax2.set_ylabel(f"val {metric}", color='orange')  
    ax2.tick_params(axis='y', colors='orange')  

     
    ax1.set_xlabel("Epoch")  
    ax1.grid(True, linestyle="--", alpha=0.5)  
    lines = ax1.get_lines() + ax2.get_lines()  
    labels = [l.get_label() for l in lines]  
    ax1.legend(lines, labels, loc="best")  
    ax1.set_title(f"{metric} Train vs Validation")  

plt.tight_layout()  
plt.savefig('training_history_nn.png')

predict_graphs_nn = []
actual_labels_nn = []
val_ds_rnn = ds_val_rnn.take(10).batch(1).prefetch(tf.data.AUTOTUNE)
cpu_time = 0
# Generate prediction
for graph, labels in val_ds_rnn:
    t1 = time.perf_counter()
    predict_graph = model_nn(graph)  # Prediction
    t2 = time.perf_counter()
    cpu_time +=  (t2-t1)
    predict_graphs_nn.append(predict_graph.numpy())  # Save prediction results
    actual_labels_nn.append(labels.numpy())  # Save actual labels
print(f'CPU time for 10 predictions = {cpu_time} s.')
# Convert to NumPy array
predict_graphs_nn = np.concatenate(predict_graphs_nn, axis=0)
actual_labels_nn = np.concatenate(actual_labels_nn, axis=0)
print(predict_graphs_nn.shape)
print(actual_labels_nn.shape)

save_plot_real_vs_pred_subsample(y_pred=predict_graphs_nn,y_real=actual_labels_nn,n_samples=800,filename='real_vs_pred_nn.png')

predict_graphs_nn = []
actual_labels_nn = []
val_ds_nn = (
    ds_val_rnn       # (23, 25, 9)
    .batch(1)         # -> (1, 23, 25, 9)
    .prefetch(tf.data.AUTOTUNE)
)
cpu_time = 0
# Generate prediction
for inputs, labels in val_ds_nn:
    t1 = time.perf_counter()
    predict_graph = model_nn(inputs)  # Prediction
    t2 = time.perf_counter()
    cpu_time +=  (t2-t1)
    predict_graphs_nn.append(predict_graph.numpy())  # Save prediction results
    actual_labels_nn.append(labels.numpy())  # Save actual labels
print(f'CPU time for {len(data_val)} predictions = {cpu_time} s.')
# Convert to NumPy array
predict_graphs_nn = np.concatenate(predict_graphs_nn, axis=0)
actual_labels_nn = np.concatenate(actual_labels_nn, axis=0)

predict_graphs_nn = np.squeeze(predict_graphs_nn,axis=-1)
actual_labels_nn = np.squeeze(actual_labels_nn,axis=-1)

#error
y_true = np.asarray(actual_labels_nn).ravel()  
y_pred = np.asarray(predict_graphs_nn).ravel()
eps = 1e-8
mask = y_true != 0  
mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
print(f"MAPE = {mape:.3f}%")  
smape = 2 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-6)) * 100  
print(f"SMAPE = {smape:.3f}%")
ape = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100    
median_ape = np.median(ape)  
print(f"Median APE = {median_ape:.3f}%")
wape = np.sum(np.abs(y_pred - y_true)) / np.sum(np.abs(y_true)) * 100  
print(f"WAPE = {wape:.3f}%")
fig = plt.figure(figsize=(8, 6))    
plt.hist(ape, bins=50)  
plt.xlabel("Absolute Percentage Error (%)")  
plt.ylabel("Count")  
plt.savefig('APE_nn.png')

num = np.linalg.norm(y_true - y_pred, ord=2)  
den = np.linalg.norm(y_true,        ord=2)  
accuracy_l2 = 1 - num/(den + eps )     
print(f"Relative Accuracy (L2) = {accuracy_l2:.4f}  ({accuracy_l2*100:.2f}%)")  
ev = explained_variance_score(y_true, y_pred)  
print(f"Explained Variance (sklearn) = {ev:.4f}  ({ev*100:.2f}%)")
r2 = r2_score(y_true, y_pred)  
print(f"R² on test set: {r2:.4f} ({r2*100:.2f}%)")  

errors_nn = list()
for i in range(len(actual_labels_nn)):
    pt = predict_graphs_nn[i]
    tl = actual_labels_nn[i]
    errors_nn.append(l2_error_ts(pt,tl))
errors_nn = np.array(errors_nn)
mean_err = errors_nn.mean(axis=0)  
std_err  = errors_nn.std(axis=0)  
start_hour = 0  
time_labels = [f"{(start_hour+i)%24:02d}:00–{(start_hour+i+1)%24:02d}:00"  
               for i in range(24)] + ["avg"]  
fig, ax = plt.subplots(figsize=(10,5))  
plt.style.use('ggplot')  
ax.plot(range(25), mean_err,  
        color='blue', linewidth=2.5, marker='o',  
        markerfacecolor='white', label='Mean L2 Error')  
ax.fill_between(range(25),  
                mean_err - std_err,  
                mean_err + std_err,  
                color='blue', alpha=0.2, label='±1 Std Dev')  
ax.set_xticks(range(25))  
ax.set_xticklabels(time_labels, rotation=45)  
ax.set_ylim(bottom=0)  
ax.set_xlabel("Time Interval")  
ax.set_ylabel("L2 Error")  
ax.set_title("Mean L2 Error with Std Dev over Time")  
ax.legend()  
ax.grid(True, linestyle='--', alpha=0.5)  
plt.tight_layout()  
plt.savefig('l2errors_nn.png')

start_hour = 4 
t_range = 20
time_labels = [f"{(start_hour+i)%24:02d}:00–{(start_hour+i+1)%24:02d}:00"  
               for i in range(t_range)] 
errors_nn = list()
for i in range(len(actual_labels_nn)):
    pt = predict_graphs_nn[i][:,start_hour:start_hour+t_range]
    tl = actual_labels_nn[i][:,start_hour:start_hour+t_range]
    errors_nn.append(l2_error_ts(pt,tl))
errors_nn = np.array(errors_nn)

mean_err = errors_nn.mean(axis=0)  
std_err  = errors_nn.std(axis=0)  
fig, ax = plt.subplots(figsize=(10,5))  
plt.style.use('ggplot')  
ax.plot(range(t_range), mean_err,  
        color='blue', linewidth=2.5, marker='o',  
        markerfacecolor='white', label='Mean L2 Error')  
ax.fill_between(range(t_range),  
                mean_err - std_err,  
                mean_err + std_err,  
                color='blue', alpha=0.2, label='±1 Std Dev')  
ax.set_xticks(range(t_range))  
ax.set_xticklabels(time_labels, rotation=45)  
ax.set_ylim(bottom=0)  
ax.set_xlabel("Time Interval")  
ax.set_ylabel("L2 Error")  
ax.set_title("Mean L2 Error with Std Dev over Time")  
ax.legend()  
ax.grid(True, linestyle='--', alpha=0.5)  
plt.tight_layout()  
plt.savefig('l2errors_4_20_nn.png')

N = mean_err.size  
nlags = N - 1  
# ACF  
acf_vals = compute_acf(mean_err, nlags=nlags)  
lags     = np.arange(nlags + 1)  
#start_hour = 7  
time_labels = [  
    f"{(start_hour + i) % 24:02d}:00–{(start_hour + i + 1) % 24:02d}:00"  
    for i in range(N)  
]  
plt.style.use('ggplot')  
fig, ax = plt.subplots(figsize=(10, 5))  
markerline, stemlines, baseline = ax.stem(  
    lags, acf_vals,  
    linefmt='teal', markerfmt='o', basefmt='k-'  
)  
plt.setp(stemlines, 'linewidth', 1.2)  
plt.setp(markerline, 'markersize', 6)  
ax.set_xticks(lags)  
ax.set_xticklabels(time_labels, rotation=45, ha='right')  
plt.plot(lags,lags*0+0.15,color='blue',linestyle='--')
plt.plot(lags,lags*0-0.15,color='blue',linestyle='--')
ax.set_xlabel('Time Interval')  
ax.set_ylabel('Autocorrelation')  
ax.set_title('Autocorrelation of Mean L2 Error')    
ax.set_xlim(-1, nlags + 1)  
#ax.set_ylim(-1.1, 1.1)  
ax.grid(True, linestyle='--', alpha=0.4)  
plt.tight_layout()  
plt.savefig('acf_nn.png')



# # Test GNn
# # Build model
# model_testgnn = build_regression_model(
#     graph_tensor_spec=model_input_graph_spec,
#     node_dim=128,
#     edge_dim=32,
#     message_dim=128,
#     next_state_dim=128,
#     output_dim=25,  
#     num_message_passing=3,
#     l2_regularization=5e-5,
#     dropout_rate=0.01, )

# # Compile model
# model_testgnn.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#     loss=tf.keras.losses.MeanSquaredError(),
#     metrics=[
#         tf.keras.metrics.MeanAbsoluteError(),
#         tf.keras.metrics.RootMeanSquaredError()
#     ]
# )

# # Print model structure
# model_testgnn.summary()



# tprint('Model Training')
# # Train model
# model_testgnn.fit(train_ds_batched_nor, steps_per_epoch=10,epochs=10,validation_data=ds_val_nor)


#Deeper NN
#Dense
model_nn2 = build_regression_model_dense( input_tensor_spec = input_spec,output_tensor_spec =label_spec,hidden_dims = (32,64,128,64,32))
# Compile model
model_nn2.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[
        tf.keras.metrics.MeanAbsoluteError(),
        tf.keras.metrics.RootMeanSquaredError()
    ]
)
model_nn2.summary()

# Train model
history_nn2 = model_nn2.fit(train_ds_batched_rnn, steps_per_epoch=10,epochs=200,validation_data=val_ds_batched_rnn)

#plot
history_plot = history_nn2 
metrics = [m for m in history_plot.history if not m.startswith("val_")]  
n = len(metrics)  

fig, axes = plt.subplots(n, 1, figsize=(8, 4 * n), squeeze=False)  
for i, metric in enumerate(metrics):  
    val_metric = f"val_{metric}"  
    if val_metric not in history_plot.history:  
        continue  

    epochs = range(1, len(history_plot.history[metric]) + 1)  
    ax1 = axes[i, 0]  
    ax2 = ax1.twinx()  

    # train  
    ax1.plot(epochs, history_plot.history[metric],  
             color='blue',  label=f"train {metric}")  
    ax1.set_ylabel(f"train {metric}", color='blue')  
    ax1.tick_params(axis='y', colors='blue')  

    # val  
    ax2.plot(epochs, history_plot.history[val_metric],  
             color='orange',  label=f"val {metric}",linestyle="--")  
    ax2.set_ylabel(f"val {metric}", color='orange')  
    ax2.tick_params(axis='y', colors='orange')  

     
    ax1.set_xlabel("Epoch")  
    ax1.grid(True, linestyle="--", alpha=0.5)  
    lines = ax1.get_lines() + ax2.get_lines()  
    labels = [l.get_label() for l in lines]  
    ax1.legend(lines, labels, loc="best")  
    ax1.set_title(f"{metric} Train vs Validation")  

plt.tight_layout()  
plt.savefig('training_history_nn2.png')


predict_graphs_nn2 = []
actual_labels_nn2 = []
val_ds_rnn = ds_val_rnn.take(10).batch(1).prefetch(tf.data.AUTOTUNE)
cpu_time = 0
# Generate prediction
for graph, labels in val_ds_rnn:
    t1 = time.perf_counter()
    predict_graph = model_nn2(graph)  # Prediction
    t2 = time.perf_counter()
    cpu_time +=  (t2-t1)
    predict_graphs_nn2.append(predict_graph.numpy())  # Save prediction results
    actual_labels_nn2.append(labels.numpy())  # Save actual labels
print(f'CPU time for 10 predictions = {cpu_time} s.')
# Convert to NumPy array
predict_graphs_nn2 = np.concatenate(predict_graphs_nn2, axis=0)
actual_labels_nn2 = np.concatenate(actual_labels_nn2, axis=0)
print(predict_graphs_nn2.shape)
print(actual_labels_nn2.shape)

save_plot_real_vs_pred_subsample(y_pred=predict_graphs_nn2,y_real=actual_labels_nn2,n_samples=800,filename='real_vs_pred_nn2.png')

predict_graphs_nn2 = []
actual_labels_nn2 = []
val_ds_nn2 = (
    ds_val_rnn       # (23, 25, 9)
    .batch(1)         # -> (1, 23, 25, 9)
    .prefetch(tf.data.AUTOTUNE)
)
cpu_time = 0
# Generate prediction
for inputs, labels in val_ds_nn2:
    t1 = time.perf_counter()
    predict_graph = model_nn2(inputs)  # Prediction
    t2 = time.perf_counter()
    cpu_time +=  (t2-t1)
    predict_graphs_nn2.append(predict_graph.numpy())  # Save prediction results
    actual_labels_nn2.append(labels.numpy())  # Save actual labels
print(f'CPU time for {len(data_val)} predictions = {cpu_time} s.')
# Convert to NumPy array
predict_graphs_nn2 = np.concatenate(predict_graphs_nn2, axis=0)
actual_labels_nn2 = np.concatenate(actual_labels_nn2, axis=0)

predict_graphs_nn2 = np.squeeze(predict_graphs_nn2,axis=-1)
actual_labels_nn2 = np.squeeze(actual_labels_nn2,axis=-1)

#error
y_true = np.asarray(actual_labels_nn2).ravel()  
y_pred = np.asarray(predict_graphs_nn2).ravel()
eps = 1e-8
mask = y_true != 0  
mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
print(f"MAPE = {mape:.3f}%")  
smape = 2 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-6)) * 100  
print(f"SMAPE = {smape:.3f}%")
ape = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100    
median_ape = np.median(ape)  
print(f"Median APE = {median_ape:.3f}%")
wape = np.sum(np.abs(y_pred - y_true)) / np.sum(np.abs(y_true)) * 100  
print(f"WAPE = {wape:.3f}%")
fig = plt.figure(figsize=(8, 6))    
plt.hist(ape, bins=50)  
plt.xlabel("Absolute Percentage Error (%)")  
plt.ylabel("Count")  
plt.savefig('APE_nn2.png')

num = np.linalg.norm(y_true - y_pred, ord=2)  
den = np.linalg.norm(y_true,        ord=2)  
accuracy_l2 = 1 - num/(den + eps )     
print(f"Relative Accuracy (L2) = {accuracy_l2:.4f}  ({accuracy_l2*100:.2f}%)")  
ev = explained_variance_score(y_true, y_pred)  
print(f"Explained Variance (sklearn) = {ev:.4f}  ({ev*100:.2f}%)")
r2 = r2_score(y_true, y_pred)  
print(f"R² on test set: {r2:.4f} ({r2*100:.2f}%)")  

errors_nn2 = list()
for i in range(len(actual_labels_nn2)):
    pt = predict_graphs_nn2[i]
    tl = actual_labels_nn2[i]
    errors_nn2.append(l2_error_ts(pt,tl))
errors_nn2 = np.array(errors_nn2)
mean_err = errors_nn2.mean(axis=0)  
std_err  = errors_nn2.std(axis=0)  
start_hour = 0  
time_labels = [f"{(start_hour+i)%24:02d}:00–{(start_hour+i+1)%24:02d}:00"  
               for i in range(24)] + ["avg"]  
fig, ax = plt.subplots(figsize=(10,5))  
plt.style.use('ggplot')  
ax.plot(range(25), mean_err,  
        color='blue', linewidth=2.5, marker='o',  
        markerfacecolor='white', label='Mean L2 Error')  
ax.fill_between(range(25),  
                mean_err - std_err,  
                mean_err + std_err,  
                color='blue', alpha=0.2, label='±1 Std Dev')  
ax.set_xticks(range(25))  
ax.set_xticklabels(time_labels, rotation=45)  
ax.set_ylim(bottom=0)  
ax.set_xlabel("Time Interval")  
ax.set_ylabel("L2 Error")  
ax.set_title("Mean L2 Error with Std Dev over Time")  
ax.legend()  
ax.grid(True, linestyle='--', alpha=0.5)  
plt.tight_layout()  
plt.savefig('l2errors_nn2.png')

start_hour = 4 
t_range = 20
time_labels = [f"{(start_hour+i)%24:02d}:00–{(start_hour+i+1)%24:02d}:00"  
               for i in range(t_range)] 
errors_nn2 = list()
for i in range(len(actual_labels_nn2)):
    pt = predict_graphs_nn2[i][:,start_hour:start_hour+t_range]
    tl = actual_labels_nn2[i][:,start_hour:start_hour+t_range]
    errors_nn2.append(l2_error_ts(pt,tl))
errors_nn2 = np.array(errors_nn2)

mean_err = errors_nn2.mean(axis=0)  
std_err  = errors_nn2.std(axis=0)  
fig, ax = plt.subplots(figsize=(10,5))  
plt.style.use('ggplot')  
ax.plot(range(t_range), mean_err,  
        color='blue', linewidth=2.5, marker='o',  
        markerfacecolor='white', label='Mean L2 Error')  
ax.fill_between(range(t_range),  
                mean_err - std_err,  
                mean_err + std_err,  
                color='blue', alpha=0.2, label='±1 Std Dev')  
ax.set_xticks(range(t_range))  
ax.set_xticklabels(time_labels, rotation=45)  
ax.set_ylim(bottom=0)  
ax.set_xlabel("Time Interval")  
ax.set_ylabel("L2 Error")  
ax.set_title("Mean L2 Error with Std Dev over Time")  
ax.legend()  
ax.grid(True, linestyle='--', alpha=0.5)  
plt.tight_layout()  
plt.savefig('l2errors_4_20_nn2.png')

N = mean_err.size  
nlags = N - 1  
# ACF  
acf_vals = compute_acf(mean_err, nlags=nlags)  
lags     = np.arange(nlags + 1)  
#start_hour = 7  
time_labels = [  
    f"{(start_hour + i) % 24:02d}:00–{(start_hour + i + 1) % 24:02d}:00"  
    for i in range(N)  
]  
plt.style.use('ggplot')  
fig, ax = plt.subplots(figsize=(10, 5))  
markerline, stemlines, baseline = ax.stem(  
    lags, acf_vals,  
    linefmt='teal', markerfmt='o', basefmt='k-'  
)  
plt.setp(stemlines, 'linewidth', 1.2)  
plt.setp(markerline, 'markersize', 6)  
ax.set_xticks(lags)  
ax.set_xticklabels(time_labels, rotation=45, ha='right')  
plt.plot(lags,lags*0+0.15,color='blue',linestyle='--')
plt.plot(lags,lags*0-0.15,color='blue',linestyle='--')
ax.set_xlabel('Time Interval')  
ax.set_ylabel('Autocorrelation')  
ax.set_title('Autocorrelation of Mean L2 Error')    
ax.set_xlim(-1, nlags + 1)  
#ax.set_ylim(-1.1, 1.1)  
ax.grid(True, linestyle='--', alpha=0.4)  
plt.tight_layout()  
plt.savefig('acf_nn2.png')


#network error reverse of logp1 (expm1)
index_t = [8,12,17,-1]
#gt = next(iter(val_ds_nn2))[0]
for ti in index_t:
    c_title = f" average {ti}:00 - {ti+1}:00"
    if ti==-1 or ti==25:
        c_title = " daily average"
    #print(f"ti = {ti}")
    hsr_np = gt.node_sets['links']['base_hrs_avg'].numpy()[:,ti]
    hrs_pt = predict_graphs_nn2[0][:,ti]
    hrs_lb = actual_labels_nn2[0][:,ti]
    
    save_plot_policy_network_2panels(  
        net_xml_path="idf_linkstats/network_idf_321.xml",  
        policy_links_txt="idf_linkstats/policy_roads_id_321.txt",  
        hrs_no_policy=hsr_np,   
        hrs_pred=np.expm1(hrs_pt),  
        hrs_real=np.expm1(hrs_lb),
        c_title=c_title,
        file_name= f'network_2panels_expm1_error_{c_title}.png' 
    )  