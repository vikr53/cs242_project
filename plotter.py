import matplotlib.pyplot as plt
import pandas as pd
import csv

# Read baseline
headers=['Wall Time', 'Step', 'Val']

df_baseline = pd.read_csv('./final_data/baseline_resnet8_tfprep8_lr0.016_test_accuracy.csv', skiprows=[0], names=headers)

# Read Topk with feedback error correction
df_topk_fbk = pd.read_csv('./final_data/resnet88_local_fbe_topk_600010_test_accuracy.csv', skiprows=[0], names=headers)

# Read Topk, k = 6000
df_topk_6000 = pd.read_csv('./final_data/resnet88_local_topk_60001_test_accuracy.csv', skiprows=[0], names=headers)

plt.plot(df_baseline['Step'][:71], df_baseline['Val'][:71], label="Baseline")
plt.plot(df_topk_fbk['Step'], df_topk_fbk['Val'], label="0.01Topk-Feedback Err. Corr.")
plt.plot(df_topk_6000['Step'][:71], df_topk_6000['Val'][:71], label="0.01Topk")

plt.title("FLNet [Batch Size=8]")
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")

plt.legend()
plt.show()
