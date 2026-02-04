import re
import csv

log_file = r'c:\Users\y.hironaka\m2fin\train_log_20240906_extracted.txt'
output_csv = r'c:\Users\y.hironaka\m2fin\training_metrics_20240906.csv'
output_summary = r'c:\Users\y.hironaka\m2fin\training_summary_20240906.txt'

metrics = []
setup_info = {}

with open(log_file, 'r', encoding='utf-8') as f:
    text = f.read()

# Setup extraction
batch_size_match = re.search(r'batch_size=(\d+)', text)
if batch_size_match:
    setup_info['Batch Size'] = batch_size_match.group(1)

model_match = re.search(r'Model Summary: Params: ([\d.]+)M, Gflops: ([\d.]+)', text)
if model_match:
    setup_info['Model Params'] = model_match.group(1) + "M"
    setup_info['GFLOPs'] = model_match.group(2)

train_ann_match = re.search(r"train_ann\s+│\s+'([^']+)'", text)
if train_ann_match:
    setup_info['Train Annotation'] = train_ann_match.group(1)

# Metrics extraction
# Loss lines look like: epoch: 1/50, iter: 2500/2587, ..., total_loss: 3.368, ...
# Evaluation lines look like: Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.492

# Split by epoch start "start train epoch"
epoch_blocks = re.split(r'---> start train epoch(\d+)', text)

# epoch_blocks[0] is setup
# Then 1, block1, 2, block2, ...

for i in range(1, len(epoch_blocks), 2):
    epoch_num = int(epoch_blocks[i])
    block_content = epoch_blocks[i+1]
    
    # Get last loss of the epoch
    loss_matches = re.findall(r'total_loss: ([\d.]+), iou_loss: ([\d.]+), l1_loss: ([\d.]+), conf_loss: ([\d.]+), cls_loss: ([\d.]+)', block_content)
    if loss_matches:
        # Take the average of losses in this epoch for more stable figures, or just the last? usually last is fine for epoch end.
        # Let's take the mean.
        avg_total_loss = sum(float(m[0]) for m in loss_matches) / len(loss_matches)
        avg_iou_loss = sum(float(m[1]) for m in loss_matches) / len(loss_matches)
    else:
        avg_total_loss = None
        avg_iou_loss = None
    
    # Get AP metrics
    ap_match = re.search(r'Average Precision  \(AP\) @\[ IoU=0.50:0.95 \| area=   all \| maxDets=100 \] = ([\d.]+)', block_content)
    ap50_match = re.search(r'Average Precision  \(AP\) @\[ IoU=0.50      \| area=   all \| maxDets=100 \] = ([\d.]+)', block_content)
    ar_match = re.search(r'Average Recall     \(AR\) @\[ IoU=0.50:0.95 \| area=   all \| maxDets=100 \] = ([\d.]+)', block_content)
    
    metrics.append({
        'Epoch': epoch_num,
        'Total Loss': round(avg_total_loss, 4) if avg_total_loss else "",
        'IoU Loss': round(avg_iou_loss, 4) if avg_iou_loss else "",
        'AP(0.5:0.95)': ap_match.group(1) if ap_match else "",
        'AP@50': ap50_match.group(1) if ap50_match else "",
        'AR': ar_match.group(1) if ar_match else ""
    })

# Write CSV
with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['Epoch', 'Total Loss', 'IoU Loss', 'AP(0.5:0.95)', 'AP@50', 'AR'])
    writer.writeheader()
    writer.writerows(metrics)

# Write Summary
with open(output_summary, 'w', encoding='utf-8') as f:
    f.write("=== Training Session Summary (2024-09-06) ===\n")
    for k, v in setup_info.items():
        f.write(f"{k}: {v}\n")
    f.write("\n=== Epochwise Metrics ===\n")
    f.write(f"{'Epoch':<6} | {'Loss':<8} | {'AP':<8} | {'AP@50':<8}\n")
    f.write("-" * 40 + "\n")
    for m in metrics:
        f.write(f"{m['Epoch']:<6} | {str(m['Total Loss']):<8} | {str(m['AP(0.5:0.95)']):<8} | {str(m['AP@50']):<8}\n")

print("Done")
