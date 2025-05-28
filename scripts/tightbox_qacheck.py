import os
import csv

def parse_yolo_line(line):
    parts = line.strip().split()
    return int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

def run_check(labels_dir, img_size=640, output_csv="qa_report.csv"):
    results = []
    for root, _, files in os.walk(labels_dir):  # 👈 เดินทุก subdir
        for fname in files:
            if not fname.endswith(".txt"): continue
            fpath = os.path.join(root, fname)
            with open(fpath, 'r') as f:
                for line in f.readlines():
                    if not line.strip(): continue
                    cls, cx, cy, w, h = parse_yolo_line(line)
                    if cls != 1: continue  # Only check person (class_id == 1)
                    w_px = w * img_size
                    h_px = h * img_size
                    issue = suggestion = ""
                    if max(w_px, h_px) > 200 and (w_px/h_px < 0.2 or h_px/w_px > 5.0):
                        issue = "aspect ratio off (tall, likely close-up)"
                        suggestion = "review manually (close camera)"
                    elif w_px < 12 or h_px < 12:
                        issue = "box too small"
                        suggestion = "ignore or delete"
                    elif w_px > img_size or h_px > img_size:
                        issue = "box oversized"
                        suggestion = "tighten"
                    elif w_px/h_px > 4 or h_px/w_px > 4:
                        issue = "aspect ratio off"
                        suggestion = "check for mislabel"
                    else:
                        issue = "ok"
                    results.append([os.path.join(root, fname), cls, int(w_px), int(h_px), issue, suggestion])

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image", "class_id", "width_px", "height_px", "issue", "suggestion"])
        writer.writerows(results)

    print(f"✅ QA report saved to: {output_csv}")

# Example usage:
if __name__ == "__main__":
    import sys
    labels_path = sys.argv[1] if len(sys.argv) > 1 else "./labels"
    run_check(labels_path, img_size=640)
