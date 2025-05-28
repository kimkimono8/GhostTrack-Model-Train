import os
import csv

def parse_yolo_line(line):
    parts = line.strip().split()
    return int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

def severity_score(w_px, h_px):
    if h_px == 0 or w_px == 0:
        return "invalid"

    ratio = h_px / w_px
    if ratio > 6.0:
        return "severe"
    elif ratio > 5.0:
        return "high"
    elif ratio > 4.0:
        return "moderate"
    elif ratio > 3.5:
        return "borderline"
    else:
        return "ok"

def run_check(labels_dir, img_size=640, output_csv="qa_report_severity.csv"):
    results = []
    for root, _, files in os.walk(labels_dir):
        for fname in files:
            if not fname.endswith(".txt"): continue
            fpath = os.path.join(root, fname)
            with open(fpath, 'r') as f:
                for line in f.readlines():
                    if not line.strip(): continue
                    cls, cx, cy, w, h = parse_yolo_line(line)
                    if cls != 1: continue  # Check only class_id == 1 (person)
                    w_px = w * img_size
                    h_px = h * img_size
                    aspect_ratio = round(h_px / w_px, 2) if w_px > 0 else 0
                    severity = severity_score(w_px, h_px)

                    issue = "ok" if severity == "ok" else "aspect ratio off"
                    suggestion = {
                        "severe": "possible mislabel or very tall box - check image",
                        "high": "likely off - check manually",
                        "moderate": "check if tight",
                        "borderline": "likely okay - verify",
                        "ok": ""
                    }[severity]

                    results.append([
                        os.path.join(root, fname),
                        cls,
                        int(w_px),
                        int(h_px),
                        aspect_ratio,
                        severity,
                        issue,
                        suggestion
                    ])

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image", "class_id", "width_px", "height_px", "aspect_ratio", "severity", "issue", "suggestion"])
        writer.writerows(results)

    print(f"✅ QA with severity saved to: {output_csv}")

# Example usage
if __name__ == "__main__":
    import sys
    labels_path = sys.argv[1] if len(sys.argv) > 1 else "./labels"
    run_check(labels_path, img_size=640)
