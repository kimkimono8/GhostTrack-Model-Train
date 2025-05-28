import cv2
import os

# ตั้งค่าพาธของโฟลเดอร์
folder_path = 'E:/GhostTrack_YOLOv12n_Train/datasets/images/train'

# ค้นหาเฉพาะไฟล์ .jpg
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
image_files.sort()

if not image_files:
    print("ไม่พบไฟล์ .jpg ในโฟลเดอร์:", folder_path)
    exit()

for filename in image_files:
    image_path = os.path.join(folder_path, filename)
    base_name = os.path.splitext(filename)[0]
    txt_path = os.path.join(folder_path, base_name + '.txt')

    img = cv2.imread(image_path)
    if img is None:
        print("❌ ไม่สามารถเปิดรูป:", filename)
        continue

    txt_exists = os.path.exists(txt_path)
    txt_empty = False

    if txt_exists:
        with open(txt_path, 'r', encoding='utf-8') as f:
            contents = f.read().strip()
            if contents == "":
                txt_empty = True
                print(f"⚠️ ไฟล์ {base_name}.txt ว่างเปล่า")

    # แสดงภาพ
    display_text = "Image Viewer - d: ลบ jpg+txt | n: ถัดไป | q: ออก"
    if not txt_exists:
        display_text = "⚠️ ไม่มีไฟล์ txt - " + display_text
    elif txt_empty:
        display_text = "⚠️ txt ว่าง - " + display_text

    cv2.imshow(display_text, img)
    key = cv2.waitKey(0) & 0xFF

    if key == ord('d'):
        try:
            os.remove(image_path)
            print("🗑️ ลบรูป:", filename)

            if txt_exists:
                os.remove(txt_path)
                print("🗑️ ลบไฟล์ txt:", base_name + '.txt')
            else:
                print("❌ ไม่พบไฟล์ txt สำหรับ:", base_name)
        except Exception as e:
            print("เกิดข้อผิดพลาดในการลบ:", e)

    elif key == ord('n'):
        print("➡️ ข้าม:", filename)
        continue

    elif key == ord('q'):
        print("👋 ออกจากโปรแกรม")
        break

cv2.destroyAllWindows()
