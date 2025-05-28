import shutil
import os

# โฟลเดอร์ที่พบใน Roaming
cache_dir = r"C:\Users\Administrator\AppData\Roaming\ultralytics"

if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print(f"✅ เคลียร์ cache ที่: {cache_dir}")
else:
    print("❌ ไม่พบโฟลเดอร์ cache ที่ระบุ")
