import os
import shutil
from icrawler.builtin import BingImageCrawler

all_keywords = {
    "slippers": [
        "slippers", "house slippers", "bedroom slippers",
        "slip on slippers", "indoor slippers", "fluffy slippers",
        "men slippers", "women slippers", "foam slippers",
        "slipper sandals", "slippers white background",
        "slippers product photo", "slippers isolated",
        "slippers top view", "rubber slippers", "flat slippers",
        "slippers amazon", "home slippers", "open toe slippers",
        "slippers pair"
    ],
    "motorcycle": [
        "motorcycle", "motorbike", "sport motorcycle",
        "cruiser motorcycle", "naked bike", "adventure motorcycle",
        "dirt bike", "street motorcycle", "yamaha motorcycle",
        "honda motorcycle", "kawasaki motorcycle", "suzuki motorcycle",
        "motorcycle side view", "motorcycle white background",
        "motorcycle isolated", "motorcycle product photo",
        "motorcycle studio", "motorcycle amazon", "motorcycle full view",
        "motorcycle photography"
    ],
    "tv_remote": [
        "tv remote control", "television remote", "remote control",
        "smart tv remote", "universal remote", "samsung remote",
        "lg remote control", "sony tv remote", "black remote control",
        "remote control white background", "remote control isolated",
        "remote control product photo", "remote control top view",
        "remote control amazon", "cable remote", "infrared remote",
        "home theater remote", "remote control buttons",
        "slim remote control", "tv remote photography"
    ]
}

for cls, keywords in all_keywords.items():
    for split, count in [("train", 240), ("validation", 30), ("test", 30)]:
        final_dir = f"./tf_classroom_data/{split}/{cls}"
        os.makedirs(final_dir, exist_ok=True)
        per_keyword = max(1, count // len(keywords))

        print(f"\n{'='*50}")
        print(f"  {cls.upper()} / {split} — target: {count} images")
        print(f"  {len(keywords)} keywords × {per_keyword} = ~{len(keywords)*per_keyword} images")
        print(f"{'='*50}")

        file_counter = 0
        for i, kw in enumerate(keywords):
            temp_dir = f"./temp/{cls}_{split}_{i}"
            os.makedirs(temp_dir, exist_ok=True)
            print(f"  [Bing] {kw} → {per_keyword} images")
            try:
                crawler = BingImageCrawler(
                    feeder_threads=2,
                    parser_threads=2,
                    downloader_threads=4,
                    storage={"root_dir": temp_dir}
                )
                crawler.crawl(keyword=kw, max_num=per_keyword)
            except Exception as e:
                print(f"  ⚠️ Failed: {e}")
                continue

            for fname in os.listdir(temp_dir):
                ext = os.path.splitext(fname)[1]
                new_name = f"{cls}_{file_counter:05d}{ext}"
                src = os.path.join(temp_dir, fname)
                dst = os.path.join(final_dir, new_name)
                shutil.move(src, dst)
                file_counter += 1
            shutil.rmtree(temp_dir)

        total = len(os.listdir(final_dir))
        print(f"\n  ✅ {cls} / {split} → {total} images collected")

if os.path.exists("./temp"):
    shutil.rmtree("./temp")

print("\n🎉 All done!")
print("\n📊 Final Count:")
for cls in all_keywords:
    for split in ["train", "validation", "test"]:
        path = f"./tf_classroom_data/{split}/{cls}"
        count = len(os.listdir(path)) if os.path.exists(path) else 0
        status = "✅" if count >= 20 else "⚠️"
        print(f"  {status} {split}/{cls}: {count} images")