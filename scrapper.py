import os
import shutil
from icrawler.builtin import BingImageCrawler

all_keywords = {
    "headphones": [
        "headphones", "over ear headphones", "headset",
        "wireless headphones", "studio headphones",
        "bluetooth headphones", "noise cancelling headphones",
        "sony headphones", "gaming headset", "on ear headphones",
        "headphones white background", "headphones product photo",
        "audio headphones", "hifi headphones", "beats headphones",
        "headphones isolated", "headphones top view", "professional headphones",
        "headphones amazon", "bose headphones"
    ],
    "mug": [
        "mug cup", "coffee mug", "ceramic mug",
        "tea mug", "drinking mug", "white mug",
        "custom mug", "large mug", "porcelain mug", "travel mug",
        "coffee cup", "mug white background", "mug product photo",
        "espresso cup", "hot drink mug", "mug isolated",
        "mug top view", "mug amazon", "plain mug", "mug photography"
    ],
    "calculator": [
        "calculator", "scientific calculator", "desk calculator",
        "casio calculator", "school calculator",
        "texas instruments calculator", "graphing calculator",
        "basic calculator", "solar calculator", "math calculator",
        "calculator white background", "calculator product photo",
        "pocket calculator", "office calculator", "digital calculator",
        "calculator isolated", "calculator top view", "calculator amazon",
        "handheld calculator", "fx calculator"
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
                    feeder_threads=2,      # faster
                    parser_threads=2,      # faster
                    downloader_threads=4,  # faster
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