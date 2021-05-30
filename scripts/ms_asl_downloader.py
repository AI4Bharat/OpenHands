import json
import os
from urllib import parse
from tqdm.auto import tqdm
import multiprocessing
from joblib import Parallel, delayed
import hashlib


class MSASLDownloader:
    def __init__(
        self, train_json_path, valid_json_path, test_json_path, save_path="./"
    ):
        self.train_json_path = train_json_path
        self.valid_json_path = valid_json_path
        self.test_json_path = test_json_path
        self.save_path = save_path

        self.train_data = json.load(open(self.train_json_path))
        self.valid_data = json.load(open(self.valid_json_path))
        self.test_data = json.load(open(self.test_json_path))

    def get_hash(self, content):
        return hashlib.sha1(content.encode("utf-8")).hexdigest()[:10]

    def download_file(self, data, videos_save_dir):

        url = data["url"]
        start_time = data["start_time"]
        end_time = data["end_time"]
        text = data["text"]

        hash = self.get_hash(text + str(start_time) + str(end_time))
        cmd = 'youtube-dl -f bestvideo[ext=mp4] "{}" -o "{}{}-%(id)s.%(ext)s"'
        cmd = cmd.format(url, videos_save_dir + os.path.sep, hash)
        rv = os.system(cmd)

        if rv:
            return

        url_parsed = parse.urlparse(url)
        video_id = parse.parse_qs(url_parsed.query)["v"][0]

        video_path = os.path.join(videos_save_dir, hash + "-" + video_id + ".mp4")
        save_name = video_id + "_" + text
        save_video_path = os.path.join(videos_save_dir, save_name + ".mp4")

        cmd = "ffmpeg -y -i {} -ss {} -to {} {}".format(
            video_path, start_time, end_time, save_video_path
        )
        rv = os.system(cmd)
        os.remove(video_path)

        if rv:
            return

        save_label_path = os.path.join(videos_save_dir, save_name + ".json").replace(
            "videos", "labels"
        )
        data["video_id"] = save_name

        with open(os.path.join(save_label_path), "w") as fout:
            json.dump([data], fout)

    def download_split(self, split_size=100, n_cores=None):
        assert split_size in [100, 200, 500, 1000]
        if n_cores is None:
            n_cores = multiprocessing.cpu_count()

        for data_type in ["train_data", "valid_data", "test_data"]:
            dataset = eval("self." + data_type)
            save_dir = os.path.join(self.save_path, "ASL_" + str(split_size), data_type)
            videos_save_dir = os.path.join(save_dir, "videos")
            labels_save_dir = videos_save_dir.replace("videos", "labels")

            if not os.path.exists(videos_save_dir):
                os.makedirs(videos_save_dir)

            if not os.path.exists(labels_save_dir):
                os.makedirs(labels_save_dir)

            filtered_dataset = []
            for entry in dataset:
                if entry["label"] >= split_size:
                    continue
                filtered_dataset.append(entry)

            Parallel(n_jobs=n_cores, backend="multiprocessing")(
                delayed(self.download_file)(entry, videos_save_dir)
                for entry in tqdm(filtered_dataset, desc=data_type)
            )

downloader = MSASLDownloader("MS-ASL/MSASL_train.json", "MS-ASL/MSASL_val.json", "MS-ASL/MSASL_test.json")
downloader.download_split(100)