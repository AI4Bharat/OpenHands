import json
import os
from urllib import parse
from tqdm.auto import tqdm
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

    def download_file(self, data, videos_save_dir):
        url = data["url"]
        start_time = data["start_time"]
        end_time = data["end_time"]
        text = data["text"]

        cmd = 'youtube-dl -f bestvideo[ext=mp4] "{}" -o "{}%(id)s.%(ext)s"'
        cmd = cmd.format(url, videos_save_dir + os.path.sep)
        rv = os.system(cmd)

        if rv:
            return False, None

        url_parsed = parse.urlparse(url)
        video_id = parse.parse_qs(url_parsed.query)["v"][0]

        video_path = os.path.join(videos_save_dir, video_id + ".mp4")
        save_video_path = os.path.join(videos_save_dir, video_id + "_" + text + ".mp4")

        cmd = "ffmpeg -y -i {} -ss {} -to {} {}".format(
            video_path, start_time, end_time, save_video_path
        )
        rv = os.system(cmd)
        os.remove(video_path)

        if rv:
            return False, None

        return True, video_id + "_" + text

    def download_split(self, split_size=100):
        assert split_size in [100, 200, 500, 1000]

        for data_type in ["train_data", "valid_data", "test_data"]:
            dataset = eval("self." + data_type)
            save_dir = os.path.join(self.save_path, "ASL_" + str(split_size), data_type)
            videos_save_dir = os.path.join(save_dir, "videos")
            if not os.path.exists(videos_save_dir):
                os.makedirs(videos_save_dir)

            data_list = []

            filtered_dataset = []
            for entry in dataset:
                if entry["label"] > split_size:
                    continue
                filtered_dataset.append(entry)

            for entry in tqdm(filtered_dataset, desc=data_type):
                success, video_id = self.download_file(entry, videos_save_dir)
                if not success:
                    continue

                entry["video_id"] = video_id
                data_list.append(entry)

            with open(os.path.join(save_dir, "labels.json"), "w") as fout:
                json.dump(data_list, fout)


