import json
import pickle

import numpy as np


def load_pickle(fp):
    """Load a pickle file"""
    with open(fp, "rb") as f:
        data = pickle.load(f)
    return data


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types

    Based on: https://stackoverflow.com/a/47626762
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if __name__ == "__main__":
    fps_annotations = ["./data/cam1/annotations.pickle", "./data/cam2/annotations.pickle",
                       "./data/cam3/annotations.pickle", "./data/cam4/annotations.pickle"]
    fps_camera_params = ["./data/cam1/camera_params.pickle", "./data/cam2/camera_params.pickle",
                         "./data/cam3/camera_params.pickle", "./data/cam4/camera_params.pickle"]
    fps_extrinsics = ["./data/cam1/extrinsics.pickle", "./data/cam2/extrinsics.pickle",
                      "./data/cam3/extrinsics.pickle", "./data/cam4/extrinsics.pickle"]

    fps_json = ["./data/cam1/config.json", "./data/cam2/config.json",
                "./data/cam3/config.json", "./data/cam4/config.json"]

    for camera in range(len(fps_camera_params)):
        data = {}
        data["intrinsics"] = load_pickle(fps_camera_params[camera])
        data["extrinsics"] = load_pickle(fps_extrinsics[camera])
        data["annotations"] = load_pickle(fps_annotations[camera])

        # Write to json file
        dump = json.dumps(data, indent=4, cls=NumpyEncoder)
        with open(fps_json[camera], "w") as f:
            f.write(dump)

            # Write to pickle
        with open(fps_json[camera].replace(".json", ".pickle"), "wb") as f:
            pickle.dump(data, f)
