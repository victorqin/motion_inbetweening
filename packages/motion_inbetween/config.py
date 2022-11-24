import os
import re
import json
import platform


project_root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
config_root = os.path.join(project_root, "configs")


def load_config(path):
    with open(path) as fh:
        return Config(json.load(fh), path)


def load_config_by_name(config_name):
    config_path = os.path.join(config_root, config_name + ".json")
    with open(config_path) as fh:
        return Config(json.load(fh), config_path)


class ConfigPathNotExistError(Exception):
    pass


class Config(dict):
    def __init__(self, config_dict, path, *args, **kwargs):
        super(Config, self).__init__(config_dict, *args, **kwargs)

        self.path = os.path.abspath(path)
        self.config_dir = os.path.dirname(self.path)

        # create workspace folder
        self["workspace"] = self["workspace"].format(name=self["name"])
        self["workspace"] = self._get_abs_path(
            self["workspace"], check_existance=False)
        if not os.path.exists(self["workspace"]):
            os.makedirs(self["workspace"])

        # checkpoint path
        self["train"]["checkpoint"] = self["train"]["checkpoint"].format(
            name=self["name"])
        self["train"]["checkpoint"] = os.path.join(
            self["workspace"], self["train"]["checkpoint"])

        # visdom environment name
        self["visdom"]["env"] = self["visdom"]["env"].format(name=self["name"])

        # convert database bvh_folder to absolute path
        for dataset_name in self["datasets"]:
            if "bvh_folder" in self["datasets"][dataset_name]:
                tmp_path = self["datasets"][dataset_name]["bvh_folder"]
                self["datasets"][dataset_name]["bvh_folder"] = \
                    self._get_abs_path(tmp_path)

    def _get_abs_path(self, path, check_existance=True):
        if platform.system() == "Windows":
            path = path.replace("/", "\\")
        else:
            path = path.replace("\\", "/")

        # check if input path is a absolute path
        if re.match(r"\/\w*|[a-zA-Z]\:[\/\\]\w*", path):
            abs_path = os.path.normpath(path)
        else:
            abs_path = os.path.normpath(
                os.path.join(self.config_dir, path))

        if check_existance and not os.path.exists(abs_path):
            raise ConfigPathNotExistError(
                "{} does not exist! Make sure the path is "
                "a valid relative path to the config file or a absolute"
                "path.".format(abs_path))

        return abs_path

    def save_to_workspace(self, workspace_path=None):
        if workspace_path is None:
            workspace_path = self["workspace"]
        path = os.path.join(workspace_path, "{}.json".format(self["name"]))
        with open(path, "w") as fh:
            json.dump(self, fh)
