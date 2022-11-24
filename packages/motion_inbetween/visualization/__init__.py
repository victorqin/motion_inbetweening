import json


def save_data_to_json(json_path, positions, rotations, foot_contact,
                      parents, global_positions=None, global_rotations=None,
                      debug=False):
    """
    Save animation data to json.

    Args:
        json_path (str): JSON file path.
        positions (ndarray or torch.Tensor):
            Joint local positions. Shape: (frames, joints, 3)
        rotations (ndarray or torch.Tensor):
            Joint local rotations. Shape: (frames, joints, 3, 3)
        foot_contact (ndarray):
            Left foot and right foot contact. Shape: (frames, 4)
        parents (1D int ndarray or torch.Tensor):
            Joint parent indices.
        debug (bool, optional): Extra data will be included in the json
            file for debugging purposes. Defaults to false.
    """
    with open(json_path, "w") as fh:
        data = {
            "positions": positions.tolist(),
            "rotations": rotations.tolist(),
            "foot_contact": foot_contact.tolist(),
            "parents": parents.tolist(),
        }

        if debug:
            from motion_inbetween.data import utils_np
            global_rot, global_pos = utils_np.fk(rotations, positions, parents)
            if global_positions is None:
                data["global_positions"] = global_pos.tolist()
            else:
                data["global_positions"] = global_positions.tolist()

            if global_rotations is None:
                data["global_rotations"] = global_rot.tolist()
            else:
                data["global_rotations"] = global_rotations.tolist()

        json.dump(data, fh)
