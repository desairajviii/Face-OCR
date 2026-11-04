# Maps a photo's numeric index (0-99) to its capture condition.
# Source: MIDV-2020 readme.txt, photo.tar condition table.
# (iPhone/Samsung ranges merged since we only care about the condition, not the phone model)

PHOTO_CONDITIONS = {
    "low_light":       list(range(70, 90)),
    "keyboard_bg":      list(range(30, 40)),
    "outdoors":         list(range(40, 50)),
    "table_bg":         list(range(50, 60)),
    "cloth_bg":         list(range(90, 100)),
    "text_docs_bg":     list(range(20, 30)),
    "projective_dist":  list(range(0, 20)),
    "highlight":        list(range(60, 70)),
}


def get_condition(photo_index: int) -> str:
    """Map a photo's numeric filename index (0-99) to its capture condition label."""
    for condition, indices in PHOTO_CONDITIONS.items():
        if photo_index in indices:
            return condition
    raise ValueError(f"Index {photo_index} not in any known condition range")