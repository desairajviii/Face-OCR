import csv
import json
from pathlib import Path
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np

from src.analysis.conditions import get_condition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mtcnn = MTCNN(image_size=160, margin=20, device=device)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

DOC_TYPES = [
    "alb_id", "aze_passport", "esp_id", "est_id", "fin_id",
    "grc_passport", "lva_passport", "rus_internalpassport",
    "srb_passport", "svk_id",
]


def load_face_box(annotations_path: str, filename: str):
    """
    Given a MIDV-2020 annotation JSON and an image filename (e.g. '00.jpg'),
    return the face region's (x, y, width, height), or None if not found.
    """
    with open(annotations_path) as f:
        data = json.load(f)

    for key, entry in data["_via_img_metadata"].items():
        if entry["filename"] != filename:
            continue
        for region in entry["regions"]:
            if region["region_attributes"].get("field_name") == "face":
                attrs = region["shape_attributes"]
                return attrs["x"], attrs["y"], attrs["width"], attrs["height"]
    return None


def embed_face_from_box(image_path: str, box):
    """
    Crop the face region from an image using a known (x, y, width, height) box,
    then embed it with FaceNet. Used when we have ground-truth face annotations
    instead of relying on MTCNN detection.
    """
    img = Image.open(image_path).convert("RGB")
    x, y, w, h = box
    face_crop = img.crop((x, y, x + w, y + h)).resize((160, 160))

    face_array = np.array(face_crop).astype("float32")
    face_tensor = torch.tensor(face_array).permute(2, 0, 1).unsqueeze(0) / 255.0
    face_tensor = (face_tensor - 0.5) / 0.5  # normalize to [-1, 1], matching facenet-pytorch's convention

    with torch.no_grad():
        embedding = resnet(face_tensor.to(device))
    return embedding.squeeze(0).cpu()


def build_gallery(templates_dir: str):
    """
    Embed every template face across all 10 document types.
    Returns {doc_id: embedding}, where doc_id is '<doctype>_<index>', e.g. 'alb_id_00'.
    """
    gallery = {}
    templates_dir = Path(templates_dir)

    for doc_type in DOC_TYPES:
        annotations_path = templates_dir / "annotations" / f"{doc_type}.json"
        images_dir = templates_dir / "images" / doc_type

        for image_path in sorted(images_dir.glob("*.jpg")):
            filename = image_path.name
            box = load_face_box(str(annotations_path), filename)
            if box is None:
                print(f"WARNING: no face box for {doc_type}/{filename}")
                continue

            doc_id = f"{doc_type}_{image_path.stem}"
            gallery[doc_id] = embed_face_from_box(str(image_path), box)

    return gallery


def retrieve_top1(query_embedding, gallery: dict):
    """Return (best_match_id, cosine_similarity) for a query embedding against the gallery."""
    best_id, best_score = None, -1.0
    for doc_id, gallery_emb in gallery.items():
        score = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0), gallery_emb.unsqueeze(0)
        ).item()
        if score > best_score:
            best_id, best_score = doc_id, score
    return best_id, best_score


def run_full_evaluation(photo_dir: str, gallery: dict, out_csv_path: str):
    """
    Run face retrieval on every photo across all 10 document types,
    writing results to a CSV: doc_id, condition, face_top1_hit, face_top1_match_id, face_similarity
    """
    photo_dir = Path(photo_dir)
    rows = []

    for doc_type in DOC_TYPES:
        annotations_path = photo_dir / "annotations" / f"{doc_type}.json"
        images_dir = photo_dir / "images" / doc_type

        for image_path in sorted(images_dir.glob("*.jpg")):
            filename = image_path.name
            index = int(image_path.stem)  # e.g. "00.jpg" -> 0
            condition = get_condition(index)
            doc_id = f"{doc_type}_{image_path.stem}"

            box = load_face_box(str(annotations_path), filename)
            if box is None:
                rows.append([doc_id, condition, 0, "", 0.0])  # no face box = treat as miss
                continue

            query_emb = embed_face_from_box(str(image_path), box)
            match_id, score = retrieve_top1(query_emb, gallery)
            is_hit = int(match_id == doc_id)
            rows.append([doc_id, condition, is_hit, match_id, round(score, 4)])

        print(f"Finished {doc_type}")

    with open(out_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_id", "condition", "face_top1_hit", "face_top1_match_id", "face_similarity"])
        writer.writerows(rows)

    print(f"Done. Wrote {len(rows)} rows to {out_csv_path}")


def embed_image(image_path: str):
    img = Image.open(image_path).convert("RGB")
    face_tensor = mtcnn(img)
    if face_tensor is None:
        return None
    with torch.no_grad():
        embedding = resnet(face_tensor.unsqueeze(0).to(device))
    return embedding.squeeze(0).cpu()