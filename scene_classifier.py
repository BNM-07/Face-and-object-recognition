"""
Scene Classification Module — Places365 Deep Learning + COCO Object Fusion
Uses a pre-trained ResNet18 (Places365) for scene recognition,
combined with detected COCO objects for contextual fusion.
Falls back to color analysis if model can't be loaded.
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from collections import Counter
import os
import streamlit as st
import urllib.request

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PLACES365 SCENE CATEGORIES → SCENE TYPE MAPPING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Maps Places365 category keywords → our scene types
NATURE_KEYWORDS = {
    "mountain","valley","cliff","canyon","volcano","glacier","hill",
    "forest","rainforest","bamboo","jungle","swamp","marsh","bog","wetland",
    "beach","coast","shore","cove","cape",
    "ocean","sea","lake","river","waterfall","pond","creek","stream","canal",
    "desert","badlands","sand","dune",
    "field","meadow","pasture","prairie","tundra","steppe","moor",
    "sky","cloud","sun","sunrise","sunset",
    "snowfield","ski","ice","igloo","iceberg",
    "garden","park","yard","lawn","orchard","vineyard","farm","crop",
    "tree","flower","rock","cave","grotto",
    "trail","path","campsite","hot_spring","geyser","coral_reef",
}
URBAN_KEYWORDS = {
    "city","downtown","skyscraper","skyline","tower","building","office_building",
    "street","highway","road","intersection","bridge","overpass","tunnel",
    "parking","gas_station","bus_station","train_station","airport","subway",
    "market","bazaar","shopping","mall","plaza","square","sidewalk",
    "construction","industrial","factory","warehouse","dock","harbor","port",
    "stadium","arena","amusement_park","fairground","playground",
    "church","mosque","temple","cathedral","palace","castle","monument",
}
INDOOR_KEYWORDS = {
    "bedroom","kitchen","living_room","dining_room","bathroom","shower",
    "office","cubicle","conference","meeting","classroom","lecture",
    "library","bookstore","gallery","museum","theater","auditorium",
    "restaurant","bar","pub","cafe","coffee","cafeteria","food_court",
    "lobby","hallway","corridor","staircase","elevator","escalator",
    "gym","spa","sauna","pool","indoor","studio","attic","basement",
    "closet","pantry","laundry","garage","cellar","nursery","playroom",
    "hospital","pharmacy","dentist","veterinarian",
    "supermarket","shop","store","boutique",
}
WATER_KEYWORDS = {"ocean","sea","lake","river","waterfall","pond","creek","stream","canal","coast","beach","shore","harbor","dock","swimming_pool","water"}
MOUNTAIN_KEYWORDS = {"mountain","mountain_snowy","mountain_path","valley","cliff","canyon","volcano","glacier","hill","ridge","summit"}
FOREST_KEYWORDS = {"forest","forest_path","forest_road","rainforest","bamboo_forest","jungle","tree_farm","woodland"}
BEACH_KEYWORDS = {"beach","coast","shore","cove","seashore","boardwalk","pier"}
DESERT_KEYWORDS = {"desert","desert_road","desert_sand","desert_vegetation","badlands","dune","sand_dune"}
SNOW_KEYWORDS = {"snowfield","ski_slope","ski_resort","ice_skating","igloo","iceberg","glacier","mountain_snowy","tundra"}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COCO OBJECT → ENVIRONMENT MAPPING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NATURE_OBJECTS = {"bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
                  "potted plant","kite","surfboard","skis","snowboard","boat","frisbee"}
URBAN_OBJECTS  = {"car","motorcycle","bus","train","truck","traffic light","fire hydrant",
                  "stop sign","parking meter"}
INDOOR_OBJECTS = {"chair","couch","bed","dining table","bottle","wine glass","cup","fork",
                  "knife","spoon","bowl","book","clock","vase","remote","keyboard","mouse",
                  "laptop","tv","microwave","oven","toaster","sink","refrigerator","toilet"}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PLACES365 MODEL LOADING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PLACES365_URL = "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar"
CATEGORIES_URL = "https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt"

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

def _download_file(url, dest):
    """Download a file with progress feedback."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        urllib.request.urlretrieve(url, dest)

@st.cache_resource(show_spinner=False)
def load_places365():
    """Load Places365 ResNet18 model + category labels. Cached across reruns."""
    try:
        # Download categories
        cat_path = os.path.join(MODEL_DIR, "categories_places365.txt")
        _download_file(CATEGORIES_URL, cat_path)
        categories = []
        with open(cat_path, "r") as f:
            for line in f:
                # Format: "/a/abbey 0" → "abbey"
                cat = line.strip().split(" ")[0]
                cat = cat.split("/")[-1]  # Remove path prefix
                categories.append(cat)

        # Download model weights
        weights_path = os.path.join(MODEL_DIR, "resnet18_places365.pth.tar")
        _download_file(PLACES365_URL, weights_path)

        # Build model
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 365)
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
        model.load_state_dict(state_dict)
        model.eval()

        # Preprocessing transform
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        return model, categories, transform
    except Exception as e:
        print(f"[SceneClassifier] Failed to load Places365: {e}")
        return None, None, None


def run_scene_model(image_bgr, model, categories, transform, top_k=5):
    """Run Places365 inference. Returns list of (category, probability) tuples."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    input_tensor = transform(pil_img).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)
        top_probs, top_indices = probs.topk(top_k, dim=1)

    results = []
    for i in range(top_k):
        idx = top_indices[0][i].item()
        prob = top_probs[0][i].item()
        results.append((categories[idx], round(prob, 4)))
    return results


def _match_keywords(category, keyword_set):
    """Check if a Places365 category matches any keyword."""
    cat_lower = category.lower().replace("-", "_")
    for kw in keyword_set:
        if kw in cat_lower or cat_lower in kw:
            return True
    return False


def _get_scene_type_from_places(predictions):
    """Determine scene type from Places365 top predictions with weighted scoring."""
    nature_score = 0.0
    urban_score = 0.0
    indoor_score = 0.0

    for cat, prob in predictions:
        if _match_keywords(cat, NATURE_KEYWORDS):
            nature_score += prob
        if _match_keywords(cat, URBAN_KEYWORDS):
            urban_score += prob
        if _match_keywords(cat, INDOOR_KEYWORDS):
            indoor_score += prob

    return {"nature": round(nature_score, 3),
            "urban": round(urban_score, 3),
            "indoor": round(indoor_score, 3)}


def _get_specific_nature_type(predictions):
    """Determine specific nature sub-type from predictions."""
    for cat, prob in predictions:
        if prob < 0.05:
            continue
        if _match_keywords(cat, MOUNTAIN_KEYWORDS):
            return "mountain"
        if _match_keywords(cat, SNOW_KEYWORDS):
            return "snow"
        if _match_keywords(cat, FOREST_KEYWORDS):
            return "forest"
        if _match_keywords(cat, BEACH_KEYWORDS):
            return "beach"
        if _match_keywords(cat, DESERT_KEYWORDS):
            return "desert"
        if _match_keywords(cat, WATER_KEYWORDS):
            return "water"
    return "general"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN CLASSIFY FUNCTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def classify_scene(detections, image_bgr):
    """
    Classify scene using Places365 model + COCO object fusion.
    Returns dict with scene_type, scene_label, environment_tags, description, etc.
    """
    labels = [d["label"] for d in detections]
    label_set = set(labels)
    counts = Counter(labels)
    person_count = counts.get("person", 0)

    # ── Load Places365 model ──
    model, categories, transform = load_places365()
    places_predictions = []
    places_scores = {"nature": 0, "urban": 0, "indoor": 0}
    nature_subtype = "general"
    env_tags = []

    if model is not None:
        places_predictions = run_scene_model(image_bgr, model, categories, transform, top_k=5)
        places_scores = _get_scene_type_from_places(places_predictions)
        nature_subtype = _get_specific_nature_type(places_predictions)

        # Build env tags from top predictions
        for cat, prob in places_predictions:
            if prob > 0.08:
                env_tags.append(cat.replace("_", " "))

    # ── Object-based scoring (supplementary) ──
    obj_nature = sum(1 for l in label_set if l in NATURE_OBJECTS) * 0.15
    obj_urban  = sum(1 for l in label_set if l in URBAN_OBJECTS)  * 0.15
    obj_indoor = sum(1 for l in label_set if l in INDOOR_OBJECTS) * 0.15

    # Fuse Places365 scores + object scores
    final_scores = {
        "nature": round(places_scores["nature"] + obj_nature, 3),
        "urban":  round(places_scores["urban"]  + obj_urban,  3),
        "indoor": round(places_scores["indoor"] + obj_indoor, 3),
    }

    # ── Determine scene type ──
    scene_type = max(final_scores, key=final_scores.get)
    sorted_vals = sorted(final_scores.values(), reverse=True)
    if sorted_vals[0] < 0.1:
        scene_type = "unknown"
    elif sorted_vals[0] - sorted_vals[1] < 0.08 and sorted_vals[0] > 0.1:
        scene_type = "mixed"

    # ── Build scene label ──
    scene_label = _build_scene_label(scene_type, nature_subtype, places_predictions)

    # ── Build fusion description ──
    description = _build_fusion_description(scene_type, scene_label, person_count,
                                            counts, env_tags, places_predictions)

    return {
        "scene_type": scene_type,
        "scene_label": scene_label,
        "scene_scores": final_scores,
        "environment_tags": env_tags[:5],
        "places365_top5": places_predictions,
        "description": description,
        "person_count": person_count,
        "nature_subtype": nature_subtype,
    }


def _build_scene_label(scene_type, nature_subtype, predictions):
    """Generate human-readable scene label using Places365 + subtype."""
    top_cat = predictions[0][0] if predictions else ""
    top_cat_display = top_cat.replace("_", " ").title()

    if scene_type == "nature":
        labels = {
            "mountain": "🏔️ Mountain Landscape",
            "snow":     "❄️ Snowy Landscape",
            "forest":   "🌲 Forest Scene",
            "beach":    "🏖️ Beach / Coastal Scene",
            "desert":   "🏜️ Desert / Arid Landscape",
            "water":    "🌊 Water Body / River",
            "general":  "🌿 Natural Outdoor Scene",
        }
        base = labels.get(nature_subtype, "🌄 Natural Scene")
        if top_cat_display and nature_subtype != "general":
            return f"{base} ({top_cat_display})"
        return base
    elif scene_type == "urban":
        return f"🏙️ Urban Scene ({top_cat_display})" if top_cat_display else "🏙️ Urban / City Scene"
    elif scene_type == "indoor":
        return f"🏠 Indoor ({top_cat_display})" if top_cat_display else "🏠 Indoor Environment"
    elif scene_type == "mixed":
        return f"🔀 Mixed Environment ({top_cat_display})" if top_cat_display else "🔀 Mixed Environment"
    return "❓ Unclassified Scene"


def _build_fusion_description(scene_type, scene_label, person_count, counts, env_tags, predictions):
    """Generate contextual fusion description combining objects + scene."""
    person_str = ""
    if person_count == 1:
        person_str = "A person"
    elif person_count > 1:
        person_str = f"{person_count} people"

    other_objs = [(o, c) for o, c in counts.most_common() if o != "person"]
    obj_strs = []
    for o, c in other_objs[:4]:
        obj_strs.append(f"{c} {o}{'s' if c > 1 else ''}" if c > 1 else f"a {o}")

    parts = []
    if person_str:
        parts.append(person_str)
    if obj_strs:
        parts.append(", ".join(obj_strs))
    subj = " with ".join(parts) if parts else "Scene"

    # Use top Places365 prediction for context
    top_scene = predictions[0][0].replace("_", " ") if predictions else ""

    if scene_type == "nature":
        ctx = f"a {top_scene}" if top_scene else "a natural setting"
        return f"{subj} in {ctx}."
    elif scene_type == "urban":
        ctx = f"a {top_scene} area" if top_scene else "an urban environment"
        return f"{subj} in {ctx}."
    elif scene_type == "indoor":
        ctx = f"a {top_scene}" if top_scene else "an indoor setting"
        return f"{subj} in {ctx}."
    elif scene_type == "mixed":
        ctx = f"a {top_scene} environment" if top_scene else "a mixed setting"
        return f"{subj} in {ctx}."
    else:
        if parts:
            return f"{subj} detected in the scene."
        return "No significant scene features detected."


def get_scene_overlay_text(scene_result):
    """Short text for overlaying on annotated images."""
    return scene_result["scene_label"]
