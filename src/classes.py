"""Классы датасета segments/sidewalk-semantic (ID 1..34 в датасете). Класс 0 («unlabeled») игнорируется:
в наших масках метки сдвинуты на -1, а «unlabeled» превращён в IGNORE_INDEX."""
CLASSES = [
    "road", "sidewalk", "crosswalk", "cyclinglane", "parkingdriveway", "railtrack", "curb",
    "person", "rider",
    "car", "truck", "bus", "tramtrain", "motorcycle", "bicycle", "caravan", "cartrailer",
    "building", "door", "wall", "fenceguardrail", "bridge", "tunnel", "stairs",
    "pole", "trafficsign", "trafficlight",
    "vegetation", "terrain", "sky",
    "ground", "dynamic", "static", "unclear",
]
NUM_CLASSES = len(CLASSES)          # 34
IGNORE_INDEX = 255
SIDEWALK = CLASSES.index("sidewalk")
