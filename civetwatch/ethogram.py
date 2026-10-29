"""The zoo's ethogram: every label CivetWatch can write, and how each is counted.

Behavior  = what the civet does (the ethogram).
Condition = how well the cameras can see it.
Both are saved for every sample so bad video is never mistaken for a behavior.
"""

# --- Duration behaviors (how long, minutes) --------------------------------
STRETCHED_OUT = "stretched_out"
LYING_DOWN = "lying_down"
SLEEP_IN_BOX = "sleep_in_box"
CURLING_UP = "curling_up"
WALKING = "walking"
ACTIVE_OTHER = "active_other"      # awake and moving in place (sniffing, standing)
NOT_VISIBLE = "not_visible"

# --- Frequency behaviors (how many times) ----------------------------------
FEEDING = "feeding"
DRINKING = "drinking"
GROOMING = "grooming"
MARKING = "territorial_marking"
URINATION = "urination"
DEFECATION = "defecation"

# --- Not a behavior: the frame cannot be judged ----------------------------
UNKNOWN = "unknown"                # poor image: blur, noise, dropped frame

INACTIVE_LABELS = [STRETCHED_OUT, LYING_DOWN, SLEEP_IN_BOX, CURLING_UP]
ACTIVE_LABELS = [WALKING, ACTIVE_OTHER, FEEDING, DRINKING, GROOMING, MARKING, URINATION, DEFECATION]
FREQUENCY_LABELS = [FEEDING, DRINKING, GROOMING, MARKING, URINATION, DEFECATION]
# Behaviors recognised by the GRU sequence model (everything else -> "other")
SEQUENCE_LABELS = [GROOMING, MARKING, URINATION, DEFECATION]

FRAME_LABELS = INACTIVE_LABELS + ACTIVE_LABELS + [NOT_VISIBLE, UNKNOWN]

INACTIVE = "Inactive"
ACTIVE = "Active"
OUT_OF_VIEW = "Not visible"        # state when the civet is unseen for longer than hold_max_s

DISPLAY = {
    STRETCHED_OUT: "Stretched out",
    LYING_DOWN: "Lying down",
    SLEEP_IN_BOX: "Sleep in a box",
    CURLING_UP: "Curling up",
    WALKING: "Locomotion / Walking",
    ACTIVE_OTHER: "Active (other)",
    NOT_VISIBLE: "Not visible",
    FEEDING: "Feeding",
    DRINKING: "Drinking",
    GROOMING: "Grooming",
    MARKING: "Territorial marking",
    URINATION: "Urination",
    DEFECATION: "Defecation",
    UNKNOWN: "Poor image",
    "possible_toileting": "Possible toileting (check)",
    "possible_marking": "Possible marking (check)",
    "possible_grooming": "Possible grooming (check)",
    "short_stop": "Short stop (check)",
    "doorway_pass": "Passing between rooms",
    INACTIVE: "Inactive",
    ACTIVE: "Active",
}


def state_of(label):
    """Inactive / Active for a frame label, or None when the label says nothing
    about the state (not visible, poor image)."""
    if label in INACTIVE_LABELS:
        return INACTIVE
    if label in ACTIVE_LABELS:
        return ACTIVE
    return None


# --- Pose keypoints (YOLO26-Pose trained on the zoo's own civet points) ----
KEYPOINTS = [
    "nose", "left_ear", "right_ear", "withers",
    "front_left_paw", "front_right_paw", "hind_left_paw", "hind_right_paw",
    "hips", "tail_base", "tail_tip",
]
KP = {name: i for i, name in enumerate(KEYPOINTS)}
# Index each keypoint swaps with on a horizontal flip (for YOLO pose training)
FLIP_IDX = [0, 2, 1, 3, 5, 4, 7, 6, 8, 9, 10]
SKELETON = [
    ("nose", "left_ear"), ("nose", "right_ear"), ("nose", "withers"),
    ("withers", "front_left_paw"), ("withers", "front_right_paw"),
    ("withers", "hips"), ("hips", "hind_left_paw"), ("hips", "hind_right_paw"),
    ("hips", "tail_base"), ("tail_base", "tail_tip"),
]

# --- Zones drawn once per camera (the tray zone also follows the detector) -
ZONE_TYPES = ["rest_box", "door", "fence", "log", "tyre", "food", "water", "latrine", "ignore"]
# 'latrine' = spot where the civet urinates / defecates (draw it where the keepers clean droppings)
MARKING_OBJECTS = ["log", "door", "rest_box"]      # things a civet rubs its scent gland on
# 'ignore' = burned-in clock / camera name, lens dirt, swaying plants: masked out of motion detection
