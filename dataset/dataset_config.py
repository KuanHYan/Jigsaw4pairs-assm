from easydict import EasyDict as edict

__C = edict()

dataset_cfg = __C

# Breaking Bad geometry assembly dataset

__C.BREAKING_BAD = edict()
__C.BREAKING_BAD.DATA_DIR = "data/breaking_bad"
__C.BREAKING_BAD.DATA_FN = (  # this would vary due to different subset, use everyday as default
    "everyday.{}.txt"
)
__C.BREAKING_BAD.DATA_KEYS = ("part_ids",)

__C.BREAKING_BAD.SUBSET = ""  # must in ['artifact', 'everyday', 'other']
__C.BREAKING_BAD.CATEGORY = ""  # empty means all categories
__C.BREAKING_BAD.ALL_CATEGORY = [
    "BeerBottle",
    "Bowl",
    "Cup",
    "DrinkingUtensil",
    "Mug",
    "Plate",
    "Spoon",
    "Teacup",
    "ToyFigure",
    "WineBottle",
    "Bottle",
    "Cookie",
    "DrinkBottle",
    "Mirror",
    "PillBottle",
    "Ring",
    "Statue",
    "Teapot",
    "Vase",
    "WineGlass",
]  # Only used for everyday

__C.BREAKING_BAD.ROT_RANGE = -1.0  # rotation range for curriculum learning
__C.BREAKING_BAD.NUM_PC_POINTS = 5000  # points per part
__C.BREAKING_BAD.MIN_PART_POINT = (
    30  # if sampled by area, want to make sure all piece have >30 points
)
__C.BREAKING_BAD.MIN_NUM_PART = 2
__C.BREAKING_BAD.MAX_NUM_PART = 20
__C.BREAKING_BAD.SHUFFLE_PARTS = False
__C.BREAKING_BAD.SAMPLE_BY = "area"

__C.BREAKING_BAD.LENGTH = -1
__C.BREAKING_BAD.TEST_LENGTH = -1
__C.BREAKING_BAD.OVERFIT = -1

__C.BREAKING_BAD.FRACTURE_LABEL_THRESHOLD = 0.025

__C.BREAKING_BAD.COLORS = [
    [0, 204, 0],
    [204, 0, 0],
    [0, 0, 204],
    [127, 127, 0],
    [127, 0, 127],
    [0, 127, 127],
    [76, 153, 0],
    [153, 0, 76],
    [76, 0, 153],
    [153, 76, 0],
    [76, 0, 153],
    [153, 0, 76],
    [204, 51, 127],
    [204, 51, 127],
    [51, 204, 127],
    [51, 127, 204],
    [127, 51, 204],
    [127, 204, 51],
    [76, 76, 178],
    [76, 178, 76],
    [178, 76, 76],
]

# custom dataset cfg
__C.CUSTOM = edict()
__C.CUSTOM.DATA_DIR = ""
__C.CUSTOM.DATA_FN = "{}_valid_pcs_list.txt"
__C.CUSTOM.NUM_PC_POINTS = 5000
__C.CUSTOM.MIN_PART_POINT = (30)
__C.CUSTOM.MIN_NUM_PART = 2
__C.CUSTOM.MAX_NUM_PART = 20
__C.CUSTOM.SHUFFLE_PARTS = False
__C.CUSTOM.ROT_RANGE = -1
__C.CUSTOM.OVERFIT = -1
__C.CUSTOM.LENGTH = -1
__C.CUSTOM.TEST_LENGTH = -1
__C.CUSTOM.FRACTURE_LABEL_THRESHOLD = 0.0025
__C.CUSTOM.CATEGORY = ""  # empty means all categories
__C.CUSTOM.ALL_CATEGORY = [
    "CYL_HOLE_out",
    "CYL_HOLE_in",
    "CUBE_HOLE_out",
    "CUBE_HOLE_in",
    "SLOT_out",
    "SLOT_in",
]  # Only used for custom dataset
__C.CUSTOM.COLORS = [
    [0, 204, 0],
    [204, 0, 0],
    [0, 0, 204],
    [127, 127, 0],
    [127, 0, 127],
    [0, 127, 127],
]
__C.CUSTOM.SUBSET = ""  # must in ['factory', 'everyday', 'other']
# 以下Key为了兼容原代码，实际不使用
__C.CUSTOM.SAMPLE_BY = "area"
__C.CUSTOM.DATA_KEYS = ("part_ids",)