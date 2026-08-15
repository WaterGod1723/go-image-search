"""Shared paths and constants for the pynet icon-retrieval experiments."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# training queries (rendered icons + manifest ground truth)
TRAIN_VIT_MANIFEST = os.path.join(ROOT, "train_set_vit", "manifest.json")
TRAIN_VIT_DIR = os.path.join(ROOT, "train_set_vit")

TRAIN_MANIFEST = os.path.join(ROOT, "train_set", "manifest.json")
TRAIN_DIR = os.path.join(ROOT, "train_set")

# reference galleries
SCRAPED_TRAIN_DIR = os.path.join(ROOT, "scraped_icons", "train")   # 2635 srcs (train domain)
SCRAPED_TEST_DIR = os.path.join(ROOT, "scraped_icons", "test")     # 1105 unseen srcs
TEST_PNGS_DIR = os.path.join(ROOT, "test_pngs")                    # 66 in-domain srcs

# evaluation sets (generated with gentest)
TEST_INDOMAIN_MANIFEST = os.path.join(ROOT, "pynet", "data", "test_indomain", "manifest.json")
TEST_INDOMAIN_DIR = os.path.join(ROOT, "pynet", "data", "test_indomain")
TEST_CROSS_MANIFEST = os.path.join(ROOT, "pynet", "data", "test_cross", "manifest.json")
TEST_CROSS_DIR = os.path.join(ROOT, "pynet", "data", "test_cross")

OUT_DIR = os.path.join(ROOT, "pynet", "out")
CACHE_DIR = os.path.join(ROOT, "pynet", "data")

SPRITE_N = 48      # sprite input resolution (48 or 64)
EMB_DIM = 64       # embedding dimension
