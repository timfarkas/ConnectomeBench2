import enum
from dataclasses import dataclass
import random
import shutil
from typing import Counter, List, Dict, Any, Optional
from glob import glob
import re
import json
import pandas as pd
import os
import numpy as np
from pathlib import Path
import time
class QuestionType(enum.Enum):
    SPLIT_VERIFICATION = 'split_verification'
    MERGE_VERIFICATION = 'merge_verification'
    SPLIT_PROPOSAL = 'split_proposal'
    JUNCTION_ERROR_IDENTIFICATION = 'junction_error_identification'
    ENDPOINT_ERROR_IDENTIFICATION = 'endpoint_error_identification'


class AnswerSpace(enum.Enum):
    YES_OR_NO = 'yes_or_no' # True or False
    SPLIT_POINTS = 'split_points' # tuple of two lists of (x, y, z) coordinates (containing a variable number of i, j points)
    ERROR_OR_CONTROL = 'error_or_control' # True or False


@dataclass
class DatasetQuestion:
    question_type: QuestionType
    answer_space: AnswerSpace
    answer: Any
    images: List[str]
    metadata: Dict[str, Any]
    sample_hash : str

    def __post_init__(self):
        self.validate()

    def validate(self):
        """
        Validate that self.answer matches the expected type for the answer_space.
        Raises:
            TypeError: if the answer type does not match the expected answer_space type.
        """
        if self.metadata is None or self.metadata == "":
            self.metadata = {}

        if self.answer_space == AnswerSpace.YES_OR_NO:
            if not isinstance(self.answer, bool):
                raise TypeError(f"For answer_space YES_OR_NO, answer must be bool, got {type(self.answer).__name__}")
        elif self.answer_space == AnswerSpace.SPLIT_POINTS:
            if not isinstance(self.answer, (tuple, list)) or len(self.answer) != 2:
                raise TypeError(f"For answer_space SPLIT_POINTS, answer must be tuple/list of (sources, sinks), got {type(self.answer).__name__}")
            sources, sinks = self.answer
            if not isinstance(sources, list) or not isinstance(sinks, list):
                raise TypeError(f"For answer_space SPLIT_POINTS, sources and sinks must be lists")
        elif self.answer_space == AnswerSpace.ERROR_OR_CONTROL:
            if not isinstance(self.answer, str) or self.answer not in ["error", "control"]:
                raise TypeError(f"For answer_space ERROR_OR_CONTROL, answer must be 'error' or 'control', got {type(self.answer).__name__}")
        else:
            raise NotImplementedError(f"Answer space {self.answer_space} not implemented")


    
    def load_geometry(self) -> Optional[np.ndarray]:
        """Load the `[V, 7, H, W]` geometry tensor tagged `"geometry"` in image_types.

        Returns None if no geometry entry is present or the file is missing.
        """
        image_types = self.metadata.get("image_types") if self.metadata else None
        if not image_types or "geometry" not in image_types:
            return None
        idx = list(image_types).index("geometry")
        if idx >= len(self.images):
            return None
        path = self.images[idx]
        if not os.path.exists(path):
            return None
        return np.load(path)

    @staticmethod
    def _convert_to_json_serializable(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, int):
            return str(obj) if abs(obj) > 2**53 - 1 else obj
        elif isinstance(obj, dict):
            return {k: DatasetQuestion._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [DatasetQuestion._convert_to_json_serializable(item) for item in obj]
        else:
            return obj

    def to_dict(self, drop_keys: List[str] = None) -> Dict[str, Any]:
        if drop_keys is None:
            drop_keys = []

        metadata = {k: v for k, v in self.metadata.items() if k not in drop_keys}

        # parquet/arrow cannot handle a struct column with *zero* fields.
        # if metadata is empty, inject a dummy key.
        metadata = self.metadata or {"__dummy__": None}

        return {
            "question_type": self.question_type.value,
            "answer_space": self.answer_space.value,
            "answer": self.answer,
            "images": self.images,
            "metadata": DatasetQuestion._convert_to_json_serializable(metadata),
            "sample_hash": self.sample_hash
        }

class QuestionDataset:
    def __init__(self, questions: List[DatasetQuestion] = None):
        self.questions : List[DatasetQuestion] = questions or []
        self._has_normalized_paths : bool = False
        self._normalized_path_dir : str = None
    
    def to_pandas(self, drop_metadata_keys: List[str] = None) -> pd.DataFrame:
        return pd.DataFrame([question.to_dict(drop_keys=drop_metadata_keys) for question in self.questions])  


    def __add__(self, other: 'QuestionDataset') -> 'QuestionDataset':
        if self._has_normalized_paths or other._has_normalized_paths:
            raise NotImplementedError("Adding datasets with normalized paths is not supported, please use merge_parquets on their parquets instead.")
        return QuestionDataset(questions=self.questions + other.questions)

    def __len__(self) -> int:
        return len(self.questions)


    def _copy_move_images(self, images_dir: str, move_images: bool = False) -> None:
        # idempotent. resolve each image path -> intended {i}_{j}_basename in
        # images_dir. files already in place stay; files in images_dir with a
        # stale {i}_{j}_ prefix get renamed in-place; files outside images_dir
        # get copied/moved in. orphans (in images_dir but not referenced by any
        # question) are removed AFTER all renames, so we never destroy a file
        # we still need.
        images_dir_abs = Path(images_dir).absolute()
        parquet_dir_abs = images_dir_abs.parent

        referenced: set = set()  # basenames we want to keep in images_dir
        new_paths_per_q: List[List[str]] = []

        for i, question in enumerate(self.questions):
            new_paths: List[str] = []
            for j, image_path in enumerate(question.images):
                p = Path(image_path)
                if p.is_absolute():
                    src = p
                else:
                    # relative paths come in two flavors:
                    # 1) stored parquet rows: "images/foo.png" (relative to parquet_dir)
                    # 2) fresh from reload_questions_from_renders: "VLM_.../_renders/.../foo.npy"
                    #    (relative to cwd, because render_dir from config is cwd-relative)
                    # try both, prefer whichever exists.
                    cwd_resolved = Path(image_path).resolve()
                    pd_resolved = (parquet_dir_abs / image_path).resolve()
                    if cwd_resolved.exists():
                        src = cwd_resolved
                    elif pd_resolved.exists():
                        src = pd_resolved
                    else:
                        src = pd_resolved  # used for error message below

                src_name = src.name
                # strip any prior {i_old}_{j_old}_ prefix to keep names stable
                stripped = re.sub(r"^\d+_\d+_", "", src_name)
                new_name = f"{i}_{j}_{stripped}"
                dst = images_dir_abs / new_name

                if dst.exists() and src.resolve() == dst.resolve():
                    pass  # already in place
                elif src.parent.resolve() == images_dir_abs.resolve():
                    # already in images_dir but with stale prefix → rename in place
                    if src.exists():
                        src.rename(dst)
                    else:
                        raise FileNotFoundError(f"image source missing: {src}")
                else:
                    if not src.exists():
                        raise FileNotFoundError(f"image source missing: {src}")
                    # prefer hardlink (same filesystem) — ~free, no data duplication.
                    # fall back to copy/move on EXDEV (cross-filesystem).
                    if dst.exists():
                        dst.unlink()
                    try:
                        os.link(str(src), str(dst))
                        if move_images:
                            src.unlink()
                    except OSError:
                        if move_images:
                            shutil.move(str(src), str(dst))
                        else:
                            shutil.copy(str(src), str(dst))

                referenced.add(new_name)
                new_paths.append(os.path.relpath(str(dst), start=str(parquet_dir_abs)))
            new_paths_per_q.append(new_paths)

        # remove orphans only AFTER all sources have been resolved
        n_removed = 0
        for f in images_dir_abs.iterdir():
            if f.is_file() and f.name not in referenced:
                f.unlink()
                n_removed += 1
        if n_removed:
            print(f"removed {n_removed} orphan files from {images_dir}")
        print(f"Finalized {len(referenced)} images at {images_dir}")

        for q, paths in zip(self.questions, new_paths_per_q):
            q.images = paths

        self._has_normalized_paths = True
        self._normalized_path_dir = images_dir_abs.as_posix()

    def to_parquet(self, path: str, move_images: bool = False, drop_metadata_keys: List[str] = None):
        """
        Convert the question dataset to a parquet file. 
        Creates a directory called 'images' in the same directory as the parquet file, and moves/copies the images to this directory.
        Args:
            path: Path to the parquet file.
            move_images: If True, move the images to the 'images' directory. If False, copy the images to the 'images' directory.
            drop_metadata_keys: List of keys to drop from the metadata.
        """
        # image dir is a directory "images" in the same directory as the parquet file
        images_dir = os.path.join(os.path.dirname(path), "images")
        os.makedirs(images_dir, exist_ok=True)
        assert os.path.dirname(path) == os.path.dirname(images_dir), "parquet_path and images_dir must be in the same directory"
        assert os.path.basename(images_dir) == "images", "images_dir must be named 'images'"

        # always run — _copy_move_images is idempotent, handles new sidecars,
        # in-place renames, and orphan cleanup safely
        self._copy_move_images(images_dir, move_images)

        # convert questions with updated image paths to df and to parquet, then return           
        self.to_pandas(drop_metadata_keys=drop_metadata_keys).to_parquet(path=path)
        print(f"Saved parquet file to {path}")
        return path
    
    @staticmethod
    def from_parquet(path: str) -> 'QuestionDataset':
        if not str(path).endswith(".parquet"):
            path = os.path.join(path, "questions.parquet")
        
        assert os.path.exists(path), f"parquet file {path} does not exist"

        print(f"Loading parquet file from {path}")
        df = pd.read_parquet(path)
        
        # check if images are present next to the parquet file
        images_dir = os.path.join(os.path.dirname(path), "images")
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"images directory {images_dir} not found")
         
        if len(df) > 0:
            def has_images_cell(cell):
                return any(isinstance(p, str) and p.startswith("images/") for p in cell)
            images_present = any(has_images_cell(cell) for cell in df["images"])
            assert (
                not images_present
                or os.path.exists(os.path.join(os.path.dirname(path), "images"))
            ), "images directory must exist if images are present in parquet file"
            
        questions = []
        for row in df.to_dict(orient='records'):
            raw_meta = row.get("metadata", {}) or {}
            if isinstance(raw_meta, str):
                raw_meta = json.loads(raw_meta)
            # strip dummy-only metadata back to {}
            if (
                isinstance(raw_meta, dict)
                and "__dummy__" in raw_meta
                and len(raw_meta) == 1
            ):
                metadata = {}
            else:
                metadata = raw_meta

            sample_hash = row["sample_hash"]

            # strip None values from evaluation_stats (pandas artifact)
            if metadata.get("evaluation_stats") is not None:
                metadata["evaluation_stats"] = {k: v for k, v in metadata["evaluation_stats"].items() if v is not None}

            # coerce answer to proper type based on answer_space
            answer = row["answer"]
            answer_space = AnswerSpace(row["answer_space"])
            if answer_space == AnswerSpace.SPLIT_POINTS:
                # parquet converts tuples to ndarrays, coerce back to tuple of lists
                if isinstance(answer, np.ndarray):
                    sources, sinks = answer[0], answer[1]
                    if isinstance(sources, np.ndarray):
                        sources = sources.tolist()
                    if isinstance(sinks, np.ndarray):
                        sinks = sinks.tolist()
                    answer = (list(sources), list(sinks))
                elif isinstance(answer, (list, tuple)):
                    sources, sinks = answer[0], answer[1]
                    answer = (list(sources), list(sinks))

            q = DatasetQuestion(
                question_type=QuestionType(row["question_type"]),
                answer_space=answer_space,
                answer=answer,
                images=json.loads(row["images"]) if isinstance(row["images"], str) else row["images"],
                metadata=metadata,
                sample_hash=sample_hash
            )
            questions.append(q)
        
        ds = QuestionDataset(questions=questions)
        ds._has_normalized_paths = True
        ds._normalized_path_dir = Path(images_dir).absolute().as_posix()
        return ds

    @staticmethod
    def _move_images(source_path: str, target_dir: str, move_images: bool = False) -> Path:
        # move/copy all images in source_path to target_path
        num_current_images_in_target_dir = len(list(glob(os.path.join(target_dir, "*.png"))))
        image_paths = list(glob(os.path.join(source_path, "*.png")))
        length = len(image_paths)
        for image_path in image_paths:
            if move_images:
                shutil.move(image_path, os.path.join(target_dir, os.path.basename(image_path)))
            else:
                shutil.copy(image_path, os.path.join(target_dir, os.path.basename(image_path)))
        assert num_current_images_in_target_dir + length == len(list(glob(os.path.join(target_dir, "*.png")))), f"Expected {num_current_images_in_target_dir + length} images, found {len(list(glob(os.path.join(target_dir, '*.png'))))} in {target_dir}"
        return Path(target_dir)

    @staticmethod
    def merge_parquets(source_directories: List[str], target_directory: str, move_images: bool = False) -> Path:
        """
        Merge multiple parquet folders into a single merged parquet folder (contains images dir and parquet file).
        Args:
            paths: List of paths to the parquet files.
        Returns:
            QuestionDataset: A question dataset containing the merged parquet files.
        """
        target_dir = target_directory
        os.makedirs(target_dir, exist_ok=True)
        os.makedirs(os.path.join(target_dir, "images"), exist_ok=True)

        assert len(glob(os.path.join(target_directory, "images", "*.png"))) == 0, f"Target directory {target_directory} is not empty, aborting."

        questions = []
        for directory in source_directories:
            assert os.path.isdir(directory), f"path {directory} is not a directory"
            image_dir = os.path.join(directory, "images")
            assert os.path.isdir(image_dir), f"images directory {image_dir} not found in {directory}"
            
            # glob match for parquet files
            parquet_paths = glob(os.path.join(directory, "*.parquet"))
            assert len(parquet_paths) == 1, f"Expected 1 parquet file, found {len(parquet_paths)} in {directory}"
            parquet_path = parquet_paths[0]

            # load parquet file
            ds = QuestionDataset.from_parquet(parquet_path)

            # move/copy images to target_dir and update question image paths to relative paths
            QuestionDataset._move_images(image_dir, os.path.join(target_dir, "images"), move_images)            
            for question in ds.questions:
                question.images = [os.path.join("images", os.path.basename(image_path)) for image_path in question.images]

            questions.extend(ds.questions)

        
        new_ds = QuestionDataset(questions=questions)
        new_ds.to_pandas().to_parquet(os.path.join(target_dir, "questions.parquet"))
        return Path(target_dir)

    def subsample(self, n: int) -> 'QuestionDataset':
        new = QuestionDataset(questions=random.sample(self.questions, n))
        new._has_normalized_paths = self._has_normalized_paths
        new._normalized_path_dir = self._normalized_path_dir
        return new

    @staticmethod
    def from_binary_splits_folder(path: str) -> 'QuestionDataset':
        assert os.path.isdir(path), f"path {path} is not a directory"

        good_dir = os.path.join(path, "good")
        bad_dir = os.path.join(path, "bad")

        assert os.path.isdir(good_dir), f"missing good/ directory at {good_dir}"
        assert os.path.isdir(bad_dir), f"missing bad/ directory at {bad_dir}"

        def load_split_dirs(root):
            # only subdirs, skip files
            dirs = [
                os.path.join(root, d)
                for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))
            ]

            out = {}
            for split_dir in dirs:
                imgs = [
                    os.path.join(split_dir, f)
                    for f in os.listdir(split_dir)
                    if f.endswith(".png")
                ]
                out[split_dir] = {
                    "images": imgs,
                    "metadata": {"split_dir": split_dir},
                }
            return out

        good_splits = load_split_dirs(good_dir)
        bad_splits = load_split_dirs(bad_dir)

        questions = []

        for split_dir, info in good_splits.items():
            questions.append(
                DatasetQuestion(
                    question_type=QuestionType.SPLIT_VERIFICATION,
                    answer_space=AnswerSpace.YES_OR_NO,
                    answer=True,
                    images=info["images"],
                    metadata=info["metadata"],
                )
            )

        for split_dir, info in bad_splits.items():
            questions.append(
                DatasetQuestion(
                    question_type=QuestionType.SPLIT_VERIFICATION,
                    answer_space=AnswerSpace.YES_OR_NO,
                    answer=False,
                    images=info["images"],
                    metadata=info["metadata"],
                )
            )

        return QuestionDataset(questions=questions)


    @staticmethod
    def from_rerendered_splits(
        images_dir: Path,
        metadata_dir: Path,
        drop_cut_edges: bool = True,
        exclude_small_splits: bool = True,
        min_candidate_size: int = 10,
    ) -> "QuestionDataset":
        assert images_dir.is_dir(), f"images_dir {images_dir} is not a directory"
        assert metadata_dir.is_dir(), f"metadata_dir {metadata_dir} is not a directory"

        print(f"Loading rerendered splits from images dir {images_dir} and metadata dir {metadata_dir}")

        views = ("front", "side", "top")

        def _candidate_size(meta: dict) -> int | None:
            try:
                return int(meta.get("evaluation_stats", {}).get("candidate_size"))
            except Exception:
                return None

        questions: list[DatasetQuestion] = []

        root_dirs = sorted(metadata_dir.iterdir(), key=lambda p: p.name)
        excluded_small_splits = 0
        excluded_missing_views = 0
        print(f"Found {len(root_dirs)} root directories")
        # iterate metadata first (require metadata to exist)
        for root_dir in root_dirs:
            if not root_dir.is_dir():
                continue

            root_id = root_dir.name
            img_root_dir = images_dir / root_id
            if not img_root_dir.is_dir():
                continue

            meta_paths = sorted(root_dir.glob("*.json"), key=lambda p: p.name)
            for meta_path in meta_paths:
                split_hash = meta_path.stem  # <split_hash>.json

                try:
                    with meta_path.open("r") as f:
                        meta = json.load(f)
                except Exception as e:
                    print(f"Skipping {meta_path} (failed to load json): {e}")
                    continue

                if exclude_small_splits:
                    cs = _candidate_size(meta)
                    if cs is not None and cs < min_candidate_size:
                        excluded_small_splits += 1
                        continue

                # require all three views to exist:
                # images_dir/<root_id>/<split_hash>_<extra>_<view>.png
                view_paths: dict[str, Path] = {}
                missing = False
                for v in views:
                    matches = sorted(img_root_dir.glob(f"{split_hash}_*_{v}.png"))
                    if len(matches) == 0:
                        missing = True
                        excluded_missing_views += 1
                        break
                    if len(matches) > 1:
                        # deterministic, but noisy
                        print(
                            f"Warning: multiple '{v}' images for root {root_id}, split {split_hash}; using {matches[0].name}"
                        )
                    view_paths[v] = matches[0]

                if missing:
                    continue

                imgs_sorted = [view_paths["front"], view_paths["side"], view_paths["top"]]

                if "is_good" not in meta:
                    continue

                answer = bool(meta["is_good"])
            

                meta_out = dict(meta)
                meta_out["root_id"] = root_id
                meta_out["split_hash"] = split_hash
                meta_out["rerender_images_dir"] = str(img_root_dir)
                meta_out["rerender_metadata_path"] = str(meta_path)

                if drop_cut_edges:
                    meta_out.pop("cut_edges", None)

                questions.append(
                    DatasetQuestion(
                        question_type=QuestionType.SPLIT_VERIFICATION,
                        answer_space=AnswerSpace.YES_OR_NO,
                        answer=answer,
                        images=[str(p) for p in imgs_sorted],
                        metadata=meta_out,
                        sample_hash=split_hash
                    )
                )
        print(f"Loaded {len(questions)} questions from rerendered splits (excluded {excluded_small_splits} small splits and {excluded_missing_views} missing views)")
        return QuestionDataset(questions=questions)

    @staticmethod
    def from_good_splits_to_split_proposal(
        images_dir: Path,
        metadata_dir: Path,
        drop_cut_edges: bool = True,
        exclude_small_splits: bool = True,
        min_candidate_size: int = 10,
        add_metadata:bool=True
    ) -> "QuestionDataset":
        """
        Create SPLIT_PROPOSAL questions from good splits only.
        Uses pre_split images and extracts sources/sinks as the answer.
        """
        assert images_dir.is_dir(), f"images_dir {images_dir} is not a directory"
        assert metadata_dir.is_dir(), f"metadata_dir {metadata_dir} is not a directory"

        print(f"Loading good splits for split proposals from images dir {images_dir} and metadata dir {metadata_dir}")

        views = ("front", "side", "top")

        def _candidate_size(meta: dict) -> int | None:
            try:
                return int(meta.get("evaluation_stats", {}).get("candidate_size"))
            except Exception:
                return None

        questions: list[DatasetQuestion] = []

        root_dirs = sorted(metadata_dir.iterdir(), key=lambda p: p.name)
        excluded_small_splits = 0
        excluded_missing_views = 0
        excluded_bad_splits = 0
        excluded_missing_split_points = 0
        excluded_missing_nodes = 0

        print(f"Found {len(root_dirs)} root directories")
        # iterate metadata first (require metadata to exist)
        for root_dir in root_dirs:
            if not root_dir.is_dir():
                continue

            root_id = root_dir.name
            img_root_dir = images_dir / root_id
            if not img_root_dir.is_dir():
                continue

            meta_paths = sorted(root_dir.glob("*.json"), key=lambda p: p.name)
            for meta_path in meta_paths:
                split_hash = meta_path.stem  # <split_hash>.json

                try:
                    with meta_path.open("r") as f:
                        meta = json.load(f)
                except Exception as e:
                    print(f"Skipping {meta_path} (failed to load json): {e}")
                    continue

                # only include good splits
                if "is_good" not in meta or not bool(meta["is_good"]):
                    excluded_bad_splits += 1
                    continue

                if exclude_small_splits:
                    cs = _candidate_size(meta)
                    if cs is not None and cs < min_candidate_size:
                        excluded_small_splits += 1
                        continue

                # extract sources and sinks for answer
                if "sources" not in meta or "sinks" not in meta:
                    excluded_missing_split_points += 1
                    continue

                sources = meta["sources"]
                sinks = meta["sinks"]
                answer = (sources, sinks)

                # only match pre_split_ images
                # require all three views to exist:
                # images_dir/<root_id>/<split_hash>_*_pre_split_*_<view>.png
                view_paths: dict[str, Path] = {}
                missing = False
                for v in views:
                    matches = sorted(img_root_dir.glob(f"{split_hash}_*_pre_split_*_{v}.png"))
                    if len(matches) == 0:
                        missing = True
                        excluded_missing_views += 1
                        break
                    if len(matches) > 1:
                        # deterministic, but noisy
                        print(
                            f"Warning: multiple '{v}' pre_split images for root {root_id}, split {split_hash}; using {matches[0].name}"
                        )
                    view_paths[v] = matches[0]

                if missing:
                    continue

                # try to load neighbor metadata (optional but nice to have)
                neighbor_metadata = None
                try:
                    neighbors_matches = list(img_root_dir.glob(f"{split_hash}_*_nodes.json"))
                    if len(neighbors_matches) > 0:
                        with neighbors_matches[0].open("r") as f:
                            neighbor_metadata = json.load(f)
                except Exception as e:
                    excluded_missing_nodes += 1
                    # not critical, just skip

                imgs_sorted = [view_paths["front"], view_paths["side"], view_paths["top"]]

                meta_out = dict(meta)
                meta_out["root_id"] = root_id
                meta_out["split_hash"] = split_hash
                meta_out["rerender_images_dir"] = str(img_root_dir)
                meta_out["rerender_metadata_path"] = str(meta_path)
                if neighbor_metadata is not None:
                    meta_out["neighbors"] = neighbor_metadata

                if drop_cut_edges:
                    meta_out.pop("cut_edges", None)

                questions.append(
                    DatasetQuestion(
                        question_type=QuestionType.SPLIT_PROPOSAL,
                        answer_space=AnswerSpace.SPLIT_POINTS,
                        answer=answer,
                        images=[str(p) for p in imgs_sorted],
                        metadata=meta_out if add_metadata else None,
                        sample_hash=meta_out["split_hash"]
                    )
                )

        print(f"Loaded {len(questions)} split proposal questions from good splits")
        print(f"  Excluded: {excluded_bad_splits} bad splits, {excluded_small_splits} small splits, "
              f"{excluded_missing_views} missing views, {excluded_missing_split_points} missing split points, "
              f"{excluded_missing_nodes} missing nodes (non-critical)")
        return QuestionDataset(questions=questions)


    @staticmethod
    def from_binary_merge_identification_folder(path: str) -> 'QuestionDataset':
        """
        Generates a question dataset from a folder of binary merge identification directories as generated by split_merge_resolution.py, with the task setting --task merge-identification.
        Requires the updated generation script that sets first_one_correct to True.

        Args:
            path: Path to the folder of merge identification directories.
        Returns:
            QuestionDataset: A question dataset containing the merge identification directories.
        """
        assert os.path.isdir(path), f"path {path} is not a directory"

        merge_directories = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

        import json

        questions = []

        for merge_directory in merge_directories:
            try:
                with open(os.path.join(merge_directory, "generation_metadata.json"), "r") as f:
                    metadata = json.load(f)
            except Exception as e:
                print(f"Error loading metadata from {merge_directory}: {e}, skipping")
                continue
            
            if not metadata["first_one_correct"]:
                raise NotImplementedError("Only first one correct merge identification is supported, please update your merge generation script")
            
            if not len(metadata["incorrect_ids"]) == 1:
                print(f"Only one incorrect id is supported, found {len(metadata['incorrect_ids'])} ({merge_directory}), skipping")
                continue

            correct_id = metadata["correct_id"]
            incorrect_id = metadata["incorrect_ids"][0]

            correct_views_paths = []
            incorrect_views_paths = []

            correct_images_dict = metadata["image_paths"]["options"][correct_id]["zoomed"]
            incorrect_images_dict = metadata["image_paths"]["options"][incorrect_id]["zoomed"]

            for view, image_path in correct_images_dict.items():
                assert os.path.exists(image_path), f"image path {image_path} does not exist"
                correct_views_paths.append(image_path)

            for view, image_path in incorrect_images_dict.items():
                assert os.path.exists(image_path), f"image path {image_path} does not exist"
                incorrect_views_paths.append(image_path)

            # remove image_paths from metadata for cleanliness
            metadata["image_paths"] = None

            correct_question = DatasetQuestion(
                question_type=QuestionType.MERGE_VERIFICATION,
                answer_space=AnswerSpace.YES_OR_NO,
                answer=True,
                images=correct_views_paths,
                metadata=metadata,
            )

            incorrect_question = DatasetQuestion(
                question_type=QuestionType.MERGE_VERIFICATION,
                answer_space=AnswerSpace.YES_OR_NO,
                answer=False,
                images=incorrect_views_paths,
                metadata=metadata,
            )

            questions.append(correct_question)
            questions.append(incorrect_question)

        return QuestionDataset(questions=questions)
    


    @staticmethod
    def from_binary_error_identification_folder(path: str, question_type: 'QuestionType' = None) -> 'QuestionDataset':
        """
        Load error identification samples from a folder with error/ and control/ subdirectories.

        Expected structure:
            path/
            ├── error/<sample_id>/
            │   ├── *_{front,side,top}.png
            │   └── metadata.json
            └── control/<sample_id>/
                ├── *_{front,side,top}.png
                └── metadata.json

        Args:
            path: Path to the folder containing error/ and control/ subdirectories.
            question_type: QuestionType to stamp on each question. Required.
        Returns:
            QuestionDataset: A question dataset for error identification.
        """
        assert question_type is not None, "question_type is required"
        import hashlib

        path = Path(path)
        assert path.is_dir(), f"path {path} is not a directory"

        error_dir = path / "error"
        control_dir = path / "control"

        questions = []
        views = ("front", "side", "top")

        def process_samples(sample_root: Path, is_error: bool):
            if not sample_root.is_dir():
                print(f"Warning: {sample_root} does not exist, skipping")
                return

            sample_dirs = sorted([d for d in sample_root.iterdir() if d.is_dir()])
            skipped = 0

            for sample_dir in sample_dirs:
                meta_path = sample_dir / "metadata.json"
                if not meta_path.exists():
                    skipped += 1
                    continue

                try:
                    with meta_path.open("r") as f:
                        meta = json.load(f)
                except Exception as e:
                    print(f"Skipping {sample_dir} (failed to load json): {e}")
                    skipped += 1
                    continue

                # find images - try from metadata first, then glob
                if "images" in meta and meta["images"]:
                    img_paths = [Path(p) for p in meta["images"]]
                    # verify they exist
                    if not all(p.exists() for p in img_paths):
                        # fall back to glob
                        img_paths = None
                else:
                    img_paths = None

                if img_paths is None:
                    # glob for images
                    prefix = "junction_error" if is_error else "junction_control"
                    view_paths = {}
                    missing = False
                    for v in views:
                        matches = sorted(sample_dir.glob(f"{prefix}_*_{v}.png"))
                        if not matches:
                            matches = sorted(sample_dir.glob(f"*_{v}.png"))
                        if not matches:
                            missing = True
                            break
                        view_paths[v] = matches[0]

                    if missing:
                        skipped += 1
                        continue

                    img_paths = [view_paths["front"], view_paths["side"], view_paths["top"]]

                # generate sample hash from sample_dir name
                sample_id = sample_dir.name
                sample_hash = hashlib.md5(sample_id.encode()).hexdigest()[:12]

                answer = "error" if is_error else "control"

                meta_out = dict(meta)
                meta_out["sample_dir"] = str(sample_dir)

                questions.append(
                    DatasetQuestion(
                        question_type=question_type,
                        answer_space=AnswerSpace.ERROR_OR_CONTROL,
                        answer=answer,
                        images=[str(p) for p in img_paths],
                        metadata=meta_out,
                        sample_hash=sample_hash
                    )
                )

            label = "error" if is_error else "control"
            print(f"Loaded {len(sample_dirs) - skipped} {label} samples (skipped {skipped})")

        process_samples(error_dir, is_error=True)
        process_samples(control_dir, is_error=False)

        print(f"Total: {len(questions)} error identification samples ({question_type.value})")
        return QuestionDataset(questions=questions)


    @staticmethod
    def from_merge_correction_folder(path: str) -> 'QuestionDataset':
        """
        Load merge correction samples from a folder with good/ and bad/ subdirectories,
        as generated by merge_sampler.py.

        Expected structure:
            path/
            ├── good/<operation_id>_<seg1>_<seg2>/
            │   ├── *_{front,side,top}.png
            │   └── metadata.json
            └── bad/<operation_id>_<seg1>_<seg2>/
                ├── *_{front,side,top}.png
                └── metadata.json

        Args:
            path: Path to the folder containing good/ and bad/ subdirectories.
        Returns:
            QuestionDataset: MERGE_VERIFICATION / YES_OR_NO questions.
        """
        import hashlib

        path = Path(path)
        assert path.is_dir(), f"path {path} is not a directory"

        good_dir = path / "good"
        bad_dir = path / "bad"

        questions = []
        views = ("front", "side", "top")

        def process_samples(sample_root: Path, is_good: bool):
            if not sample_root.is_dir():
                print(f"Warning: {sample_root} does not exist, skipping")
                return

            sample_dirs = sorted([d for d in sample_root.iterdir() if d.is_dir()])
            skipped = 0

            for sample_dir in sample_dirs:
                meta_path = sample_dir / "metadata.json"
                if not meta_path.exists():
                    skipped += 1
                    continue

                try:
                    with meta_path.open("r") as f:
                        meta = json.load(f)
                except Exception as e:
                    print(f"Skipping {sample_dir} (failed to load json): {e}")
                    skipped += 1
                    continue

                # find images from metadata or glob
                if "images" in meta and meta["images"]:
                    img_paths = [Path(p) for p in meta["images"]]
                    if not all(p.exists() for p in img_paths):
                        img_paths = None
                else:
                    img_paths = None

                if img_paths is None:
                    view_paths = {}
                    missing = False
                    for v in views:
                        matches = sorted(sample_dir.glob(f"*_{v}.png"))
                        if not matches:
                            missing = True
                            break
                        view_paths[v] = matches[0]

                    if missing:
                        skipped += 1
                        continue

                    img_paths = [view_paths["front"], view_paths["side"], view_paths["top"]]

                sample_id = sample_dir.name
                sample_hash = hashlib.md5(sample_id.encode()).hexdigest()[:12]

                meta_out = dict(meta)
                meta_out["sample_dir"] = str(sample_dir)

                questions.append(
                    DatasetQuestion(
                        question_type=QuestionType.MERGE_VERIFICATION,
                        answer_space=AnswerSpace.YES_OR_NO,
                        answer=is_good,
                        images=[str(p) for p in img_paths],
                        metadata=meta_out,
                        sample_hash=sample_hash
                    )
                )

            label = "good" if is_good else "bad"
            print(f"Loaded {len(sample_dirs) - skipped} {label} samples (skipped {skipped})")

        process_samples(good_dir, is_good=True)
        process_samples(bad_dir, is_good=False)

        print(f"Total: {len(questions)} merge correction samples")
        return QuestionDataset(questions=questions)


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert binary splits to parquet format."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        nargs='+',
        help="Input directory or directories containing binary splits (and metadata)."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Output path for the generated parquet file."
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="Mode to run the script in.",
        choices=["split", "merge", "merge_correction", "split_proposal", "junction_error_identification", "endpoint_error_identification"],
    )
    parser.add_argument(
        "--move-images",
        action="store_true",
        help="Move images to the output directory instead of copying them."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing parquet file if it exists."
    )
    args = parser.parse_args()

    start_time = time.time()
    input_paths = [Path(p) for p in args.input_path]
    output_path = Path(args.output_path)
    print(f"Input paths: {input_paths}")
    print(f"Output path: {output_path}")
    print(f"Mode: {args.mode}")
    print(f"Move images: {args.move_images}")

    for p in input_paths:
        if not os.path.isdir(p):
            parser.error(f"Input path {p} is not a directory")
    if not os.path.isdir(output_path.parent):
        parser.error(f"Output parent directory {output_path.parent} does not exist")


    if args.overwrite and os.path.exists(output_path):
        print(f"Overwriting existing parquet file at {output_path}")
        os.remove(output_path)

    if not os.path.exists(output_path):
        if args.mode == "merge":
            assert len(input_paths) == 1, f"Expected 1 input path for merge mode, got {len(input_paths)}"
            ds = QuestionDataset.from_binary_merge_identification_folder(input_paths[0])
        elif args.mode == "merge_correction":
            assert len(input_paths) == 1, f"Expected 1 input path for merge_correction mode, got {len(input_paths)}"
            ds = QuestionDataset.from_merge_correction_folder(str(input_paths[0]))
        elif args.mode in ["split", "junction_error_correction"]:
            assert len(input_paths) == 2, f"Expected 2 input paths for split mode (images_dir, metadata_dir), got {len(input_paths)}"
            images_path = Path(input_paths[0])
            metadata_path = Path(input_paths[1])
            assert os.path.isdir(images_path), f"images path {images_path} is not a directory"
            assert os.path.isdir(metadata_path), f"metadata path {metadata_path} is not a directory"
            ds = QuestionDataset.from_rerendered_splits(Path(images_path), Path(metadata_path), drop_cut_edges=True)
        elif args.mode == "split_proposal":
            assert len(input_paths) == 2, f"Expected 2 input paths for split_proposal mode (images_dir, metadata_dir), got {len(input_paths)}"
            images_path = Path(input_paths[0])
            metadata_path = Path(input_paths[1])
            assert os.path.isdir(images_path), f"images path {images_path} is not a directory"
            assert os.path.isdir(metadata_path), f"metadata path {metadata_path} is not a directory"
            ds = QuestionDataset.from_good_splits_to_split_proposal(
                Path(images_path),
                Path(metadata_path),
                drop_cut_edges=True,
                exclude_small_splits=True,
                min_candidate_size=10
            )
            for q in ds.questions:
                q.metadata["evaluation_stats"] = {"None": None}
        elif args.mode == "junction_error_identification":
            assert len(input_paths) == 1, f"Expected 1 input path for junction_error_identification mode, got {len(input_paths)}"
            ds = QuestionDataset.from_binary_error_identification_folder(str(input_paths[0]), question_type=QuestionType.JUNCTION_ERROR_IDENTIFICATION)
        elif args.mode == "endpoint_error_identification":
            assert len(input_paths) == 1, f"Expected 1 input path for endpoint_error_identification mode, got {len(input_paths)}"
            ds = QuestionDataset.from_binary_error_identification_folder(str(input_paths[0]), question_type=QuestionType.ENDPOINT_ERROR_IDENTIFICATION)
        else:
            parser.error(f"Invalid mode provided: {args.mode}.")
        ds.to_parquet(output_path, move_images=args.move_images, drop_metadata_keys=["cut_edges"])
    else:
        print(f"Splits parquet file already exists at {output_path}, skipping conversion")

    print(f"Done generating parquet file at {output_path}.")
    elapsed = int(time.time() - start_time)
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Time taken: {hours}:{minutes:02d}:{seconds:02d}.") 
    print("\n")

    print(f"Checking parquet file...")
    print("="*100)
    ds = QuestionDataset.from_parquet(output_path)
    print(f"Loaded {len(ds.questions)} questions from parquet file at {output_path}.")
    print("="*100)
    if ds.questions:
        print(f"First question: {ds.questions[0]}")
        print(f"Normalized path dir: {ds._normalized_path_dir}")
        print("="*100)
        answer_counter = Counter()
        for question in ds.questions:
            try:
                answer_counter[question.answer] += 1
            except TypeError:
                answer_counter[str(question.answer)] += 1
        print(answer_counter)
    else:
        print("WARNING: 0 questions generated — check upstream sampler/renderer logs")
    print("="*100)
    answer_counter = Counter()
    for question in ds.questions:
        try:
            answer_counter[question.answer] += 1
        except TypeError:
            answer_counter[str(question.answer)] += 1
    print(answer_counter)
    print("="*100)


    
