from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any


@dataclass
class ModelParams:
    dense_units: int = 512
    freeze_backbone: bool = True
    dropout_rate: float = 0.5
    saved_model_name = "model.keras"


@dataclass
class TrainingParams:
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = "adam"
    loss_function: str = "binary_crossentropy"
    training_name: str = "training_name"


@dataclass
class Callback:
    type: str
    args: Dict[str, any] = field(default_factory=dict)

    def to_dict(self):
        """Convert the Callback to a dictionary format."""
        return asdict(self)


@dataclass
class Paths:
    subsets_path: str = (
        r"C:\Users\user\Projekty\BaldOrNot\src\data\list_eval_partition.csv"
    )
    labels_path: str = (
        r"C:\Users\user\Projekty\BaldOrNot\src\data\list_attr_celeba.csv"
    )
    images_dir: str = r"C:\Users\user\Projekty\BaldOrNot\src\data\img_align_celeba\img_align_celeba"


@dataclass
class BaldOrNotConfig:
    model_params: ModelParams = field(default_factory=lambda: ModelParams())
    training_params: TrainingParams = field(
        default_factory=lambda: TrainingParams()
    )
    callbacks: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            Callback(
                type="EarlyStopping",
                args={"monitor": "val_loss", "patience": 5},
            ).to_dict(),
            Callback(
                type="TensorBoard",
                args={"log_dir": "logs", "histogram_freq": 1},
            ).to_dict(),
        ]
    )
    metrics: List[str] = field(default_factory=lambda: ["accuracy"])
    paths: Paths = field(default_factory=lambda: Paths())

    def __post_init__(self):
        self.model_params = (
            ModelParams(**self.model_params)
            if isinstance(self.model_params, dict)
            else self.model_params
        )
        self.training_params = (
            TrainingParams(**self.training_params)
            if isinstance(self.training_params, dict)
            else self.training_params
        )
        self.callbacks = [
            Callback(**params).to_dict()
            if isinstance(params, dict)
            else params
            for params in self.callbacks
        ]
        for callback in self.callbacks:
            if "args" not in callback:
                callback["args"] = {}
        self.paths = (
            Paths(**self.paths) if isinstance(self.paths, dict) else self.paths
        )


@dataclass
class PredictParams:
    model_path: str = r"C:\Users\user\Projekty\BaldOrNot\trainings\training_name2024-10-14_23-34-02\model.keras"
    image_dir_path = (
        r"C:\Users\user\Projekty\BaldOrNot\scrapping\downloaded_images"
    )
    image_name = "image_89.jpg"


@dataclass
class PredictConfig:
    predict_params: PredictParams = field(
        default_factory=lambda: PredictParams()
    )

    def __post_init__(self):
        self.predict_params = (
            PredictParams(**self.predict_params)
            if isinstance(self.predict_params, dict)
            else self.predict_params
        )
