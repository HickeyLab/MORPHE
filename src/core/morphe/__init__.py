# # disco/core/disco/__init__.py
# from __future__ import annotations

# from typing import Any

# from .artifact import DiscoArtifact
# from .config import DiscoConfig
# from .inferencer import DiscoInferencer
# from .trainer import DiscoTrainer


# class Disco:
#     """
#     High-level public facade for DISCO.
#     """

#     def __init__(self, inferencer: DiscoInferencer) -> None:
#         self._inferencer = inferencer

#     # -------------------------
#     # Training entry
#     # -------------------------

#     @classmethod
#     def fit(cls, *, data: Any, cfg: DiscoConfig, **kwargs: Any) -> DiscoArtifact:
#         return DiscoTrainer.fit(data=data, cfg=cfg, **kwargs)

#     # -------------------------
#     # Inference entry
#     # -------------------------

#     @classmethod
#     def from_artifact(
#         cls,
#         artifact: DiscoArtifact,
#         *,
#         device: str = "cpu",
#         dtype=None,
#         **build_kwargs: Any,
#     ) -> "Disco":
#         inferencer = artifact.build_inferencer(device=device, dtype=dtype, **build_kwargs)
#         return cls(inferencer)

#     # -------------------------
#     # Task API (ONLY public methods)
#     # -------------------------

#     def inpaint(self, *, strategy: Any, **kwargs: Any) -> Any:
#         return self._inferencer.inpaint(strategy=strategy, **kwargs)

#     def outpaint(self, *, strategy: Any, **kwargs: Any) -> Any:
#         return self._inferencer.outpaint(strategy=strategy, **kwargs)

#     def impute_2d(self, *, strategy: Any, **kwargs: Any) -> Any:
#         return self._inferencer.impute_2d(strategy=strategy, **kwargs)

#     def impute_3d(self, *, strategy: Any, **kwargs: Any) -> Any:
#         return self._inferencer.impute_3d(strategy=strategy, **kwargs)