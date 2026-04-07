import pandas as pd

from core.pipeline.inferencer import MorpheInferencer
from core.pipeline.trainer import MorpheTrainer
from constants import InferenceMode


df = pd.read_csv(
    "/Users/kevl0215/Documents/codex_data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv"
)
feature_cols = ['CD34', 'CD38', 'Cytokeratin', 'CD19', 'CD4', 'CD49f', 'aSMA', 'CD161', 'CD163', 'NKG2D', 'CD16', 'CD49a', 'CD138',  'CD8', 'CD206',  'CD117', 'CD36', 'CD7', 'SOX9', 'CD68', 'CD57', 'Podoplanin', 'CD44', 'CD56', 'CD31', 'MUC1', 'aDef5', 'CD3', 'Synapto', 'ITLN1', 'Vimentin', 'CD15', 'CD11c', 'CD45', 'CD66', 'HLADR', 'Ki67', 'CD21',  'CD90', 'CHGA', 'CD123']
inference_mode = InferenceMode.GAPFILL
dir = "/Users/kevl0215/Documents/codex_data"

artifact = MorpheTrainer.train(
    df=df, 
    root_dir=dir, 
    feature_cols=feature_cols, 
    inference_mode=inference_mode, 
    pd_precomputer_out_dir=dir,
    pd_precomputer_root_dir=dir,
    verbose=True
)

MorpheInferencer.from_artifact(artifact=artifact).run_gapfill(
    df=df,
    root_dir=dir,
    input_dir=dir,
    save_dir=dir
)