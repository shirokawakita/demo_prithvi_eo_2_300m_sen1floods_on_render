import streamlit as st
import torch
import yaml
import os
from huggingface_hub import hf_hub_download
from pathlib import Path

class PrithviModelLoader:
    def __init__(self):
        self.repo_id = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11"
        self.model_filename = "Prithvi-EO-V2-300M-TL-Sen1Floods11.pt"
        self.config_filename = "config.yaml"
        self.cache_dir = Path("/tmp/prithvi_cache")
        self.cache_dir.mkdir(exist_ok=True)
    
    @st.cache_resource
    def download_model(_self):
        """Hugging Face HubからPrithviモデルをダウンロード"""
        try:
            with st.spinner("Prithviモデルをダウンロード中... (約1.28GB)"):
                # モデルファイルをダウンロード
                model_path = hf_hub_download(
                    repo_id=_self.repo_id,
                    filename=_self.model_filename,
                    cache_dir=str(_self.cache_dir)
                )
                
                # 設定ファイルをダウンロード
                config_path = hf_hub_download(
                    repo_id=_self.repo_id,
                    filename=_self.config_filename,
                    cache_dir=str(_self.cache_dir)
                )
                
                return model_path, config_path
        except Exception as e:
            st.error(f"モデルのダウンロードに失敗しました: {e}")
            return None, None
    
    @st.cache_resource
    def load_model(_self):
        """モデルを読み込み"""
        model_path, config_path = _self.download_model()
        
        if model_path is None or config_path is None:
            return None, None
        
        try:
            # 設定ファイル読み込み
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # モデル読み込み
            device = torch.device('cpu')  # Renderではcpu使用
            model = torch.load(model_path, map_location=device)
            model.eval()
            
            return model, config
        except Exception as e:
            st.error(f"モデルの読み込みに失敗しました: {e}")
            return None, None