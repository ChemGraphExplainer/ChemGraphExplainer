import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter import PhotoImage
import threading
import sys
import subprocess
import os
from omegaconf import OmegaConf

import time  
import yaml
import torch
import hydra
from tqdm import tqdm

from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import numpy 
from torch import Tensor
from typing import List, Dict, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem
import tempfile, base64, zlib



sys.path.append(r'C:/Users/alica/DIG')

from benchmarks.xgraph.graph_visualize import RDKitMoleculeDrawer
from benchmarks.xgraph.utils import fix_random_seed
from benchmarks.xgraph.gnnNets import get_gnnNets
from benchmarks.xgraph.gnnNets2 import get_gnnNets2
from benchmarks.xgraph.dataset import get_dataset, get_dataloader, SynGraphDataset
from benchmarks.xgraph.utils import check_dir, fix_random_seed, Recorder, perturb_input
from benchmarks.xgraph.createdata import create_data

from dig.xgraph.method import GNN_GI
from dig.xgraph.method import GNN_LRP
from dig.xgraph.method import GradCAM
from dig.xgraph.method import GNNExplainer
from dig.xgraph.method import DeepLIFT
from dig.xgraph.method import PGExplainer
from dig.xgraph.method import SubgraphX
from benchmarks.xgraph.pgexplainer_edges import PGExplainer_edges
from dig.xgraph.evaluation import XCollector
from dig.xgraph.utils.compatibility import compatible_state_dict
from torch_geometric.utils import add_remaining_self_loops
import cirpy




start_time = time.time()

root = tk.Tk()

root.title("ChemGraph Explainer")
root.geometry("850x500")
root.resizable(width=False, height=False)
root.configure(background='lightgray')



dataset_txt = ''
method_txt = ''
smiles = ''
model_txt = ''

def button_clickd(buttond_text):
    global dataset_txt
   
    for buttond in dataset_buttons:
        if buttond["text"] == buttond_text:
            buttond.config(bg="lightyellow2")  # Seçilen düğmenin rengini gri yap
            buttond.config(state="disabled")  # Seçilen düğmeyi devre dışı bırak
            if buttond_text == "HIV":
                dataset_txt = "hiv"
            elif buttond_text == "BBBP":
                dataset_txt = "bbbp"
            elif buttond_text == "BACE":
                dataset_txt = "bace"
            elif buttond_text == "MUTAG":
                dataset_txt = "mutag"
        else:
            buttond.config(bg="white")  # Diğer düğmelerin rengini varsayılan yap
            buttond.config(state="normal")  # Diğer düğmeleri etkinleştir
           
def button_clickmethods(buttonm_text):
    global method_txt
   
    for buttonm in methods_buttons:
        if buttonm["text"] == buttonm_text:
            buttonm.config(bg="lightyellow2")  # Seçilen düğmenin rengini gri yap
            buttonm.config(state="disabled")  # Seçilen düğmeyi devre dışı bırak
            if buttonm_text == "GradCAM":
                method_txt = "grad_cam"
            elif buttonm_text == "GNN-LRP":
                method_txt = "gnn_lrp"
            elif buttonm_text == "GNN-GI":
                method_txt = "gnn_gi"
            elif buttonm_text == "GNNExplainer":
                method_txt = "gnn_explainer"
            elif buttonm_text == "DeepLift":
                method_txt = "deep_lift"
            elif buttonm_text == "SubgraphX":
                method_txt = "subgraphx"
               
        else:
            buttonm.config(bg="white")  # Diğer düğmelerin rengini varsayılan yap
            buttonm.config(state="normal")

def button_clickmodels(button_model_text):
    global model_txt
   
    for button_model in models_buttons:
        if button_model["text"] == button_model_text:
            button_model.config(bg="lightyellow2")
            button_model.config(state="disabled")
            if button_model_text == "GCN":
                model_txt = "gcn"
            elif button_model_text == "GAT":
                model_txt = "gat"
            elif button_model_text == "GIN":
                model_txt = "gin"
            #model_txt = button_model_text
           
        else:
            button_model.config(bg="white")
            button_model.config(state="normal")

def get_input():
    user_input = entry.get()
    #user_input = entry_smiles.get()
    print("Entered text:", user_input)

def show_results():
    # Sonuçları görüntülemek için bir işlev
    #print("Show results button clicked!")
    with open("C:/Users/alica/DIG/benchmarks/xgraph/config/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    if 'defaults' in config and isinstance(config['defaults'], list) and len(config['defaults']) > 0:
        # Listenin her öğesini kontrol et
        for item in config['defaults']:
            # Eğer öğe içinde 'datasets' anahtarı varsa, değeri 'hiv' olarak değiştir
            if 'datasets' in item:
                #print("dataset_txt = ", dataset_txt)
                item['datasets'] = dataset_txt
                continue
            if 'explainers' in item:
                #print("dataset_txt = ", dataset_txt)
                item['explainers'] = method_txt
                continue
            if 'models' in item:
                #print("dataset_txt = ", dataset_txt)
                item['models'] = model_txt
                continue
    if 'smiles' in config:
        # config['smiles'] = entry.get() asdsa
        if entry.get() and entry.get() != "Enter SMILES":
            config['smiles'] = entry.get()
        else:
            smiles = cirpy.resolve(entry_second.get(), 'smiles')
            config['smiles'] = smiles
        #config['smiles'] = entry_smiles.get()
    with open("C:/Users/alica/DIG/benchmarks/xgraph/config/config.yaml", "w") as file:
        yaml.dump(config, file)
       
        ####
   
    if method_txt == 'pgexplainer_edges':
        path_sparsity = "C:/Users/alica/DIG/benchmarks/xgraph/config/explainers/pgexplainer.yaml"
    else:
        path_sparsity = "C:/Users/alica/DIG/benchmarks/xgraph/config/explainers/"+method_txt+".yaml"
       
    with open(path_sparsity, "r") as file:
       
        sparsity = yaml.safe_load(file)
   
    if 'sparsity' in sparsity:
        try:
            sparsity_value = float(entry_sparsity.get())
            sparsity['sparsity'] = sparsity_value
        except ValueError:
            pass
       
    with open(path_sparsity, "w") as file:
        yaml.dump(sparsity, file)
       
        ###
       
    if method_txt == 'gnn_gi':    
        exec(open("C:/Users/alica/DIG/benchmarks/xgraph/gnn_gi.py").read())
    elif method_txt == 'gnn_lrp':
        exec(open("C:/Users/alica/DIG/benchmarks/xgraph/gnn_lrp.py").read())
    elif method_txt == 'grad_cam':
        exec(open("C:/Users/alica/DIG/benchmarks/xgraph/grad_cam.py").read())
    elif method_txt == 'gnn_explainer':
        exec(open("C:/Users/alica/DIG/benchmarks/xgraph/gnn_explainer.py").read())
    elif method_txt == 'deep_lift':
        exec(open("C:/Users/alica/DIG/benchmarks/xgraph/deep_lift.py").read())
    elif method_txt == 'pgexplainer_edges':
        exec(open("C:/Users/alica/DIG/benchmarks/xgraph/pgexplainer.py").read())

#%% dataset buttons


buttons_frame = tk.Frame(root, bd=2, relief="groove", padx=10, pady=13)
buttons_frame.grid(row=1, column=0)
buttons_frame.configure(background='lightgray')

dataset_button_texts = ["HIV", "BBBP", "BACE", "MUTAG","CLINTOX", "ESOL", "FREESOLV", "LIPO", "MUV", "PCBA", "SIDER", "TOX21", "TOXCAST"]
dataset_buttons = []
for index, text in enumerate(dataset_button_texts):
    button = tk.Button(buttons_frame, text=text, command=lambda t=text: button_clickd(t))
    button.grid(row=1, column=index, padx=5, pady=5)
    dataset_buttons.append(button)
   
frame_label = tk.Label(buttons_frame, text="Datasets", font=("Helvetica", 7, "bold"))
frame_label.grid(row=0, columnspan=len(dataset_button_texts), sticky="w", padx=5, pady=5)  

#%% methods buttons

methods_frame = tk.Frame(root, bd=2, relief="groove", padx=10, pady=13)
methods_frame.grid(row=2, column=0)
methods_frame.configure(background='lightgray')

methods_button_texts = ["GradCAM", "GNN-LRP", "GNN-GI", "GNNExplainer", "DeepLift", "SubgraphX"]
methods_buttons = []
for index, text in enumerate(methods_button_texts):
    button = tk.Button(methods_frame, text=text, command=lambda t=text: button_clickmethods(t))
    button.grid(row=1, column=index, padx=5, pady=5)
    methods_buttons.append(button)

# Metotlar başlığı
methods_label = tk.Label(methods_frame, text="Methods", font=("Helvetica", 7, "bold"))
methods_label.grid(row=0, columnspan=len(methods_button_texts), sticky="w", padx=5, pady=5)

#%% Models buttons
#%%

models_frame = tk.Frame(root, bd=2, relief="groove", padx=10, pady=13)
models_frame.grid(row=3, column=0)
models_frame.configure(background='lightgray')

models_button_texts = ["GCN", "GIN", "GAT"]
models_buttons = []

for index, text in enumerate(models_button_texts):
    button = tk.Button(models_frame, text=text, command=lambda t=text: button_clickmodels(t))
    button.grid(row=1, column=index, padx=5, pady=5)
    models_buttons.append(button)

# Modeller başlığı
models_label = tk.Label(models_frame, text="Models", font=("Helvetica", 7, "bold"))
models_label.grid(row=0, columnspan=len(models_button_texts), sticky="w", padx=5, pady=5)


#%% entry part
#%% entry smiles

entry = tk.Entry(root, width=50)
entry.insert(0, "Enter SMILES")

def on_entry_click(event):
    if entry.get() == "Enter SMILES":
        entry.delete(0, "end")


entry.bind("<FocusIn>", on_entry_click)

entry.grid(row=5, column=0, sticky="w", padx=(50, 0), pady=5)

#%% İkinci giriş bloğu (İlk girişin altına)
entry_second = tk.Entry(root, width=50)
entry_second.insert(0, "Enter IUPAC")

def on_second_entry_click(event):
    if entry_second.get() == "Enter IUPAC":
        entry_second.delete(0, "end")

entry_second.bind("<FocusIn>", on_second_entry_click)
entry_second.grid(row=6, column=0, sticky="w", padx=(50, 0), pady=5)

#%% entry sparsity
entry_sparsity = tk.Entry(root, width=20)
entry_sparsity.insert(0, "Enter Sparsity")


def on_entry_click2(event):
    if entry_sparsity.get() == "Enter Sparsity":
        entry_sparsity.delete(0, "end")


entry_sparsity.bind("<FocusIn>", on_entry_click2)

entry_sparsity.grid(row=5, column=0, sticky="w", padx=(500, 0), pady=5)


#%% results button

results_button = tk.Button(root, text="Results", command=show_results)
results_button.grid(row=7, column=0, pady=10)


entry_gnn_path = tk.Entry(root, width=50)
entry_gnn_path.insert(0, "Custom GNN Path")

def on_gnn_path_entry_click(event):
    if entry_gnn_path.get() == "Custom GNN Path":
        entry_gnn_path.delete(0, "end")

entry_gnn_path.bind("<FocusIn>", on_gnn_path_entry_click)
entry_gnn_path.grid(row=8, column=0, sticky="w", padx=(50, 0), pady=5)


train_button = tk.Button(root, text="Train")
train_button.grid(row=8, column=0, sticky="w", padx=(500, 0), pady=5)

root.mainloop()
