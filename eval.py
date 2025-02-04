from comet_ml import Experiment

import os
import random
import yaml
import argparse
from tqdm import tqdm
import pdb
from collections import defaultdict


import torch
torch.autograd.set_detect_anomaly(True)
import pandas as pd
from yaml_config_override import add_arguments

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

from transformers import get_linear_schedule_with_warmup

from models.ssl_classification_model import SSLClassificationModel
from datasets.audio_classification_dataset import AudioClassificationDataset

from yaml_config_override import add_arguments
from addict import Dict
from collections import defaultdict

import numpy as np
from train import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def compute_metrics_by_speaker(spk_metrics, spk_ids, reference, predictions, verbose=False, is_binary_classification=False):
    
    if len(spk_ids) != len(reference) or len(spk_ids) != len(predictions):
        print("Lengths of spkids, reference, and predictions do not match")
        return None
    # print(spk_ids, reference, predictions)
    spk_dict = defaultdict(lambda: {"reference": [], "prediction": []})
    for s, r, p in zip(spk_ids, reference, predictions):
        spk_dict[s]["reference"].append(r)
        spk_dict[s]["prediction"].append(p)

    
    for spk in spk_dict.keys():
        reference = spk_dict[spk]["reference"]
        predictions = spk_dict[spk]["prediction"]
        # print(reference, predictions)
        m_dict = compute_metrics_one(
            reference, predictions, verbose=verbose, is_binary_classification=is_binary_classification
        )
        spk_metrics[spk] = m_dict

    return spk_metrics


DEMOGR_FILE = "/home/yzhong/gits/TurnTakingPD/sync_private/demogr_perpp.txt"

def load_demographics(file_path):
    demogr = {}
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
        header = lines[0].split()
        idx_id = header.index("participantnummer")
        idx_age = header.index("leeftijd")
        idx_sex = header.index("geslacht")
        for line in lines[1:]:
            parts = line.split()
            if len(parts) < 3:
                continue
            pid = parts[idx_id]
            age = parts[idx_age]
            sex = parts[idx_sex]
            demogr[pid] = (age, sex)
    return demogr


def load_pcgita_metadata(csv_path):
    df = pd.read_csv(csv_path)

    # Display columns to verify 'speaker_id' exists
    print("Columns in CSV:", df.columns.tolist())

    # Ensure 'speaker_id' column exists
    if 'speaker_id' not in df.columns:
        raise KeyError("The CSV file must contain a 'speaker_id' column.")

    # Select relevant columns including 'speaker_id'
    selected_columns = ['speaker_id', 'status', 'UPDRS', 'UPDRS-speech', 'H/Y', 'SEX', 'AGE', 'time after diagnosis']
    df_selected = df[selected_columns]

    # Check for duplicate 'speaker_id's
    duplicates = df_selected.duplicated(subset='speaker_id', keep=False)
    if duplicates.any():
        duplicated_entries = df_selected[duplicates]
        
        # Verify that all duplicated entries have identical values in selected columns
        duplicates_unique = duplicated_entries.drop_duplicates()
        if duplicates_unique.shape[0] != duplicated_entries['speaker_id'].nunique():
            raise ValueError("Duplicate 'speaker_id' values found with differing data in selected columns.")
        else:
            # Drop duplicate rows, keeping the first occurrence
            df_selected = df_selected.drop_duplicates(subset='speaker_id', keep='first')

    # Create dictionary with 'speaker_id' as keys and selected columns as values
    data_dict = df_selected.set_index('speaker_id').to_dict(orient='index')
    # Create new dict with renamed keys
    renamed_dict = {}

    for spkid, values in data_dict.items():
        # Apply renaming rules
        if "AVPEPUDEAC" in spkid:
            new_spkid = int(spkid.replace("AVPEPUDEAC", "2"))
        elif "AVPEPUDEA" in spkid:
            new_spkid = int(spkid.replace("AVPEPUDEA", "1"))
        else:
            raise ValueError(f"Unrecognized speaker ID format: {spkid}")
            
        renamed_dict[new_spkid] = values

    data_dict = renamed_dict
    return data_dict


def compute_metrics_one(reference, predictions, verbose=False, is_binary_classification=False):
    
    accuracy = accuracy_score(reference, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(reference, predictions, average="macro")
    if is_binary_classification:
        try:
            roc_auc = roc_auc_score(reference, predictions)
            cm = confusion_matrix(reference, predictions)
            tp = cm[1, 1]
            tn = cm[0, 0]
            fp = cm[0, 1]
            fn = cm[1, 0]
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
        except:
            print("all one class, no ROC AUC")
            roc_auc = 0.0
            sensitivity = 0.0
            specificity = 0.0

        
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2,
    }

if __name__ == "__main__":
    config = add_arguments()
    config = Dict(config)
    set_all_seeds(config.training.seed)

    # create checkpoint path if it does not exist
    if not os.path.exists(config.training.checkpoint_path):
        os.makedirs(config.training.checkpoint_path, exist_ok=True)

    # create comet experiment if needed
    if config.training.use_comet:
        experiment = Experiment(
            api_key="zAyPDMX062pRKbbjm6JY6z6RP",
            workspace="terryyizhongru",
            project_name=config.training.comet_project_name,
        )
        experiment.set_name(config.training.comet_experiment_name)
        experiment.log_parameters(config)
    else:
        experiment = None


    # ------------------------------------------
    # Data preparation
    # ------------------------------------------
    if config.training.label_key == "status":
        class_mapping = {'hc':0, 'pd':1}
        is_binary_classification = True
        print(f"Class mapping: {class_mapping}")
    elif config.training.label_key == "UPDRS-speech":
        is_binary_classification = False
        class_mapping = {0: 0, 1: 1, 2: 2, 3: 3}
        print(f"Class mapping: {class_mapping}")


    test_results = {
        "accuracy": {},
        "precision": {},
        "recall": {},
        "f1": {},
        "roc_auc": {},
        "sensitivity": {},
        "specificity": {},
        "balanced_accuracy": {},
    }
    test_results_by_speaker = defaultdict(dict)
    
    for test_fold in range(1, config.data.num_folds+1):
        
        # info about the fold
        fold_path = config.data.fold_root_path + f"/TRAIN_TEST_{test_fold}/"
        test_path = fold_path + "test.csv"
        train_dl, val_dl, test_dl = get_dataloaders(test_path, test_path, class_mapping, config)
        
        # create model
        model = get_model(config)
        model, device = manage_devices(model, use_cuda=config.training.use_cuda, multi_gpu=config.training.multi_gpu)
        # print the number of parameters
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

        # load the best model
        # for epoch in range(config.training.num_epochs):
        #     print(f"Epoch: {epoch + 1}/{config.training.num_epochs}")
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(torch.load(config.training.checkpoint_path + f"/fold_{test_fold}.pt"))
        else:
            # model.load_state_dict(torch.load(config.training.checkpoint_path + f"/fold_{test_fold}.pt"))
            model.load_state_dict(torch.load(config.training.checkpoint_path + f"/fold_{test_fold}.pt"))

            
        if is_binary_classification: loss_fn = torch.nn.BCELoss()
        else: loss_fn = torch.nn.CrossEntropyLoss()

        print(loss_fn)
        
            
        # evaluate
        test_loss, test_reference, test_predictions = eval_one_epoch(
            model=model,
            eval_dataloader=test_dl,
            device=device,
            loss_fn=loss_fn,
            is_binary_classification=is_binary_classification,
        )

        if config.training.validation.by_speaker or config.training.validation.by_gender:
            test_spk_ids = []
            for batch in test_dl:
                test_spk_ids.extend(batch['spk_id'].tolist())
            
            if config.training.validation.dataset == "pcgita":
                metadata_dict = load_pcgita_metadata(test_path)
                print(metadata_dict)

            spk_m_dict = {}
            spk_m_dict = compute_metrics_by_speaker(
                spk_m_dict, test_spk_ids, test_reference, test_predictions, verbose=config.training.verbose, is_binary_classification=is_binary_classification
            )

            print(spk_m_dict)

            for spk_id in spk_m_dict.keys():
                test_results_by_speaker["accuracy"][spk_id] = spk_m_dict[spk_id]["accuracy"]
                test_results_by_speaker["precision"][spk_id] = spk_m_dict[spk_id]["precision"]
                test_results_by_speaker["recall"][spk_id] = spk_m_dict[spk_id]["recall"]
                test_results_by_speaker["f1"][spk_id] = spk_m_dict[spk_id]["f1"]
                test_results_by_speaker["roc_auc"][spk_id] = spk_m_dict[spk_id]["roc_auc"]
                test_results_by_speaker["sensitivity"][spk_id] = spk_m_dict[spk_id]["sensitivity"]
                test_results_by_speaker["specificity"][spk_id] = spk_m_dict[spk_id]["specificity"]
                test_results_by_speaker["balanced_accuracy"][spk_id] = spk_m_dict[spk_id]["balanced_accuracy"]
                if config.training.validation.dataset == "pcgita":
                    test_results_by_speaker["UPDRS"][spk_id] = metadata_dict[spk_id]["UPDRS"]
                    test_results_by_speaker["UPDRS-speech"][spk_id] = metadata_dict[spk_id]["UPDRS-speech"]
                    test_results_by_speaker["HY"][spk_id] = metadata_dict[spk_id]["H/Y"]
                    test_results_by_speaker["age"][spk_id] = metadata_dict[spk_id]["AGE"]
                    test_results_by_speaker["sex"][spk_id] = metadata_dict[spk_id]['SEX']
                    test_results_by_speaker["TAD"][spk_id] = metadata_dict[spk_id]["time after diagnosis"]
                    test_results_by_speaker["status"][spk_id] = metadata_dict[spk_id]["status"]
                
            # calculate metrics
        m_dict = compute_metrics(
            test_reference, test_predictions, verbose=config.training.verbose, is_binary_classification=is_binary_classification
        )
    
        accuracy = m_dict["accuracy"]
        precision = m_dict["precision"]
        recall = m_dict["recall"]
        f1 = m_dict["f1"]
        roc_auc = m_dict["roc_auc"]
        sensitivity = m_dict["sensitivity"]
        specificity = m_dict["specificity"]
        balanced_accuracy = m_dict["balanced_accuracy"]
        
        test_results["accuracy"][test_fold] = accuracy
        test_results["precision"][test_fold] = precision
        test_results["recall"][test_fold] = recall
        test_results["f1"][test_fold] = f1
        test_results["roc_auc"][test_fold] = roc_auc
        test_results["sensitivity"][test_fold] = sensitivity
        test_results["specificity"][test_fold] = specificity
        test_results["balanced_accuracy"][test_fold] = balanced_accuracy

        # print(f"Accuracy test fold {test_fold}: {accuracy:.3f}")
        # print(f"Precision test fold {test_fold}: {precision:.3f}")
        # print(f"Recall test fold {test_fold}: {recall:.3f}")
        # print(f"F1 test fold {test_fold}: {f1:.3f}")
        # print(f"ROC AUC test fold {test_fold}: {roc_auc:.3f}")
        # print(f"Sensitivity test fold {test_fold}: {sensitivity:.3f}")
        # print(f"Specificity test fold {test_fold}: {specificity:.3f}")
        # print(f"Balanced accuracy test fold {test_fold}: {balanced_accuracy:.3f}")
        # print(f"-" * 50)

        # save results

    results_df = pd.DataFrame(test_results)
    results_df.index.name = "fold_or_id"
    results_df.reset_index(inplace=True)
    
    results_df.to_csv(config.training.checkpoint_path + "/test_results_all_folds.csv", index=False)

    fw = open(config.training.checkpoint_path + "/test_results_average_fold.txt", "w")
    
    for metric in results_df.select_dtypes(include=['number']).columns:
        mean_metric = results_df[metric].mean()
        std_metric = results_df[metric].std()
        print(f"{metric}: {mean_metric*100:.2f} +/- {std_metric*100:.3f}")
        fw.write(f"{metric}: {mean_metric*100:.2f} +/- {std_metric*100:.3f}\n")
    
    fw.close()
    
    if config.training.validation.by_speaker or config.training.validation.by_gender:
        if config.training.validation.dataset == "TT":
            results_df = pd.DataFrame(test_results_by_speaker)
            results_df.index.name = "fold_or_id"
            results_df.reset_index(inplace=True)
        
            demo_dict = load_demographics(DEMOGR_FILE)
            results_df['sex'] = results_df['fold_or_id'].apply(lambda x: demo_dict.get(str(x), ('',''))[1])
            results_df['age'] = results_df['fold_or_id'].apply(lambda x: float(demo_dict.get(str(x), ('',''))[0]))
            results_df = results_df[['fold_or_id', 'accuracy', 'sex', 'age']]

            df_hc = results_df[results_df['fold_or_id'].astype(int) < 2200].sort_values(by="accuracy", ascending=True)
            df_pd = results_df[results_df['fold_or_id'].astype(int) >= 2200].sort_values(by="accuracy", ascending=True)
            results_df = results_df.sort_values(by="accuracy", ascending=True)
            results_df.to_csv(config.training.checkpoint_path + "/test_results_by_speaker.csv", index=False) 
            df_hc.to_csv(config.training.checkpoint_path + "/test_results_by_speaker_hc.csv", index=False)
            df_pd.to_csv(config.training.checkpoint_path + "/test_results_by_speaker_pd.csv", index=False)

            # if config.training.validation.by_gender:
            numeric_cols = results_df.drop(['fold_or_id'], axis=1).select_dtypes(include=['number']).columns
            grouped_df = results_df.groupby('sex')[numeric_cols].mean()
            grouped_df.to_csv(config.training.checkpoint_path + "/test_results_by_sex.csv", index=True)

            # cal age range
            bins_3 = [54, 65, 75, 85]
            labels_3 = ["54-65", "65-75", "75-85"]
            df_hc_3 = results_df[results_df['fold_or_id'].astype(int) < 2200].copy(deep=True)
            df_pd_3 = results_df[results_df['fold_or_id'].astype(int) >= 2200].copy(deep=True)
            results_df_3 = results_df.copy(deep=True)
            df_hc_3["age_group"] = pd.cut(df_hc["age"], bins=bins_3, right=False, labels=labels_3)
            df_pd_3["age_group"] = pd.cut(df_pd["age"], bins=bins_3, right=False, labels=labels_3)
            results_df_3["age_group"] = pd.cut(results_df["age"], bins=bins_3, right=False, labels=labels_3)

            hc_age_group_mean = df_hc_3.drop(['fold_or_id', 'sex'], axis=1).groupby("age_group").mean()
            pd_age_group_mean = df_pd_3.drop(['fold_or_id', 'sex'], axis=1).groupby("age_group").mean()
            all_age_group_mean = results_df_3.drop(['fold_or_id', 'sex'], axis=1).groupby("age_group").mean()

            
            bins_5yr = [54, 59, 64, 69, 74, 79, 85]
            labels_5yr = ["54-59", "59-64", "64-69", "69-74", "74-79", "79-85"]
            df_hc_5 = results_df[results_df['fold_or_id'].astype(int) < 2200].copy(deep=True)
            df_pd_5 = results_df[results_df['fold_or_id'].astype(int) >= 2200].copy(deep=True)
            results_df_5 = results_df.copy(deep=True)
            df_hc_5["age_group_5yr"] = pd.cut(df_hc["age"], bins=bins_5yr, right=False, labels=labels_5yr)
            df_pd_5["age_group_5yr"] = pd.cut(df_pd["age"], bins=bins_5yr, right=False, labels=labels_5yr)
            results_df_5["age_group_5yr"] = pd.cut(results_df["age"], bins=bins_5yr, right=False, labels=labels_5yr)

            hc_age_group_5yr_mean = df_hc_5.drop(['fold_or_id', 'sex'], axis=1).groupby("age_group_5yr").mean()
            pd_age_group_5yr_mean = df_pd_5.drop(['fold_or_id', 'sex'], axis=1).groupby("age_group_5yr").mean()
            all_age_group_5yr_mean = results_df_5.drop(['fold_or_id', 'sex'], axis=1).groupby("age_group_5yr").mean()

            with open(config.training.checkpoint_path + "/age_group_results.txt", "w") as f:
                f.write("HC by 3-range:\n")
                f.write(hc_age_group_mean.to_string())
                f.write("\n\nPD by 3-range:\n") 
                f.write(pd_age_group_mean.to_string())
                f.write("\n\nAll by 3-range:\n")
                f.write(all_age_group_mean.to_string())
                f.write("\n\nHC by 5-year steps:\n")
                f.write(hc_age_group_5yr_mean.to_string())
                f.write("\n\nPD by 5-year steps:\n") 
                f.write(pd_age_group_5yr_mean.to_string())
                f.write("\n\nAll by 5-year steps:\n")
                f.write(all_age_group_5yr_mean.to_string())

            fw = open(config.training.checkpoint_path + "/test_results_average_speaker.txt", "w")
            for metric in results_df.drop(['fold_or_id'], axis=1).select_dtypes(include=['number']).columns:
                mean_metric = results_df[metric].mean()
                std_metric = results_df[metric].std()
                print(f"{metric}: {mean_metric*100:.2f} +/- {std_metric*100:.3f}")
                fw.write(f"{metric}: {mean_metric*100:.2f} +/- {std_metric*100:.3f}\n")
            fw.close()
            
        # print average of each metric (column)
        elif config.training.validation.dataset == "pcgita":
            results_df = pd.DataFrame(test_results_by_speaker)
            results_df.index.name = "fold_or_id"
            results_df.reset_index(inplace=True)

            columns_to_drop = ['precision', 'recall', 'f1', 'roc_auc', 'sensitivity', 'specificity', 'balanced_accuracy']
            results_df.drop(columns=columns_to_drop, inplace=True)        

            df_hc = results_df[results_df['status']=='hc'].sort_values(by="accuracy", ascending=True)
            df_pd = results_df[results_df['status']=='pd'].sort_values(by="accuracy", ascending=True)
            results_df = results_df.sort_values(by="accuracy", ascending=True)
            results_df.to_csv(config.training.checkpoint_path + "/test_results_by_speaker.csv", index=False) 
            df_hc.to_csv(config.training.checkpoint_path + "/test_results_by_speaker_hc.csv", index=False)
            df_pd.to_csv(config.training.checkpoint_path + "/test_results_by_speaker_pd.csv", index=False)

            # if config.training.validation.by_gender:
            numeric_cols = results_df.drop(['fold_or_id'], axis=1).select_dtypes(include=['number']).columns
            grouped_df = results_df.groupby('sex')[numeric_cols].mean()
            grouped_df.to_csv(config.training.checkpoint_path + "/test_results_by_sex.csv", index=True)

                      # cal age range
            bins_3 = [30, 40, 50, 60, 70, 80, 90]  # 6 bins covering 30-90
            labels_3 = ["30-40", "40-50", "50-60", "60-70", "70-80", "80-90"]
            df_hc_3 = results_df[results_df['status']=='hc'].copy(deep=True)
            df_pd_3 = results_df[results_df['status']=='pd'].copy(deep=True)
            results_df_3 = results_df.copy(deep=True)
            df_hc_3["age_group"] = pd.cut(df_hc["age"], bins=bins_3, right=False, labels=labels_3)
            df_pd_3["age_group"] = pd.cut(df_pd["age"], bins=bins_3, right=False, labels=labels_3)
            results_df_3["age_group"] = pd.cut(results_df["age"], bins=bins_3, right=False, labels=labels_3)
            pdb.set_trace()
            numeric_cols = results_df.drop(['fold_or_id'], axis=1).select_dtypes(include=['number']).columns
            # hc_age_group_mean = df_hc_3.drop(['fold_or_id', 'sex'], axis=1).select_dtypes(include=['number']).columns
            hc_age_group_mean = df_hc_3.groupby("age_group")[numeric_cols].mean()
            pd_age_group_mean = df_pd_3.groupby("age_group")[numeric_cols].mean()
            all_age_group_mean = results_df_3.groupby("age_group")[numeric_cols].mean()
            
            with open(config.training.checkpoint_path + "/age_group_results.txt", "w") as f:
                f.write("HC by 10years-range:\n")
                f.write(hc_age_group_mean.to_string())
                f.write("\n\nPD by 10years-range:\n") 
                f.write(pd_age_group_mean.to_string())
                f.write("\n\nAll by 10years-range:\n")
                f.write(all_age_group_mean.to_string())
    

            fw = open(config.training.checkpoint_path + "/test_results_average_speaker.txt", "w")
            for metric in results_df.drop(['fold_or_id'], axis=1).select_dtypes(include=['number']).columns:
                mean_metric = results_df[metric].mean()
                std_metric = results_df[metric].std()
                print(f"{metric}: {mean_metric*100:.2f} +/- {std_metric*100:.3f}")
                fw.write(f"{metric}: {mean_metric*100:.2f} +/- {std_metric*100:.3f}\n")
            fw.close()