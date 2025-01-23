from comet_ml import Experiment

import os
import random
import yaml
import argparse
from tqdm import tqdm


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

import numpy as np
from train import *


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

    results = {
        "accuracy": {},
        "precision": {},
        "recall": {},
        "f1": {},
        "roc_auc": {},
        "sensitivity": {},
        "specificity": {},
    }

    test_results = {
        "accuracy": {},
        "precision": {},
        "recall": {},
        "f1": {},
        "roc_auc": {},
        "sensitivity": {},
        "specificity": {},
    }
    
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
        for epoch in range(config.training.num_epochs):
            print(f"Epoch: {epoch + 1}/{config.training.num_epochs}")
            if isinstance(model, torch.nn.DataParallel):
                model.module.load_state_dict(torch.load(config.training.checkpoint_path + f"/fold_{test_fold}.pt"))
            else:
                # model.load_state_dict(torch.load(config.training.checkpoint_path + f"/fold_{test_fold}.pt"))
                model.load_state_dict(torch.load(config.training.checkpoint_path + f"/fold_{test_fold}_epoch{epoch}.pt"))

                
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

            # calculate metrics
            m_dict = compute_metrics(
                test_reference, test_predictions, verbose=config.training.verbose, is_binary_classification=is_binary_classification
            )
            # print("test_reference", test_reference)
            # print("test_predictions", test_predictions)
            
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

            print(f"Accuracy test fold {test_fold}: {accuracy:.3f}")
            print(f"Precision test fold {test_fold}: {precision:.3f}")
            print(f"Recall test fold {test_fold}: {recall:.3f}")
            print(f"F1 test fold {test_fold}: {f1:.3f}")
            print(f"ROC AUC test fold {test_fold}: {roc_auc:.3f}")
            print(f"Sensitivity test fold {test_fold}: {sensitivity:.3f}")
            print(f"Specificity test fold {test_fold}: {specificity:.3f}")
            print(f"Balanced accuracy test fold {test_fold}: {balanced_accuracy:.3f}")
            print(f"-" * 50)

            # log metrics to comet
            if experiment is not None:
                experiment.log_metric("test_loss_fold_" + str(test_fold), test_loss)
                experiment.log_metric("test_accuracy_fold_" + str(test_fold), accuracy)
                experiment.log_metric("test_precision_fold_" + str(test_fold), precision)
                experiment.log_metric("test_recall_fold_" + str(test_fold), recall)
                experiment.log_metric("test_f1_fold_" + str(test_fold), f1)
                experiment.log_metric("test_roc_auc_fold_" + str(test_fold), roc_auc)
                experiment.log_metric("test_sensitivity_fold_" + str(test_fold), sensitivity)
                experiment.log_metric("test_specificity_fold_" + str(test_fold), specificity)

    # save results
    results_df = pd.DataFrame(test_results)
    results_df.to_csv(config.training.checkpoint_path + "/test_results2.csv", index=False)
    
    fw = open(config.training.checkpoint_path + "/test_results2.txt", "w")
    
    # print average of each metric (column)
    for metric in results_df.columns:
        mean_metric = results_df[metric].mean()
        std_metric = results_df[metric].std()
        print(f"{metric}: {mean_metric*100:.2f} +/- {std_metric*100:.3f}")
        fw.write(f"{metric}: {mean_metric*100:.2f} +/- {std_metric*100:.3f}\n")
        
        if experiment is not None:
            experiment.log_metric("mean_" + metric, mean_metric)
            experiment.log_metric("std_" + metric, std_metric)
            
    fw.close()