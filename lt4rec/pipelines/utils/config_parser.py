# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from components.utils.types import FeatureConfig


def parse_feature_configs(feature_configs):
    return  [
        FeatureConfig(name=name, dtype=config["dtype"], size=config["size"])
        for name, config in feature_configs.items()
    ]


def parse_metric_configs(metric_configs):
    metrics = {}
    for metric_config in metric_configs:
        if metric_config["type"] == "auc":
            from components.metrics.auc_metric import AUC
            metrics[metric_config["name"]] = AUC(
                label_name=metric_config["label"],
            )
        elif metric_config["type"] == "weighted_auc":
            from components.metrics.auc_metric import WeightedAUC
            metrics[metric_config["name"]] = WeightedAUC(
                label_name=metric_config["label"],
                weight_name=metric_config["weight"],
            )
        
        elif metric_config["type"] == "mse":
            from components.metrics.mse_metric import MSE
            metrics[metric_config["name"]] =MSE(
                label_name=metric_config["label"],
            )
        elif metric_config["type"] == "weighted_mse":
            from components.metrics.mse_metric import WeightedMSE
            metrics[metric_config["name"]] =WeightedMSE(
                label_name=metric_config["label"],
            )
        elif metric_config["type"] == "weighted_rmse":
            from components.metrics.mse_metric import WeightedRMSE
            metrics[metric_config["name"]] =WeightedRMSE(
                label_name=metric_config["label"],
            )
        else:
            raise NotImplementedError
    return metrics


def parse_loss_configs(loss_config):
    if loss_config["type"] == "cross_entropy":
        from components.losses.cross_entropy_loss import CrossEntropyLoss
        return CrossEntropyLoss(
            label_name=loss_config["label"],
        )
    elif loss_config["type"] == "weighted_cross_entropy":
        from components.losses.cross_entropy_loss import WeightedCrossEntropyLoss
        return WeightedCrossEntropyLoss(
            label_name=loss_config["label"],
            weight_name=loss_config["weight"],
        )

    elif loss_config["type"] == "mse":
        from components.losses.mse_loss import MSELoss
        return MSELoss(
            label_name=loss_config["label"],
        )
    elif loss_config["type"] == "weighted_mse":
        from components.losses.mse_loss import WeightedMSELoss
        return WeightedMSELoss(
            label_name=loss_config["label"],
        )

    elif loss_config["type"] == "focal_loss":
        from components.losses.cross_entropy_loss import FocalLoss
        return FocalLoss(
            label_name=loss_config["label"],
            alpha=loss_config["alpha"],
            gamma=loss_config["gamma"]
        )
    else:
        raise NotImplementedError
