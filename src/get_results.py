import os
import warnings
from attrdict import AttrDict

import click
import numpy as np
import pandas as pd

from .run import get_runner
import src.utils as global_utils

warnings.filterwarnings("ignore")


@click.command()
@click.option('--goal', default=False, type=bool)
# Dataset configuration
@click.option('--data_name')
@click.option('--obs_len', default=8, help='number of observation steps')
@click.option('--pred_len', default=8, help='number of prediction steps')
@click.option('--skip', default=1)
@click.option('--get_last', default=False, type=bool)
@click.option('--min_ped', default=1)
@click.option('--drop', default=0)
@click.option('--use_goal', default=True, type=bool)
@click.option('--noise_thresh', type=float)
@click.option('--exclude')
@click.option('--rotate', default=False, type=bool)
@click.option('--shift', default=False, type=bool)
@click.option('--scale', default=False, type=bool)
@click.option('--perspective', default=True, type=bool)
@click.option('--batch_size', default=64)
@click.option('--dummy_goal', default=False, type=bool)
# Checkpoint
@click.option('--model_name', default="goal_gru")
@click.option('--checkpoint_name')
@click.option('--lastest', default=True, type=bool)
@click.option('--best', default=False, type=bool)
@click.option('--test_all', default=True, type=bool)
# Output
@click.option('--filename')
def get_results(model_name, checkpoint_name, lastest, best, test_all, filename, **kwargs):

    if not kwargs["goal"]:
        columns = [["test", "test"],
                   ["ade", "fde"]]
    else:
        columns = [["test", "test"],
                   ["top1", "top3"]]

    columns = pd.MultiIndex.from_tuples(list(zip(*columns)))

    indexes = []
    trajectory_results = []
    goal_results = []

    if not test_all:
        print(checkpoint_name)

        checkpoint_path = global_utils.get_checkpoint(checkpoint_name)

        model_runner = get_runner(model_name).create_runner(checkpoint_path, data_args=kwargs,
                                                            load_best=best)

        indexes.append(kwargs["data_name"])

        (_, _,
         (test_dataset, test_loader)) = global_utils.prepare_datasets(test=True, **kwargs)

        if not kwargs["goal"]:
            # Test trajectory_results
            metrics_test = model_runner["loader_get_accuracy"](test_loader)

            trajectory_results = [metrics_test["ade"], metrics_test["fde"]]
        else:
            metrics_test = model_runner["loader_get_goal_accuracy"](test_loader)

            trajectory_results = [metrics_test["top1"], metrics_test["top3"]]

    else:
        trajectory_path = global_utils.get_trajectory_path()
        for dataname in sorted(os.listdir(trajectory_path)):
            data_kwargs = kwargs.copy()
            data_kwargs['data_name'] = dataname
            print(dataname)
            try:
                (_, _,
                 (test_dataset, test_loader)) = global_utils.prepare_datasets(test=True, **data_kwargs)

                checkpoint_name = "obs{}_pred{}_{}_{}".format(data_kwargs["obs_len"],
                                                                   data_kwargs["pred_len"],
                                                                   dataname,
                                                                   model_name)
                checkpoint_path = global_utils.get_checkpoint(checkpoint_name)
                
                indexes.append(dataname)
                model_runner = get_runner(model_name).create_runner(checkpoint_path, data_args=data_kwargs,
                                                                    load_best=best)

                if not kwargs["goal"]:
                    # Test trajectory_results
                    metrics_test = model_runner["loader_get_accuracy"](test_loader)

                    data_trajectory_results = [metrics_test["ade"], metrics_test["fde"]]
                else:
                    metrics_test = model_runner["loader_get_goal_accuracy"](test_loader)

                    data_trajectory_results = [metrics_test["top1"], metrics_test["top3"]]

                

                print(data_trajectory_results)
                trajectory_results.append(np.array(data_trajectory_results))
            except Exception as e:
                print("Error in dataset", dataname, e)

    if not test_all:
        trajectory_results = np.expand_dims(trajectory_results, axis=0)
    else:
        avg = np.average(trajectory_results, axis=0)
        trajectory_results.append(avg)
        indexes.append("avg")

    trajectory_results = [np.around(x, 2) for x in trajectory_results]

    df = pd.DataFrame(trajectory_results, columns=columns, index=indexes)
    print(df)

    if filename is not None:
        df.to_csv(os.path.join(TABLE_PATH, "{}.csv".format(filename)))



if __name__ == "__main__":
    get_results()
