import argparse

from metrics import METRICS
from datasets import Dataset
import pandas as pd

available_datasets = ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions',
'squirrel', 'squirrel-directed', 'squirrel-filtered', 'squirrel-filtered-directed',
'chameleon', 'chameleon-directed', 'chameleon-filtered', 'chameleon-filtered-directed',
'actor', 'texas', 'texas-4-classes', 'cornell', 'wisconsin']

available_metrics = METRICS.keys()
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--out', type=str, default='metrics_output.csv', help='output file name')

    parser.add_argument('--datasets', type=str, nargs='*',  
                            default=available_datasets, 
                            choices=available_datasets)
    parser.add_argument('--metrics', type=str, nargs='*',  
                            default=available_metrics, 
                            choices=available_metrics)


    parser.add_argument('--use_sgc_features', default=False, action='store_true')
    parser.add_argument('--use_identity_features', default=False, action='store_true')
    parser.add_argument('--use_adjacency_features', default=False, action='store_true')
    parser.add_argument('--do_not_use_original_features', default=False, action='store_true')

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--verbose', default=False, action='store_true')

    args = parser.parse_args()


    return args

def cal_metrics(dataset_name):
    args = get_args()
    dataset = Dataset(name=dataset_name,
                    add_self_loops=False,
                    device=args.device,
                    use_sgc_features=args.use_sgc_features,
                    use_identity_features=args.use_identity_features,
                    use_adjacency_features=args.use_adjacency_features,
                    do_not_use_original_features=args.do_not_use_original_features,
                    bin2float=False)
    labels = dataset.labels
    features = dataset.node_features
    graph = dataset.graph
    results = []

    for name, metric in METRICS.items():
        if args.metrics == 'all' or name in args.metrics:
            value = metric(labels, features, graph)  # Calculate the metric
            results.append({"Metric": name, dataset_name: value})  # Append to results
    # print(results)
    ret = pd.DataFrame(results)
    return ret

def main():
    args = get_args()
    print('datasets:',args.datasets)
    print('metrics:',args.metrics)
    cols = []
    for dataset_name in args.datasets:
        cols.append(cal_metrics(dataset_name))
        cols[-1].set_index('Metric', inplace=True)
    df = pd.concat(cols, axis=1)
    df.to_csv(args.out)
    # for name,metric in METRICS.items():
    #     if args.metric == 'all' or args.metric == name:
    #         print("%s: %s"%(name,metric(labels,features,graph)))


if __name__ == '__main__':
    main()
