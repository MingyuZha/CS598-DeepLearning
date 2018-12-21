from nlgeval import compute_metrics
dataset_tested = 'MSCOCO'
metrics_dict = compute_metrics(hypothesis='./Metrics/examples/hyp.txt',
                              references=['./Metrics/examples/ref1.txt', './Metrics/examples/ref2.txt'])