from nlgeval import compute_metrics
dataset_tested = 'MSCOCO'
metrics_dict = compute_metrics(hypothesis='./'+ dataset_tested+'/predict.txt',
                               references=['./'+ dataset_tested + '/ground_truth1.txt', './'+dataset_tested+'/ground_truth2.txt','./'+dataset_tested+'/ground_truth3.txt','./'+dataset_tested+'/ground_truth4.txt','./'+dataset_tested+'/ground_truth5.txt'])