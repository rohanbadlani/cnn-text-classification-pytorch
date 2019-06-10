import pandas as pd
import numpy as np
import pickle

def get_labels_from_csv(csv_file, label):
	df = pd.read_csv(csv_file)
	return np.array(df[[label]].values)

def get_overlap(arr1: list, arr2: list) -> (int, int, int, int, int):
	arr1, arr2 = np.array(arr1), np.array(arr2)
	total_overlap = np.sum(arr1 == arr2)
	true_positive_overlap = np.sum(np.logical_and(arr1 == arr2, arr1 == 1))/np.sum(arr1 == 1)
	true_negative_overlap = np.sum(np.logical_and(arr1 == arr2, arr1 == 0))/np.sum(arr1 == 0)
	false_positive_overlap = np.average(np.logical_and(arr1 == 1, arr2 == 0))
	false_negative_overlap = np.average(np.logical_and(arr1 == 0, arr2 == 1))
	return total_overlap, true_positive_overlap, true_negative_overlap, false_positive_overlap, false_negative_overlap

def better_predictions(arr1, arr2, gt):
	res = np.argwhere(np.logical_and(arr1 == gt, arr2 != gt))
	return res

# def graph_visualize(mat):

if __name__ == "__main__":
	# print(get_overlap([1,1,1,0,0], [1,0,1,1,0]))
	results = {"true_positive_overlap": np.zeros((5, 5)),
				"true_negative_overlap": np.zeros((5, 5)),
				"false_positive_overlap": np.zeros((5, 5)),
				"false_negative_overlap": np.zeros((5, 5))}
	
	# file_names = ["only_sentiment/only_sentiment_out.csv",
	#               "only_sarcasm/only_sarcasm_out.csv",
	#               "only_humour/only_humour_out.csv",
	#               "only_hate_speech/only_hate_speech_out.csv"]
	
	file_names = ["/home/nishitasnani/shawn-na-repo/cnn-text-classification-pytorch/final_snapshot_sentiment_out.csv",
					"./sentiment_hate_speech_humour_sarcarm2/test_sentiment_hate_speech_humour_sarcarm2.csv",
					"/home/nishitasnani/shawn-na-repo/cnn-text-classification-pytorch/final_snapshot_2_sarc.csv",
					"/home/nishitasnani/shawn-na-repo/cnn-text-classification-pytorch/final_snapshot_sarc_humor.csv"]

	all_arrays = {}
	targets = None
	predictions_label, targets_label = "predictions", "targets"
	for n, fn in enumerate(file_names):
		predictions = get_labels_from_csv(fn, predictions_label)
		all_arrays[n] = predictions
		if targets is None:
			targets = get_labels_from_csv(fn, targets_label)
			all_arrays["ground_truth"] = targets

		find_overlap = False
		if find_overlap:
			for i, (task1, arr1) in enumerate(all_arrays.items()):
				for j, (task2, arr2) in enumerate(all_arrays.items()):
					print(i, task1, " ... ", j, task2)
					_, tpo, tno, fpo, fno = get_overlap(arr1, arr2)
					results["true_positive_overlap"][i][j] = tpo
					results["true_negative_overlap"][i][j] = tno
					results["false_positive_overlap"][i][j] = fpo
					results["false_negative_overlap"][i][j] = fno

			print(results)
			with open('results.pkl', 'wb') as outfile:
				pickle.dump(results, outfile, protocol=pickle.HIGHEST_PROTOCOL)

		else:
			results = better_predictions(all_arrays[1], all_arrays[0], all_arrays["ground_truth"])
			print(results)
			with open('resutls2.pkl', 'wb') as outfile:
					pickle.dump(results, outfile, protocol=pickle.HIGHEST_PROTOCOL)
	