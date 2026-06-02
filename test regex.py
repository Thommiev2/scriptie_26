
class Pipe:
	def __init__(self, models, metrics, datasets):
		self.models = models
		self.metrics = metrics
		self.datasets = datasets

	def run(self):
		for dataset in self.datasets:
			ground_truth = dataset[ground_truth]
