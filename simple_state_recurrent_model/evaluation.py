import numpy as np
import pathos.multiprocessing as mp
from simple_state_recurrent_model.model import SimpleRecurrentModel


class Evaluator(object):
    def __init__(self, model_args, input_data, target_data):
        self.model_args = model_args
        self.input_data = input_data
        self.target_data = target_data

        self.data_preproc_model = SimpleRecurrentModel(**self.model_args)
        self.loo_cross_validation_results = None
        self.accuracy_curve_results = None


    def loo_cross_validation(self, batch_size=32, train_rate=0.1, steps=1, epochs_per_step=100000, threads=4):
        def loo_x_validate(loo_cand):
            model = SimpleRecurrentModel(**self.model_args)
            filtered_inputs = [self.input_data[i] for i in range(len(self.input_data)) if i != loo_cand]
            filtered_targets = [self.target_data[i] for i in range(len(self.target_data)) if i != loo_cand]
            processed_inputs, processed_targets = model.assemble_data(filtered_inputs, filtered_targets)

            training_results = {}
            for s in range(1, steps + 1):
                model._raw_train(processed_inputs, processed_targets, batch_size, train_rate, epochs_per_step)
                training_results[s * epochs_per_step] = model.compute_inference(self.input_data[loo_cand])
            return training_results

        loo_candidates = range(len(self.input_data))
        p = mp.Pool(threads)
        results = p.map(loo_x_validate, loo_candidates)
        p.close()

        processed_results = {}
        for k in results[0].keys():
            processed_results[k] = [r[k] for r in results]
        self.loo_cross_validation_results = processed_results
        return self.loo_cross_validation_results


    def compute_accuracy_curve(self, resolution):
        if self.loo_cross_validation_results is None:
            raise Exception("loo_cross_validation results not yet computed")
        keys = self.loo_cross_validation_results.keys()
        accuracy_curve = {k: [] for k in keys}

        for k in keys:
            for i in range(resolution + 1):
                thresh = i / float(resolution)
                d = []
                for r in range(len(self.loo_cross_validation_results)):
                    model_pos = np.where(np.array(self.loo_cross_validation_results[k][r]) > thresh)[0]
                    target_pos = [i for i in range(len(self.input_data[r])) if any([i >= j[0] and i < j[1] for j in self.target_data[r]])]
                    d.append(set(model_pos) == set(target_pos))
                accuracy_curve[k].append(sum(d) / len(d))

        self.accuracy_curve_results = accuracy_curve
        return self.accuracy_curve_results
