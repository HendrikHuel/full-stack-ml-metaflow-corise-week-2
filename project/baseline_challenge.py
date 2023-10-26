# TODO: In this cell, write your BaselineChallenge flow in the baseline_challenge.py file.

from metaflow import (
    FlowSpec,
    step,
    Flow,
    current,
    Parameter,
    IncludeFile,
    card,
    current,
)
from metaflow.cards import Table, Markdown, Artifact, Image
import numpy as np
from dataclasses import dataclass

from utils.preprocessing import labeling_function

# TODO: Define your labeling function here.
labeling_function = labeling_function


@dataclass
class ModelResult:
    "A custom struct for storing model evaluation results."
    name: None
    params: None
    pathspec: None
    acc: None
    rocauc: None


class BaselineChallenge(FlowSpec):
    split_size = Parameter("split-sz", default=0.2)
    data = IncludeFile("data", default="data/Womens Clothing E-Commerce Reviews.csv")
    kfold = Parameter("k", default=5)
    scoring = Parameter("scoring", default="accuracy")

    @step
    def start(self):
        """Preprocess data."""
        import pandas as pd
        import io
        from sklearn.model_selection import train_test_split

        from utils.preprocessing import clean_data

        # load dataset packaged with the flow.
        # this technique is convenient when working with small datasets that need to move to remove tasks.
        # TODO: load the data.
        df = pd.read_csv(io.StringIO(self.data), index_col=0)
        # Look up a few lines to the IncludeFile('data', default='Womens Clothing E-Commerce Reviews.csv').
        # You can find documentation on IncludeFile here: https://docs.metaflow.org/scaling/data#data-in-local-files

        # filter down to reviews and labels
        self.df = clean_data(df)

        # split the data 80/20, or by using the flow's split-sz CLI argument
        self.traindf, self.valdf = train_test_split(self.df, test_size=self.split_size)
        print(f"num of rows in train set: {self.traindf.shape[0]}")
        print(f"num of rows in validation set: {self.valdf.shape[0]}")

        self.next(self.baseline, self.nn_model)

    @step
    def baseline(self):
        "Compute the baseline"

        from sklearn.dummy import DummyClassifier
        from utils.metrics import calc_scores

        self._name = "baseline"
        params = "Always predict 1"
        pathspec = f"{current.flow_name}/{current.run_id}/{current.step_name}/{current.task_id}"       

        majority_class = self.traindf["label"].mean().round(0)

        clfd = DummyClassifier(strategy="most_frequent")
        clfd.fit(X=self.traindf.drop(columns=["label"]), y=self.traindf["label"])

        assert np.mean(clfd.predict(self.traindf)) == majority_class

        # TODO: predict the majority class
        self.valdf["dummy_model"] = clfd.predict(self.valdf)
        # TODO: return the accuracy_score of these predictions
        # TODO: return the roc_auc_score of these predictions

        acc, rocauc = calc_scores(self.valdf)

        self.fitted_models = [ModelResult("Baseline", params, pathspec, acc, rocauc)]
        self.next(self.aggregate)

    @step
    def nn_model(self):
        """Train the NN model for a defined parameter set."""
        # TODO: import your model if it is defined in another file.
        from model import NbowModel
        self._name = "nn_model"
        # NOTE: If you followed the link above to find a custom model implementation,
        # you will have noticed your model's vocab_sz hyperparameter.
        # Too big of vocab_sz causes an error. Can you explain why?
        self.hyperparam_set = [{"vocab_sz_rev": 100}, {"vocab_sz_rev": 300}, {"vocab_sz_rev": 500}]
        pathspec = f"{current.flow_name}/{current.run_id}/{current.step_name}/{current.task_id}"

        self.fitted_models = []
        for params in self.hyperparam_set:
            model = NbowModel(**params)  # TODO: instantiate your custom model here!
            model.fit(X=self.traindf.drop(columns=["label"]), y=self.traindf["label"])
            # TODO: evaluate your custom model in an equivalent way to accuracy_score.
            # TODO: evaluate your custom model in an equivalent way to roc_auc_score.
            acc, rocauc = model.evaluate(self.valdf, self.valdf["label"])

            self.fitted_models.append(
                ModelResult(
                    f"{self._name} - vocab_sz: {params['vocab_sz_rev']}",
                    params,
                    pathspec,
                    acc,
                    rocauc,
                )
            )

        self.next(self.aggregate)

    @step
    def aggregate(self, inputs):
        """Find the best model."""
        def score(inp):
            return inp.name, inp.acc, inp.params

        self.results = sorted(map(score, sum([i.fitted_models for i in inputs], [])), key=lambda x: -x[1]) 
        self.model = self.results[0][0]


        self.next(self.end)

    @step
    def end(self):
        """
        End of flow!
        """
        print('Scores:')
        print('\n'.join('%s %f %s' % res for res in self.results))
        print('Best model:')
        print(self.model)


if __name__ == "__main__":
    BaselineChallenge()
