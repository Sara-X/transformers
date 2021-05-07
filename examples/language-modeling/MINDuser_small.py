"""TODO(wikitext): Add a description here."""


import os

import datasets


class MINDuserConfig(datasets.BuilderConfig):
    """BuilderConfig for GLUE."""

    def __init__(self, **kwargs):
        """BuilderConfig for Wikitext
        Args:
          data_url: `string`, url to the dataset (word or raw level)
          **kwargs: keyword arguments forwarded to super.
        """
        super(MINDuserConfig, self).__init__(**kwargs)



class MINDuser(datasets.GeneratorBasedBuilder):
    """TODO(wikitext_103): Short description of my dataset."""

    # TODO(wikitext_103): Set up version.
#     VERSION = datasets.Version("0.1.0")
    BUILDER_CONFIGS = [
       MINDuserConfig(name="MINDuser", version = datasets.Version("1.0.0"), description = "MIND user dataset")
    ]

    def _info(self):
        # TODO(wikitext): Specifies the datasets.DatasetInfo object
        return datasets.DatasetInfo(
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    "input_ids": datasets.Sequence(datasets.Value("int64")),
                    # These are the features of your dataset like images, labels ...
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_files = {'train': '/scratch/jx880/capstone/transformers/examples/language-modeling/data/small/small_train_user_entities.txt',
                             'dev': '/scratch/jx880/capstone/transformers/examples/language-modeling/data/small/small_dev_user_entities.txt',
#                             'test': '/scratch/jx880/capstone/transformers/examples/language-modeling/data/test_user_entities.txt'
        }

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"data_file": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"data_file": downloaded_files["dev"]}),
#             datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
       
                        ]

    def _generate_examples(self, data_file):

        """Yields examples."""
        # TODO(wikitext): Yields (key, example) tuples from the dataset
        with open(data_file, encoding="utf-8") as f:
            for idx, row in enumerate(f):
                if len(row) == 1:
                    continue
                str_list = row.strip().split(" ")
#                 logger.info(str_list)
                int_list = [int(num) for num in str_list]
                
                yield idx, {"input_ids": int_list}
