# coding=utf-8
# Copyright 2020 HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition"""

import datasets


logger = datasets.logging.get_logger(__name__)

# _TRAINING_FILE = "large_train.txt"
# _DEV_FILE = "large_valid.txt"


class MINDConfig(datasets.BuilderConfig):
    """BuilderConfig for Conll2003"""

    def __init__(self, **kwargs):
        """BuilderConfig forMIND.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MINDConfig, self).__init__(**kwargs)


class MIND(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        MINDConfig(name="MIND", version=datasets.Version("1.0.0"), description="MIND dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
#             description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=['O', 'B-P', 'I-P', 'E-P', 'B-J', 'I-J', 'E-J', 'B-D', 'I-D', 'E-D', 'B-C', 'I-C', 'E-C', 'B-G', 'I-G', 'E-G', 'B-H', 'I-H', 'E-H', 'B-V', 'I-V', 'E-V', 'B-W', 'I-W', 'E-W', 'B-F', 'I-F', 'E-F', 'B-U', 'I-U', 'E-U', 'B-N', 'I-N', 'E-N', 'B-S', 'I-S', 'E-S', 'B-M', 'I-M', 'E-M', 'B-X', 'I-X', 'E-X', 'B-L', 'I-L', 'E-L', 'B-Y', 'I-Y', 'E-Y', 'B-K', 'I-K', 'E-K', 'B-Z', 'I-Z', 'E-Z', 'B-R', 'I-R', 'E-R', 'B-Q', 'I-Q', 'E-Q', 'B-A', 'I-A', 'E-A', 'B-T', 'I-T', 'E-T']
                        )
                    ),
                }
            ),
            supervised_keys=None
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        downloaded_files = {'train': '/scratch/jx880/capstone/transformers/examples/token-classification/data/large_train.txt',
                             'dev': '/scratch/jx880/capstone/transformers/examples/token-classification/data/large_dev.txt',
                            'test': '/scratch/jx880/capstone/transformers/examples/token-classification/data/large_test.txt'
        }

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            ner_tags = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "ner_tags": ner_tags,
                        }
                        guid += 1
                        tokens = []
                        ner_tags = []
                else:
                    # conll2003 tokens are space separated
                    splits = line.split(" ")
                    tokens.append(splits[0])
                    ner_tags.append(splits[1].rstrip())
            # last example
            yield guid, {
                "id": str(guid),
                "tokens": tokens,
                "ner_tags": ner_tags,
            }
