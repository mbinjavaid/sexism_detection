This release contains the `Call me sexist but (CMSB)` dataset. This is the companion data for the paper:

Samory, Mattia, Indira Sen(+), Julian Kohne(+), Fabian Floeck, and Claudia Wagner. ""Call me sexist but...": Revisiting Sexism Detection Using Psychological Scales and Adversarial Samples." ICWSM (2021). (+Indira Sen and Julian Kohne contributed equally to this work)

##In this release:

* `DATASTATEMENT.md` explains the rationale for data collection as well as details the nature of the natural language contents of the dataset. This document may help addresses critical scientific and ethical concerns regarding the use of this dataset, such as its generalizability and its intended use.
* `scales.csv` includes names, references, and descriptions of the scales used to source the sexism scale items. See the paper for full details.
* `sexist_data.csv` contains the data and binary sexism labels in csv format. Each row is an example. Each example has the following fields as columns in the file:
    - `id`: (int >= 0) the serial id of the example
    - `dataset`: (str, one of `benevolent, hostile, callme, scales, other`) the dataset of origin of the example (see the data statement and the paper for details on new and pre-existing datasets)
    - `text`: (str) the text of the example (some preprocessing has been applied to remove PII and avoid data collection confounders, see the data statement and the paper)
    - `toxicity`: (float) toxicity score of the text obtained from the Perspective API
    - `sexist`: (boolean) binary label that tells if the text is sexist or not (see the paper for details on how to derive the binary labels from the fine-grained annotations in `sexist_annotations.csv`) 
    - `of_id`: (int) if the example is a modification this field is the `id` of the original example, `-1` otherwise (see the paper for details on how crowdworkers generated adversarial, non-sexist examples from original, sexist ones) 
* `sexist_annotations.csv` contains the annotations for the examples in `sexist_data.csv` in csv format. Each example has at least 5 annotations from 5 different annotators. One row is one annotation, with the following fields as columns:
    - `phrasing`: (int, 1--3) category for sexist phrasing (1=sexist, 2=uncivil but not sexist, 3=not sexist, see codebook in the paper) 
    - `content`: (int, 1--6) category for sexist content (1=behavioral expectations, 2=stereotypes and comparisons, 3=endorsement of inequality, 4=denying inequality and rejecting feminism, 5=cannot tell, 6=not sexist, see codebook in the paper)
    - `worker`: (int >= 0) pseudonymized id of the crowdworker providing the annotation
    - `id`: (int >= 0) id of the example in `sexist_data.csv` that the annotation refers to
