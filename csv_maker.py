import csv
import sys
import glob
import json
import collections

(REVIEW, REBUTTAL, REVIEW_LABELS,
    REBUTTAL_LABELS) = "review rebuttal reviewlabels rebuttallabels".split()

class LabelSet(object):
  Coarse = "coarse"
  Fine = "fine"
  Polarity = "pol"
  Aspect = "asp"
  Relation = "responsetype"
  CoarseRelation = "coarseresponse"

  ALL = [Coarse, Fine, Polarity, Aspect, Relation, CoarseRelation]  # Add no label labels??
  REVIEW_LABELS = [Coarse, Fine, Polarity, Aspect]
  REBUTTAL_LABELS = [Relation, CoarseRelation]

  label_map = {
      REVIEW: REVIEW_LABELS,
      REBUTTAL: REBUTTAL_LABELS
      }


class Subset(object):
  train = "train"
  dev = "dev"
  test = "test"
  ALL = [train, dev, test]



LABEL_BUILDER = {
    label:set(["None"]) for label in LabelSet.ALL
    }

def create_sentence_examples(example_obj, offset, review_or_rebuttal):
  print(example_obj["metadata"])
  for k, v in example_obj.items():
    print(k, len(v))
  assert review_or_rebuttal in "review rebuttal".split()
  if review_or_rebuttal == "review":
    which_labels = REVIEW_LABELS
  else:
    which_labels = REBUTTAL_LABELS
  examples = []
  for i, sentence in enumerate(example_obj[review_or_rebuttal]):
    builder = {
        key: example_obj[which_labels][i]["labels"][key]
        for key in LabelSet.label_map[review_or_rebuttal]
    }
    for k, v in builder.items():
      LABEL_BUILDER[k].add(v)
      if not v:
        builder[k] = "None"
    builder["id"] = offset + i
    builder["sentence"] = sentence["sentence"]
    examples.append(builder)
  return examples


def create_rebuttal_sentence_examples(example_obj):
  return create_sentence_examples(REBUTTAL, example_obj)


ReviewExample = collections.namedtuple("ReviewExample",
                                       ["sentence"] + LabelSet.REVIEW_LABELS)


class ExampleType(object):
  ReviewSentence = "review-sentence"
  RebuttalSentence = "rebuttal-sentence"



def main():
  data_dir = sys.argv[1]
  # Form csv files
  offset = 0
  for subset in Subset.ALL:
    glob_path = "/".join([data_dir, subset, "*.json"])
    sentence_examples = {
        REVIEW: [],
        REBUTTAL: []
        }
    for filename in glob.glob(glob_path):
      with open(filename, 'r') as f:
        example_obj = json.load(f)
        sentence_examples[REVIEW] += create_sentence_examples(
            example_obj, offset, REVIEW)
        offset += len(example_obj[REVIEW])
        sentence_examples[REBUTTAL] += create_sentence_examples(
            example_obj, offset, REBUTTAL)
        offset += len(example_obj[REBUTTAL])

    for rev_or_reb, examples in sentence_examples.items():
      FIELDS = "id sentence".split() + LabelSet.label_map[rev_or_reb]
      with open("".join(["csvs/", rev_or_reb, "_sentence_", subset, ".csv"]),
              'w') as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        for example in examples:
          writer.writerow(example)

  with open("labels.json", 'w') as f:
    json.dump({k:list(sorted(v)) for k, v in LABEL_BUILDER.items()} ,f)



if __name__ == "__main__":
  main()
