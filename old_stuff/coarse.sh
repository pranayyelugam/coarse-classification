cd /iesl/canvas/nnayak/coarse-classification
source class_ve/bin/activate
module load python3/3.9.1-2102
python train.py csvs/review_sentence_train.csv coarse
