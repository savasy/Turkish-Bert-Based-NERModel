# How the model was trained
This model is used for Named-Entity Recognition based on BERTurk for Turkish Language 
https://huggingface.co/dbmdz/bert-base-turkish-cased

Ner Model Link
https://huggingface.co/savasy/bert-base-turkish-ner-cased


# Quick Run

```
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

model = AutoModelForTokenClassification.from_pretrained("savasy/bert-base-turkish-ner-cased")
tokenizer = AutoTokenizer.from_pretrained("savasy/bert-base-turkish-ner-cased")
ner=pipeline('ner', model=model, tokenizer=tokenizer)
ner("Mustafa Kemal Atatürk 19 Mayıs 1919'da Samsun'a ayak bastı.")
```


## DataSet
Training dataset is WikiAnn

* The WikiANN dataset (Pan et al. 2017) is a dataset with NER annotations for PER, ORG and LOC. It has been constructed using the linked entities in Wikipedia pages for 282 different languages including Danish. The dataset can be loaded with the DaNLP package:

https://www.aclweb.org/anthology/P17-1178.pdf

Thank to @stefan-it, I downloaded the data from the link as follows

```
mkdir tr-data

cd tr-data

for file in train.txt dev.txt test.txt labels.txt
do
  wget https://schweter.eu/storage/turkish-bert-wikiann/$file
done
```

## Fine-tuning the bert-model 
The base bert model is dbmdz/bert-base-turkish-cased . With following system environment

```
export MAX_LENGTH=128
export BERT_MODEL=dbmdz/bert-base-turkish-cased
export OUTPUT_DIR=tr-new-model
export BATCH_SIZE=32
export NUM_EPOCHS=3
export SAVE_STEPS=625
export SEED=1

```

I run the following ner-training code(you can find it under transformer github repo)


```
Then run training:

python3 run_ner.py --data_dir ./tr-data3 \
--model_type bert \
--labels ./tr-data/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR-$SEED \
--max_seq_length $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_train \
--do_eval \
--do_predict \
--fp16
```

If you dont have GPU-enabled computer, please skip last --fp16 parameter.
Finally, you can find your trained model and model performance unde tr-new-model folder


## Some Results

###Performance for WikiANN dataset
```
cat tr-new-model-1/eval_results.txt
cat tr-new-model-1/test_results.txt

*Eval Results:*

precision = 0.916400580551524
recall = 0.9342309684101502
f1 = 0.9252298787412536
loss = 0.11335893666411284

*Test Results:*
precision = 0.9192058759362955
recall = 0.9303010230367262
f1 = 0.9247201697271198
loss = 0.11182546521618497

```

### Performance with another dataset at the link
https://github.com/stefan-it/turkish-bert/files/4558187/nerdata.txt

```
savas@savas-lenova:~/Desktop/trans/tr-new-model-1$ cat eval_results.txt
precision = 0.9461980692049029
recall = 0.959309358847465
f1 = 0.9527086063783312
loss = 0.037054269206847804

savas@savas-lenova:~/Desktop/trans/tr-new-model-1$ cat test_results.txt
precision = 0.9458370635631155
recall = 0.9588201928530913
f1 = 0.952284378344882
loss = 0.035431676572445225
```

# Usage

You should install transformer library first

```
pip install transformers
```

And, open python environment and run the following code

```
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

model = AutoModelForTokenClassification.from_pretrained("savasy/bert-base-turkish-ner-cased")
tokenizer = AutoTokenizer.from_pretrained("savasy/bert-base-turkish-ner-cased")
ner=pipeline('ner', model=model, tokenizer=tokenizer)
ner("Mustafa Kemal Atatürk 19 Mayıs 1919'da Samsun'a ayak bastı.")

[{'word': 'Mustafa', 'score': 0.9938516616821289, 'entity': 'B-PER'}, {'word': 'Kemal', 'score': 0.9881671071052551, 'entity': 'I-PER'}, {'word': 'Atatürk', 'score': 0.9957979321479797, 'entity': 'I-PER'}, {'word': 'Samsun', 'score': 0.9059973359107971, 'entity': 'B-LOC'}]

```

















