# TableIE: Capturing the Interactions among Sub-tasks in Information Extraction via Double Tables

This is the implementation for ICASSP2023 paper: TableIE: Capturing the Interactions among Sub-tasks in Information Extraction via Double Tables


## Dataset

We follow the data pre-process in [OneIE](http://blender.cs.illinois.edu/software/oneie/) for ACE05 train/dev/test set.


## Files

```
TableIE/
├── checkpoint/
├── dataset/
    ├── schema/
        ├── entity.json
        ├── relation.json
        ├── event.json
        ├── role.json
    ├── train.json
    ├── dev.json
    ├── test.json
├── roberta-large/ (optional)
├── data.py
├── model.py
├── train.py
├── test.py
├── utils.py
├─ README.md
```



## Train and Test

```bash
>> python train.py  # for train
>> python test.py  # for test
```


## Acknowledgement
If you use this code as part of your research, please cite the following paper:
```

```