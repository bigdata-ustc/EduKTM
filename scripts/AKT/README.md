## AKT load_data.py

This script is used for loading data from a txt file.

### Data Format

The data format in txt file could be:

1. without problem_id
    ```
    seqlen
    question_id_1,question_id_2,question_id_3
    answer_1, answer_2, answer_3
    ```
   for example:
   ```
   7
   83,83,83,83,20,84,19
   0,1,1,1,1,1,1
   ```
   
2. with problem_id
   ```
   seqlen
   problem_id_1,problem_id_2,problem_id_3
   question_id_1,question_id_2,question_id_3
   answer_1, answer_2, answer_3
   ```
   for example:
   ```
   7
   13060,12972,13087,13057,3758,13366,3605
   83,83,83,83,20,84,19
   0,1,1,1,1,1,1
   ```
   
It will be translated to following format:
```python
# without problem_id
tuple(question_array, question_answer_array, student_id_array)
question_array: np_array(np_array, np_array, ...)
question_answer_array: np_array(np_array, np_array, ...)
student_id_array: np_array(1, 2, ...)

# with problem_id
tuple(question_array, question_answer_array, problem_array)
question_array: np_array(np_array, np_array, ...)
question_answer_array: np_array(np_array, np_array, ...)
problem_array: np_array(np_array, np_array, ...)
```

### Usage

```python
from scripts.AKT import DATA, PID_DATA

model_type = 'pid'
n_question = 123
n_pid = 17751
seqlen = 200

if model_type == 'pid':
    dat = PID_DATA(n_question=n_question, seqlen=seqlen, separate_char=',')
else:
    dat = DATA(n_question=n_question, seqlen=seqlen, separate_char=',')

train_data = dat.load_data('../../data/2009_skill_builder_data_corrected/train_pid.txt')

q, qa, p = train_data
```