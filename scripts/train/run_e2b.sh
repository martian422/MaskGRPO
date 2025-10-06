export E2B_API_KEY=""
# if you want to train ddm on code tasks,
# run me before the train.sh.
# you also need to pip install the following packages:
# e2b-code-interpreter      1.0.5
# morphcloud                0.1.67
nohup python grpo/code_utils/e2b_router.py > e2b_router.log 2>&1 &