# visit https://e2b.dev/docs and paste your E2B API key here
export E2B_API_KEY="YOUR KEY HERE"
# if you want to train or evaluate ddm on code tasks,
# run me before the main script.
# you also need to pip install the following packages:
# pip install e2b-code-interpreter==1.0.5
# pip install morphcloud==0.1.67
# pip install fastapi
nohup python grpo/code_utils/e2b_router.py > e2b_router.log 2>&1 &

# to kill it, run
# pkill -f e2b_router.py