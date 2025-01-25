python main.py train --dataset comta --crossval --model_type dkt-sem --model_name dkt-sem_comta

python main.py train --dataset comta --crossval --model_type dkt-sem-rdrop-attn --model_name dkt-sem-rdrop-attn_comta


python main.py train --dataset comta --hyperparam_sweep --model_type dkt-sem-rdrop


python main.py train --dataset comta --hyperparam_sweep --model_type dkt-sem-rdrop-attn
