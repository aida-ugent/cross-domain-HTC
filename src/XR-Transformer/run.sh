data=$1
seeds=(0 42 666 23333 12345678)

data_dir="./xmc-base/${data}"

if [ ${data} == "eurlex-4k" ]; then
	models=(bert)
	ens_method=softmax_average
elif [ ${data} == "eurlex-dc" ]; then
	models=(bert)
	ens_method=softmax_average
elif [ ${data} == "wiki10-31k" ]; then
	models=(bert)
	ens_method=rank_average
elif [ ${data} == "amazoncat-13k" ]; then
	models=(bert roberta xlnet)
	ens_method=softmax_average
elif [ ${data} == "wiki-500k" ]; then
	models=(bert1 bert2 bert3)
	ens_method=sigmoid_average
elif [ ${data} == "amazon-670k" ]; then
	models=(bert1 bert2 bert3)
	ens_method=softmax_average
elif [ ${data} == "amazon-3m" ]; then
	models=(bert1 bert2 bert3)
	ens_method=rank_average
elif [ ${data} == "wos" ]; then
	models=(bert)
	ens_method=softmax_average
elif [ ${data} == "nyt" ]; then
	models=(bert)
	ens_method=softmax_average
elif [ ${data} == "nyt_new" ]; then
	models=(bert)
	ens_method=softmax_average
elif [ ${data} == "scihtc" ]; then
	models=(bert)
	ens_method=softmax_average
elif [ ${data} == "scihtc83" ]; then
	models=(bert)
	ens_method=softmax_average
elif [ ${data} == "mimic3-clean" ]; then
	models=(bert)
	ens_method=softmax_average
elif [ ${data} == "mimic3-clean-digits" ]; then
	models=(roberta)
	ens_method=softmax_average
elif [ ${data} == "uspto-2m" ]; then
	models=(scibert)
	ens_method=softmax_average
elif [ ${data} == "uspto-10k" ]; then
	models=(bert)
	ens_method=softmax_average
elif [ ${data} == "uspto-100k" ]; then
	models=(bert)
	ens_method=softmax_average
else
	echo Unknown dataset $1!
	exit
fi

Preds=""
Tags=""

for mm in "${models[@]}"; do
	for seed in "${seeds[@]}"; do
		bash train_and_predict.sh ${data} ${mm} ${data_dir} ${seed}
		Preds="${Preds} models/${data}/${mm}_${seed}/Pt.npz"
		Tags="${Tags} ${mm}"
	done
done

# Y_tst=${data_dir}/Y.tst.npz # test label matrix

# python ensemble_evaluate.py \
# 	-y ${Y_tst} \
# 	-p ${Preds} \
# 	--tags ${Tags} \
# 	--ens-method ${ens_method} \
#     |& tee models/${data}/ensemble.log

python avg_seeds.py --base_dir models/${data} --name ${models[0]}