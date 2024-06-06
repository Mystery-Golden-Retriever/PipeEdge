import os
import pandas as pd
import argparse
import sys 
sys.path.append("..") 
import model_cfg

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Pipeline Parallelism Runtime",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Model options
	parser.add_argument("-m", "--model-name", type=str, default="google/vit-base-patch16-224",
                        choices=model_cfg.get_model_names(),
                        help="the neural network model for loading")
	parser.add_argument("-o", "--output-dir", type=str, default="/home1/haonanwa/projects/EdgePipe/results")
	parser.add_argument("-n", "--name-suffix", type=str, default="")
	args = parser.parse_args()

	table = {}
	result_dir = os.path.join(args.output_dir, args.model_name.split('/')[1])
	file_list = os.listdir(result_dir)
	for file_name in file_list:
		pt,bit = file_name.split('.')[0].split('_')[1:]
		pt = int(pt.split(',')[1])
		bit = int(bit)
		with open(os.path.join(result_dir, file_name)) as f:
			lines = f.readlines()
			acc = lines[-1].strip()
		table.setdefault(pt, {})[bit] = float(acc)

	df = pd.DataFrame.from_dict(data=table)
	if args.name_suffix:
		excel_name = args.model_name.split('/')[1]+"_acc_vs_pt_"+args.name_suffix+".xlsx"
	else:
		excel_name = args.model_name.split('/')[1]+"_acc_vs_pt.xlsx"	
	df.to_excel(excel_name)
