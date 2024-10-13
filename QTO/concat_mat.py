import torch
import os
fraction = 10
thrshd = 0.01
device = 'cuda'
dataset_name = 'FB15k'
filename_list = os.listdir(f'./neural_adj_plm/{dataset_name}')
relation_embeddings = []
base_file_name = '_'.join(filename_list[0].split('_')[:3])
for i in range(len(filename_list)):
    filename = base_file_name + '_{}.pt'.format(str(i))
    print(filename)
    file_path = os.path.join(f'./neural_adj_plm/{dataset_name}', filename)
    relation_embeddings += torch.load(file_path, map_location=device)
    os.remove(file_path)
    # print()
torch.save(relation_embeddings, './neural_adj_plm/{}.pt'.format(base_file_name))
print()
