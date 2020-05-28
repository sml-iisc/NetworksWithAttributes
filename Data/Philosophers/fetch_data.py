import wikipedia
import numpy as np


# Get the list of philosophers
list_pages = ['List_of_philosophers_(A–C)', 'List_of_philosophers_(D–H)', 'List_of_philosophers_(I–Q)',
              'List_of_philosophers_(R–Z)']
phil_list = []
for list_page in list_pages:
    links = wikipedia.page(list_page, auto_suggest=False).links
    for link in links:
        if 'Index of' in link or 'Dictionary' in link or 'International' in link or \
                'Philosophy' in link or 'Encyclopedia' in link or \
                'philosophy' in link or 'List of' in link:
            continue
        if link not in phil_list:
            phil_list.append(link)


# Save the node id with names
name_to_idx = dict()
idx_to_name = dict()
idx = 0
for phil in phil_list:
    name_to_idx[phil] = idx
    idx_to_name[idx] = phil
    idx += 1
num_nodes = idx

with open('./node_names.txt', 'w') as f:
    for i in range(num_nodes):
        f.write(idx_to_name[i] + '\t' + str(i) + '\n')


# Fetch the links for all philosophers
idx_to_others = dict()
others_to_idx = dict()
links = [[] for _ in range(num_nodes)]
for i in range(num_nodes):
    try:
        links[i] = wikipedia.page(idx_to_name[i], auto_suggest=False).links
        num_links = len(links[i])
        for j in range(num_links):
            if links[i][j] in name_to_idx:
                links[i][j] = name_to_idx[links[i][j]]
                continue
            if links[i][j] in others_to_idx:
                links[i][j] = others_to_idx[links[i][j]]
            else:
                idx_to_others[idx] = links[i][j]
                others_to_idx[links[i][j]] = idx
                links[i][j] = idx
                idx += 1        
        print('Processed: ', i + 1, 'of', num_nodes)
    
    except:
        print('Problem with:', idx_to_name[i])


# Create the network
adj_mat = np.zeros((num_nodes, num_nodes))
for i in range(num_nodes):
    for link in links[i]:
        if link < num_nodes:
            adj_mat[i, link] = 1


# Get node features
node_features = np.zeros((num_nodes, idx - num_nodes))
for i in range(num_nodes):
    for link in links[i]:
        if link >= num_nodes:
            node_features[i, link - num_nodes] = 1


# Save node features
with open('feat_names.txt', 'w') as f:
    for i in range(num_nodes, idx):
        f.write(str(i - num_nodes) + '\t' + idx_to_others[i] + '\n')


# Save the data
np.save('adj_mat.npy', adj_mat)
np.save('node_features.npy', node_features)
