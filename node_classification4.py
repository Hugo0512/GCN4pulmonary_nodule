import numpy as np
import torch
import torch_geometric.datasets as datasets
import torch_geometric.data as data
import torch_geometric.transforms as transforms
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import SAGEConv
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

minmaxscaler=MinMaxScaler()
train_x=np.loadtxt('./node2/merge_node_feature.txt')
train_x=minmaxscaler.fit_transform(train_x)
train_edge=np.loadtxt('./node2/merge_adjacent_info_30.txt')
train_edge1=train_edge[0,:].astype(int)
train_edge2=train_edge[1,:].astype(int)
train_edge_attribute=np.hstack((train_edge[2,:],train_edge[2,:]))
# train_edge_attribute=np.expand_dims(train_edge_attribute,1)
# train_edge=np.concatenate((train_edge1,train_edge2,train_edge3),axis=0)
new_train_edge1=np.hstack((train_edge1,train_edge2))
new_train_edge2=np.hstack((train_edge2,train_edge1))
new_train_edge=np.vstack((new_train_edge1,new_train_edge2))
train_label=np.loadtxt('./node2/merge_label.txt')
edge_attribute=torch.tensor(train_edge_attribute)
train_mask=np.loadtxt('./node2/train_mask.txt')
train_mask=np.array(train_mask,dtype=bool)
trainmask=torch.tensor(train_mask)
# print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{")
# print(trainmask.dtype)
test_mask=np.loadtxt('./node2/test_mask.txt')
test_mask=np.array(test_mask,dtype=bool)
testmask=torch.tensor(test_mask)
# print(testmask.dtype)
# print(train_x.shape,edge_attribute.shape,new_train_edge.shape)
# train_graph = data.Data(x=train_x, edge_index=new_train_edge, edge_attr=train_edge_attribute, y=train_label)
test_graph = data.Data(x=torch.tensor(train_x).to(torch.float32), edge_attr=edge_attribute,edge_index=torch.tensor(new_train_edge), y=torch.tensor(train_label).to(torch.long),train_mask=trainmask,test_mask=testmask)
# kwargs = {'batch_size': 10, 'num_workers': 2, 'persistent_workers': True,'drop_last':False}
# train_loader = NeighborLoader(data=train_graph, input_nodes=None,num_neighbors=[5, 10], shuffle=True,replace=False,directed=True,transform=None, **kwargs)

print("????????????????????????????????")
print(type(test_graph.x),type(test_graph.y),type(test_graph.edge_attr),type(test_graph.edge_index),type(test_graph.train_mask),type(test_graph.test_mask))
print(test_graph.x.shape,test_graph.y.shape,test_graph.edge_attr.shape,test_graph.edge_index.shape,test_graph.train_mask.shape,test_graph.test_mask.shape)



train_x=np.loadtxt('./node/train_node_feature.txt')
train_x=minmaxscaler.fit_transform(train_x)
train_edge=np.loadtxt('./node/train_adjacent_info_30.txt')
train_edge1=train_edge[0,:].astype(int)
train_edge2=train_edge[1,:].astype(int)
train_edge_attribute=np.hstack((train_edge[2,:],train_edge[2,:]))
# train_edge_attribute=np.expand_dims(train_edge_attribute,1)
# train_edge=np.concatenate((train_edge1,train_edge2,train_edge3),axis=0)
new_train_edge1=np.hstack((train_edge1,train_edge2))
new_train_edge2=np.hstack((train_edge2,train_edge1))
new_train_edge=np.vstack((new_train_edge1,new_train_edge2))
train_label=np.loadtxt('./node/train_label.txt')
edge_attribute=torch.tensor(train_edge_attribute)
train_mask=np.loadtxt('./node1/train_mask.txt')
position=0;
for index in range(train_mask.shape[0]):
    if train_mask[index]==0:
        position=index
        break
train_mask=train_mask[0:position]
print(np.sum(train_mask,axis=0))
train_mask=np.array(train_mask,dtype=bool)
trainmask=torch.tensor(train_mask)
# print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{")
# print(trainmask.dtype)
# print(testmask.dtype)
# print(train_x.shape,edge_attribute.shape,new_train_edge.shape)
# train_graph = data.Data(x=train_x, edge_index=new_train_edge, edge_attr=train_edge_attribute, y=train_label)
train_graph = data.Data(x=torch.tensor(train_x).to(torch.float32), edge_attr=edge_attribute,edge_index=torch.tensor(new_train_edge), y=torch.tensor(train_label).to(torch.long),train_mask=trainmask)
# kwargs = {'batch_size': 10, 'num_workers': 2, 'persistent_workers': True,'drop_last':False}
# train_loader = NeighborLoader(data=train_graph, input_nodes=None,num_neighbors=[5, 10], shuffle=True,replace=False,directed=True,transform=None, **kwargs)

print("????????????????????????????????")

print(type(train_graph.x),type(train_graph.y),type(train_graph.edge_attr),type(train_graph.edge_index),type(train_graph.train_mask))
print(train_graph.x.shape,train_graph.y.shape,train_graph.edge_attr.shape,train_graph.edge_index.shape,train_graph.train_mask.shape)





class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels)) # 注意使用modulelist
        self.convs.append(SAGEConv(hidden_channels, out_channels))
# 第一个gnn层的in_channels表示的是初始的node的原始features的size，后面的hidden channels可以随便设，其实就类似keras
# 中两个叠加的dense层，out channels表示最终输出的维度，这个不一定要根据任务层来设置，可以设置大一点然后后面再接一些常规
# 的dnn结构

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

def inference(model,train_graph,prob_filename,original_prob_filename):


    # Compute representations of nodes layer by layer, using *all*
    # available edges. This leads to faster computation in contrast to
    # immediately computing the final representations of each batch:
    # xs_previous = {}
    # for i, conv in enumerate(self.convs):
    #     xs = []
    #     for batch in subgraph_loader:
    #         # print("==================================",i)
    #         # print(batch.__dict__.items())
    #         # x = x_all[batch.n_id.to(x_all.device)].to(device)
    #         x=batch.x.to(device)
    #         x = conv(x, batch.edge_index.to(device))
    #         if i < len(self.convs) - 1:
    #             x = x.relu_()
    #         xs.append(x[:batch.batch_size].cpu())
    #         pbar.update(batch.batch_size)
    #     x_all = torch.cat(xs, dim=0)
    # pbar.close()
    # return x_all
    # for i, conv in enumerate(self.convs):
    xs = []
    model.eval()
    confusion_matrix=np.zeros((2,2))
    with torch.no_grad():
        # print("==================================",i)
        # print(batch.__dict__.items())
        # x = x_all[batch.n_id.to(x_all.device)].to(device)
        prob = model(train_graph.x.to(device), train_graph.edge_index.to(device))
        prob2 = prob
        prob=F.softmax(prob,dim=1)[train_graph.train_mask]
        prob1 = prob2[train_graph.train_mask, :]
        prob1_np = prob1.cpu().numpy()
        val_pred_prob = prob.cpu().numpy()
        pred_result = np.argmax(val_pred_prob, axis=1)
        pred_result_np=pred_result
        for index in range(pred_result_np.shape[0]):
            col_position=pred_result_np[index]
            row_position=train_graph.y[train_graph.train_mask][index]
            confusion_matrix[row_position,col_position]=confusion_matrix[row_position,col_position]+1
        xs.append(prob.cpu())

        x_all = torch.cat(xs, dim=0)
    x_all_np=np.array(x_all)
    np.savetxt(prob_filename,x_all_np)
    print(confusion_matrix)
    np.savetxt(original_prob_filename, prob1_np)
    return x_all_np,prob1_np




def inference1(model,train_graph,prob_filename,original_prob_filename):


    # Compute representations of nodes layer by layer, using *all*
    # available edges. This leads to faster computation in contrast to
    # immediately computing the final representations of each batch:
    # xs_previous = {}
    # for i, conv in enumerate(self.convs):
    #     xs = []
    #     for batch in subgraph_loader:
    #         # print("==================================",i)
    #         # print(batch.__dict__.items())
    #         # x = x_all[batch.n_id.to(x_all.device)].to(device)
    #         x=batch.x.to(device)
    #         x = conv(x, batch.edge_index.to(device))
    #         if i < len(self.convs) - 1:
    #             x = x.relu_()
    #         xs.append(x[:batch.batch_size].cpu())
    #         pbar.update(batch.batch_size)
    #     x_all = torch.cat(xs, dim=0)
    # pbar.close()
    # return x_all
    # for i, conv in enumerate(self.convs):
    xs = []
    model.eval()
    confusion_matrix=np.zeros((2,2))
    with torch.no_grad():
        # print("==================================",i)
        # print(batch.__dict__.items())
        # x = x_all[batch.n_id.to(x_all.device)].to(device)
        prob = model(train_graph.x.to(device), train_graph.edge_index.to(device))
        prob2=prob
        prob=F.softmax(prob,dim=1)[train_graph.test_mask]
        prob1 =prob2[train_graph.test_mask,:]
        prob1_np=prob1.cpu().numpy()
        val_pred_prob = prob.cpu().numpy()
        pred_result = np.argmax(val_pred_prob, axis=1)
        pred_result_np=pred_result
        for index in range(pred_result_np.shape[0]):
            col_position=pred_result_np[index]
            row_position=train_graph.y[train_graph.test_mask][index]
            confusion_matrix[row_position,col_position]=confusion_matrix[row_position,col_position]+1
        xs.append(prob.cpu())

        x_all = torch.cat(xs, dim=0)
    x_all_np=np.array(x_all)
    np.savetxt(prob_filename,x_all_np)
    print(confusion_matrix)
    np.savetxt(original_prob_filename,prob1_np)
    return x_all_np,prob1_np

def train():
    model.train()



    total_loss = total_correct = total_examples = 0


    optimizer.zero_grad()
    y = train_graph.y.to(device)
    y_hat = model(train_graph.x.to(device), train_graph.edge_index.to(device)).to(device) # 这个地方的设定也是醉了
    pred_prob = F.softmax(y_hat, dim=1)
    loss = criterion(y_hat[train_graph.train_mask], y[train_graph.train_mask])

    loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        y_hat = model(train_graph.x.to(device), train_graph.edge_index.to(device)).to(device)  # 这个地方的设定也是醉了
        pred_prob = F.softmax(y_hat, dim=1)
        # print("+++++++++++++++++++++++++++++++++++++")
        # print(pred_prob.size())

        total_correct = (pred_prob.argmax(dim=1) == y)[train_graph.train_mask].float().sum()
        # print(total_correct)
        total_examples=train_graph.x.shape[0]
        # print(total_examples)

    return loss, total_correct / total_examples


# ,label_smoothing=0.1,
model = SAGE(in_channels=train_graph.x.shape[1], hidden_channels=128, out_channels=train_graph.y.unique().shape[0]).to(device)
criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor([1,0.3]).to(torch.float32)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# for batch in subgraph_loader:
#     print(batch)
# print("==============================================")

def test():
    val_total_loss = val_total_correct = val_total_examples = 0
    model.eval()
    with torch.no_grad():
        y_hat = model(test_graph.x.to(device), test_graph.edge_index.to(device))
        val_pred_prob=F.softmax(y_hat,dim=1)
        val_total_loss=criterion(y_hat[test_graph.test_mask], test_graph.y[test_graph.test_mask].to(device))
        val_total_correct += (val_pred_prob.argmax(dim=1) == test_graph.y.to(device))[test_graph.test_mask].float().sum()

    return val_total_loss,val_total_correct/test_graph.y[test_graph.test_mask].shape[0]




best_val_accuracy=0;
for epoch in range(1, 31):

    loss, acc = train()
    print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
    val_acc, test_acc = test()
    print(f'Epoch: {epoch:02d}, Val: {val_acc:.4f}, '
          f'Test: {test_acc:.4f}')
    # print("ttttttttttttttttttttttttttttttttttttttt")
    # print(best_val_accuracy)
    if test_acc>best_val_accuracy:
        # print("savesavesavesavesavesavesavesavesavesavesavesavesavesave")
        torch.save(model, 'best_model_30.pth')
        best_val_accuracy=test_acc.cpu().item()
torch.save(model,'model_30.pth')
print("==============================================")
print(best_val_accuracy)
model=torch.load('best_model_30.pth')
inference(model,train_graph,'prob_train_30.txt','logit_train_30.txt')
inference1(model,test_graph,'prob_test_30.txt','logit_test_30.txt')



