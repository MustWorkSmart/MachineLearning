#%%
##### movie recommendation system, Pytorch, NN, training, Evaluation, Pandas #####
# collaborative filtering

#see this for some background info: 
#https://developers.google.com/machine-learning/recommendation/collaborative/basics

import pandas as pd
from sklearn import model_selection, preprocessing
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from sklearn.metrics import mean_squared_error
import seaborn as sns
#%% data import
df = pd.read_csv("ratings.csv") #Source: https://grouplens.org/datasets/movielens/
df.head(2)
#%%
print(df.describe())
print(f"\nUnique Users: {df.userId.nunique()}, Unique Movies: {df.movieId.nunique()}")

#%% Data Class
class MovieDataset(Dataset):
    def __init__(self, users, movies, ratings):
        self.users = users
        self.movies = movies
        self.ratings = ratings
    # len(movie_dataset)
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        users = self.users[idx] 
        movies = self.movies[idx]
        ratings = self.ratings[idx]
        
        return torch.tensor(users, dtype=torch.long), torch.tensor(movies, dtype=torch.long),torch.tensor(ratings, dtype=torch.long),
       
#%% Model Class
class RecSysModel(nn.Module):
    def __init__(self, n_users, n_movies, n_embeddings = 32):
        super().__init__()        
        self.user_embed = nn.Embedding(n_users, n_embeddings)
        self.movie_embed = nn.Embedding(n_movies, n_embeddings)
        self.out = nn.Linear(n_embeddings * 2, 1)

    def forward(self, users, movies):
        user_embeds = self.user_embed(users)
        movie_embeds = self.movie_embed(movies)
        x = torch.cat([user_embeds, movie_embeds], dim=1)     
        x = self.out(x)       
        return x

#%% encode user and movie id to start from 0 
lbl_user = preprocessing.LabelEncoder()
lbl_movie = preprocessing.LabelEncoder()
df.userId = lbl_user.fit_transform(df.userId.values)
df.movieId = lbl_movie.fit_transform(df.movieId.values)
print(df.describe())
# now, user and movie id's starting from 0, skipped ones no longer skipped
# can check max value to confirm and compare to #unique values above
#%% create train test split
df_train, df_test = model_selection.train_test_split(
    df, test_size=0.2, random_state=42, stratify=df.rating.values
) # stratify -> avoiding large imbalance in the distribution of the target classes

#%% Dataset Instances
train_dataset = MovieDataset(
    users=df_train.userId.values,
    movies=df_train.movieId.values,
    ratings=df_train.rating.values
)

valid_dataset = MovieDataset(
    users=df_test.userId.values,
    movies=df_test.movieId.values,
    ratings=df_test.rating.values
)

#%% Data Loaders
BATCH_SIZE = 4
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True
                          ) 

test_loader = DataLoader(dataset=valid_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True
                          ) 
#%% Model Instance, Optimizer, and Loss Function
print(len(lbl_user.classes_)) #matching df.userId.nunique() above
print(len(lbl_movie.classes_)) #matching df.movieId.nunique() above
model = RecSysModel(
    n_users=len(lbl_user.classes_),
    n_movies=len(lbl_movie.classes_))

optimizer = torch.optim.Adam(model.parameters())  
criterion = nn.MSELoss()

#%% Model Training
NUM_EPOCHS = 5
model.train() 
losses = []
losses_epoch = []
DEBUG = True
for epoch_i in range(NUM_EPOCHS):
    for i, train_item in enumerate(train_loader):
    #for users, movies, ratings in train_loader:
        users, movies, ratings = train_item
        optimizer.zero_grad() # do NOT forget to zero the gradients 1st
        y_pred = model(users, movies) # forward pass    
        if DEBUG:
            print(y_pred.shape) #torch.Size([4, 1])
            print(ratings.shape) #torch.Size([4])
        y_true = ratings.unsqueeze(dim=1).to(torch.float32)
        #torch.unsqueeze adds a new dimension of size one to a tensor at a specified position
        if DEBUG:
            print(y_true.shape) #torch.Size([4, 1])
            DEBUG = False
        loss = criterion(y_pred, y_true)
        losses.append(loss.item()) # Tensor.item() returns a standard Python number
        loss.backward() # backprop
        optimizer.step() # update weights
        if i % 5000 == 0:
            print(f'Epoch {epoch_i+1}/{NUM_EPOCHS}, Step {i+1}/{len(train_loader)},'
            f' Loss: {loss.item():.4f}')
    losses_epoch.append(loss.item())
    if (epoch_i % 2 == 0):
        print(f"At the end of Epoch {epoch_i+1}/{NUM_EPOCHS}, Loss: {loss.item():.4f}")

sns.scatterplot(x=range(len(losses_epoch)), y=losses_epoch)

#%% Model Evaluation 
y_preds = []
y_trues = []

model.eval() 
#model.eval() is a kind of switch for some specific layers/parts of the model that behave differently during training and inference (evaluating) time. For example, Dropouts Layers, BatchNorm Layers etc. You need to turn them off during model evaluation, and .eval() will do it for you.
with torch.no_grad(): # disables gradient calculation, as there's no backprop
    for users, movies, ratings in test_loader: 
        y_true = ratings.detach().numpy().tolist()
        y_pred = model(users, movies).squeeze().detach().numpy().tolist()
        #.squeeze() - Returns a tensor with all specified dimensions of input of size 1 removed.
        #.detach() - Returns a new Tensor, detached from the current graph.
        y_trues.append(y_true)
        y_preds.append(y_pred)

mse = mean_squared_error(y_trues, y_preds) #from sklearn.metrics
print(f"Mean Squared Error: {mse}")

#%% Users and multiple "Prev vs Actual" ratings (due to multiple movies) for each user
user_movie_test = defaultdict(list)
 
with torch.no_grad():
    for users, movies, ratings in test_loader:         
        y_pred = model(users, movies)
        for i in range(len(users)):
            user_id = users[i].item()
            movie_id = movies[i].item() 
            pred_rating = y_pred[i][0].item() #as shown above, y_pred got an extra dimension
            true_rating = ratings[i].item()
            
            print(f"User: {user_id}, Movie: {movie_id}, Pred: {pred_rating}, True: {true_rating}")
            user_movie_test[user_id].append((pred_rating, true_rating))

#%% Precision and Recall - special definitions for Rec Sys -> @k

# Precision@k where k is for the top k predicted ratings
# = #relevant recommendations / #recommended items
# .. measures ability to recommend all relevant items
# .. as recommended items are those top k "predicted" ratings that are also above "threshold"
# .. "relevant" because the corresponding "true" ratings are also above "threshold"

# Recall@k where k is for the top k predicted ratings
# = #relevant recommendations / #all possible relevant items
# .. measures ability to reject non-relevant items
# .. #all possible relevant items are those "true" ratings above "threshold"

precisions = {}
recalls = {}

k = 10
thres = 3.5

for uid, user_ratings in user_movie_test.items():
    # Sort user ratings by "pred_rating", reverse=True to get the top "k" highest rated ones
    user_ratings.sort(key=lambda x: x[0], reverse=True)

    # count of relevant items
    n_rel = sum((rating_true >= thres) for (_, rating_true) in user_ratings)

    # count recommended items that are predicted relevent and within topk
    n_rec_k = sum((rating_pred >= thres) for (rating_pred, _) in user_ratings[:k])

    # count recommended AND relevant item 
    n_rel_and_rec_k = sum(
        ((rating_true >= thres) and (rating_pred >= thres))
        for (rating_pred, rating_true) in user_ratings[:k]
    )

    print(f"uid {uid},  n_rel {n_rel}, n_rec_k {n_rec_k}, n_rel_and_rec_k {n_rel_and_rec_k}")

    precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0 # taking care of "div by 0" case

    recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

print(f"Precision @ {k}: {sum(precisions.values()) / len(precisions)}")

print(f"Recall @ {k} : {sum(recalls.values()) / len(recalls)}")
# %%
